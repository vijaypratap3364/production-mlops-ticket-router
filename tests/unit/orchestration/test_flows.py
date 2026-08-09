from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest

from ticket_router.db.contracts import RetrainingRun
from ticket_router.monitoring.service import MonitoringExecution
from ticket_router.orchestration import flows
from ticket_router.orchestration.contracts import (
    CandidateWorkflowResult,
    RetrainingDecision,
)
from ticket_router.orchestration.tasks import download_data_task


class StubTask:
    def __init__(self, name: str, calls: list[tuple[str, dict[str, object]]]) -> None:
        self.name = name
        self.calls = calls

    def with_options(self, **options: object) -> StubTask:
        self.calls.append((f"{self.name}:options", options))
        return self

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.calls.append((self.name, kwargs))
        if self.name == "validate":
            return {"valid_rows": 1, "removed_rows": 0}
        if self.name == "analyze":
            return {"selected_classes": ["queue"]}
        return f"{self.name}-result"


class ReturningTask:
    def __init__(self, result: object, calls: list[object] | None = None) -> None:
        self.result = result
        self.calls = calls

    def with_options(self, **options: object) -> ReturningTask:
        return self

    def __call__(self, *args: object, **kwargs: object) -> object:
        if self.calls is not None:
            self.calls.append(args[1] if len(args) > 1 else kwargs)
        return self.result


def test_ingest_flow_composes_tasks_in_order_with_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    for attribute, name in (
        ("download_data_task", "download"),
        ("normalize_data_task", "normalize"),
        ("validate_data_task", "validate"),
        ("analyze_data_task", "analyze"),
        ("prepare_splits_task", "prepare"),
    ):
        monkeypatch.setattr(flows, attribute, StubTask(name, calls))

    result = flows.ingest_data_flow.fn(project_root=str(Path.cwd()))

    assert [name for name, _ in calls if not name.endswith(":options")] == [
        "download",
        "normalize",
        "validate",
        "analyze",
        "prepare",
    ]
    retry_options = [options for name, options in calls if name.endswith(":options")]
    assert retry_options == [
        {"retries": 2, "retry_delay_seconds": 10},
        {"retries": 2, "retry_delay_seconds": 10},
    ]
    assert result["split_manifest"] == "prepare-result"


def test_orchestration_has_no_automatic_champion_promotion() -> None:
    source = inspect.getsource(flows.conditional_retraining_flow.fn)

    assert "promote_candidate" not in source
    assert "champion_alias" not in source


def test_temporary_network_task_is_named_and_uncached() -> None:
    assert download_data_task.name == "download pinned support-ticket data"
    assert download_data_task.persist_result is False


def test_conditional_flow_stops_after_candidate_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identifier = str(uuid4())
    monitoring = cast(
        MonitoringExecution,
        SimpleNamespace(
            decision=SimpleNamespace(status="critical"),
            summary_path=Path("monitoring-summary.json"),
        ),
    )
    decision = RetrainingDecision(True, "automatic", ("sustained drift",), 150, 2)
    candidate = CandidateWorkflowResult(
        workflow_run_id="retrain-fixture",
        selected_candidate="fixture",
        mlflow_run_id="a" * 32,
        registered_model_name="ticket-router",
        candidate_model_version="3",
        promotion_gates_passed=True,
        champion_promotion_performed=False,
        summary_path="candidate.json",
    )
    stored: list[object] = []
    monkeypatch.setattr(flows, "monitoring_flow", lambda **_: monitoring)
    monkeypatch.setattr(flows, "recent_monitoring_statuses_task", ReturningTask(("critical",)))
    monkeypatch.setattr(flows, "retraining_decision_task", ReturningTask(decision))
    monkeypatch.setattr(flows, "store_retraining_run_task", ReturningTask(None, stored))
    monkeypatch.setattr(
        flows,
        "prepare_retraining_dataset_task",
        ReturningTask(Path("runs/retrain-fixture/split_manifest.json").as_posix()),
    )
    monkeypatch.setattr(flows, "train_candidate_flow", lambda **_: candidate)
    monkeypatch.setattr(flows, "_write_conditional_summary", lambda *_: None)

    result = flows.conditional_retraining_flow.fn(
        project_root=str(Path.cwd()),
        run_id=identifier,
    )

    assert result.status == "candidate_registered"
    assert result.candidate == candidate
    assert result.champion_promotion_performed is False
    assert [cast(RetrainingRun, run).status for run in stored] == [
        "preparing_data",
        "candidate_registered",
    ]


def test_conditional_flow_skips_training_when_feedback_is_insufficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitoring = cast(
        MonitoringExecution,
        SimpleNamespace(
            decision=SimpleNamespace(status="critical"),
            summary_path=Path("monitoring-summary.json"),
        ),
    )
    decision = RetrainingDecision(False, "none", ("too few labels",), 10, 2)
    monkeypatch.setattr(flows, "monitoring_flow", lambda **_: monitoring)
    monkeypatch.setattr(flows, "recent_monitoring_statuses_task", ReturningTask(("critical",)))
    monkeypatch.setattr(flows, "retraining_decision_task", ReturningTask(decision))
    monkeypatch.setattr(flows, "_write_conditional_summary", lambda *_: None)
    monkeypatch.setattr(
        flows,
        "train_candidate_flow",
        lambda **_: pytest.fail("training must not run with insufficient feedback"),
    )

    result = flows.conditional_retraining_flow.fn(project_root=str(Path.cwd()))

    assert result.status == "insufficient_feedback"
    assert result.candidate is None
