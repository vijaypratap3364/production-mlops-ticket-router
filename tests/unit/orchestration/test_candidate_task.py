import json
from pathlib import Path
from typing import cast

import pytest

from ticket_router.modeling.train_candidates import CandidateExperimentResult
from ticket_router.orchestration import tasks
from ticket_router.orchestration.candidate import read_candidate_workflow_result
from ticket_router.orchestration.contracts import CandidateWorkflowResult
from ticket_router.registry.config import FinalModelConfig


def test_candidate_task_delegates_registration_without_promotion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = CandidateWorkflowResult(
        workflow_run_id="workflow-1",
        selected_candidate="fixture",
        mlflow_run_id="a" * 32,
        registered_model_name="ticket-router",
        candidate_model_version="2",
        promotion_gates_passed=True,
        champion_promotion_performed=False,
        summary_path="summary.json",
    )
    observed: dict[str, object] = {}

    def fake_register(**kwargs: object) -> CandidateWorkflowResult:
        observed.update(kwargs)
        return expected

    monkeypatch.setattr(tasks, "register_and_gate_candidate", fake_register)
    experiment = cast(CandidateExperimentResult, object())
    config = FinalModelConfig.load()

    result = tasks.register_candidate_task.fn(
        experiment=experiment,
        final_config=config,
        workflow_run_id="workflow-1",
        dataset_manifest_sha256="b" * 64,
        orchestration_configuration_sha256="c" * 64,
        summary_path=Path("summary.json"),
    )

    assert result == expected
    assert observed["experiment"] is experiment
    assert result.champion_promotion_performed is False


def test_completed_candidate_reuse_requires_matching_lineage(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    payload = {
        "workflow_run_id": "workflow-1",
        "selected_candidate": "fixture",
        "mlflow_run_id": "a" * 32,
        "registered_model_name": "ticket-router",
        "candidate_model_version": "2",
        "promotion_gates_passed": True,
        "champion_promotion_performed": False,
        "summary_path": path.as_posix(),
        "dataset_manifest_sha256": "b" * 64,
        "orchestration_configuration_sha256": "c" * 64,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = read_candidate_workflow_result(
        path,
        expected_dataset_manifest_sha256="b" * 64,
        expected_orchestration_configuration_sha256="c" * 64,
    )

    assert result.workflow_run_id == "workflow-1"
    with pytest.raises(ValueError, match="different dataset"):
        read_candidate_workflow_result(
            path,
            expected_dataset_manifest_sha256="d" * 64,
            expected_orchestration_configuration_sha256="c" * 64,
        )
