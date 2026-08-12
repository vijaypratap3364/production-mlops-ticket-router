"""Tests for Prefect task adapters and their resource/lineage boundaries."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import SecretStr

import ticket_router.orchestration.tasks as tasks
from ticket_router.config import Settings
from ticket_router.data.split_manifest import SplitManifest
from ticket_router.db.contracts import RetrainingRun
from ticket_router.hashing import sha256_file
from ticket_router.monitoring.config import MonitoringConfig
from ticket_router.monitoring.service import MonitoringExecution
from ticket_router.orchestration.config import OrchestrationConfig
from ticket_router.orchestration.contracts import RetrainingDecision


class FixtureLogger:
    def info(self, message: str, *args: object) -> None:
        del message, args


def test_ingestion_task_adapters_delegate_with_scoped_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings.load(env_file=None)
    observed: dict[str, dict[str, object]] = {}
    monkeypatch.setattr(tasks, "get_run_logger", FixtureLogger)
    monkeypatch.setattr(tasks, "configuration_hash", lambda _: "a" * 64)

    def download(**kwargs: object) -> SimpleNamespace:
        observed["download"] = kwargs
        return SimpleNamespace(row_count=3)

    def normalize(**kwargs: object) -> SimpleNamespace:
        observed["normalize"] = kwargs
        return SimpleNamespace(output_row_count=2)

    def analyze(**kwargs: object) -> dict[str, object]:
        observed["analyze"] = kwargs
        return {"selected_classes": ["Billing"]}

    def prepare(**kwargs: object) -> None:
        observed["prepare"] = kwargs

    monkeypatch.setattr(tasks, "download_dataset", download)
    monkeypatch.setattr(tasks, "normalize_dataset", normalize)
    monkeypatch.setattr(tasks, "run_analysis", analyze)
    monkeypatch.setattr(tasks, "prepare_dataset", prepare)

    raw_manifest = tasks.download_data_task.fn(settings, force=False, project_root=tmp_path)
    normalization_manifest = tasks.normalize_data_task.fn(
        settings,
        force=True,
        project_root=tmp_path,
    )
    analysis = tasks.analyze_data_task.fn(settings, project_root=tmp_path)
    split_manifest = tasks.prepare_splits_task.fn(settings, force=False, project_root=tmp_path)

    assert raw_manifest.endswith("data/raw/data_manifest.json")
    assert normalization_manifest.endswith("data/interim/normalization_manifest.json")
    assert split_manifest.endswith("data/processed/split_manifest.json")
    assert analysis == {"selected_classes": ["Billing"]}
    assert observed["download"]["force"] is False
    assert observed["normalize"]["force"] is True
    assert observed["prepare"]["reference_dir"] == tmp_path / "data/reference"


def test_candidate_manifest_verification_checks_feature_contract_and_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    for split in ("train", "validation"):
        (processed / f"{split}.parquet").write_bytes(split.encode("utf-8"))
    manifest_path = processed / "split_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    manifest = cast(
        SplitManifest,
        SimpleNamespace(
            model_feature_columns=("model_text",),
            output_files={
                split: SimpleNamespace(sha256=sha256_file(processed / f"{split}.parquet"))
                for split in ("train", "validation")
            },
        ),
    )
    monkeypatch.setattr(
        tasks,
        "SplitManifest",
        SimpleNamespace(read=lambda _: manifest),
    )

    assert tasks.verify_data_manifests_task.fn(processed, manifest_path) == sha256_file(
        manifest_path
    )

    invalid = cast(SplitManifest, SimpleNamespace(model_feature_columns=("body",)))
    monkeypatch.setattr(
        tasks,
        "SplitManifest",
        SimpleNamespace(read=lambda _: invalid),
    )
    with pytest.raises(ValueError, match="only model_text"):
        tasks.verify_data_manifests_task.fn(processed, manifest_path)


def test_retraining_decision_task_rejects_non_object_and_delegates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "monitoring.json"
    path.write_text("[]", encoding="utf-8")
    config = OrchestrationConfig.load()
    with pytest.raises(ValueError, match="JSON object"):
        tasks.retraining_decision_task.fn(
            monitoring_summary_path=path,
            recent_statuses=(),
            config=config,
            manual_trigger=False,
        )

    path.write_text(json.dumps({"status": "healthy", "feedback_count": 0}), encoding="utf-8")
    expected = cast(RetrainingDecision, SimpleNamespace(should_retrain=False))
    monkeypatch.setattr(tasks, "evaluate_retraining_conditions", lambda **_: expected)
    result = tasks.retraining_decision_task.fn(
        monitoring_summary_path=path,
        recent_statuses=("healthy",),
        config=config,
        manual_trigger=True,
    )
    assert result is expected


def test_database_tasks_close_resources_and_allow_unconfigured_fixture_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings.load(env_file=None)
    assert tasks.recent_monitoring_statuses_task.fn(settings, limit=3) == ()
    run_record = cast(RetrainingRun, SimpleNamespace())
    assert tasks.store_retraining_run_task.fn(settings, run_record) is None
    with pytest.raises(ValueError, match="DATABASE_URL"):
        tasks.execute_monitoring_task.fn(
            settings=settings,
            config=MonitoringConfig.load(),
            start=datetime(2026, 8, 1, tzinfo=UTC),
            end=datetime(2026, 8, 2, tzinfo=UTC),
            output_root=tmp_path,
            run_id=None,
        )

    observed: dict[str, object] = {}

    class FixtureManager:
        session_factory = object()

        def __init__(self, database_url: str) -> None:
            observed["url"] = database_url

        def connect(self) -> None:
            observed["connected"] = True

        def close(self) -> None:
            observed["closed"] = True

    configured = settings.model_copy(
        update={"database_url": SecretStr("sqlite:///orchestration-fixture.db")}
    )
    execution = cast(MonitoringExecution, SimpleNamespace(run_id="monitoring-fixture"))
    monkeypatch.setattr(tasks, "DatabaseSessionManager", FixtureManager)
    monkeypatch.setattr(tasks, "SQLAlchemyMonitoringDataRepository", lambda _: object())
    monkeypatch.setattr(tasks, "SQLAlchemyMonitoringRunRepository", lambda _: object())
    monkeypatch.setattr(tasks, "execute_monitoring_run", lambda **_: execution)

    result = tasks.execute_monitoring_task.fn(
        settings=configured,
        config=MonitoringConfig.load(),
        start=datetime(2026, 8, 1, tzinfo=UTC),
        end=datetime(2026, 8, 2, tzinfo=UTC),
        output_root=tmp_path,
        run_id="monitoring-fixture",
    )

    assert result is execution
    assert observed == {
        "url": "sqlite:///orchestration-fixture.db",
        "connected": True,
        "closed": True,
    }
