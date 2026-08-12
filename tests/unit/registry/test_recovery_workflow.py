"""Post-evaluation recovery tests that never load the held-out test dataset."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from mlflow.entities import Run

from ticket_router.config import Settings
from ticket_router.registry import recover_final
from ticket_router.registry.config import FinalModelConfig
from ticket_router.registry.recover_final import FinalRecoveryError
from ticket_router.registry.service import ModelRegistryService, RegisteredVersion


class FixturePredictor:
    def predict(self, values: Sequence[str]) -> object:
        return np.asarray(["Billing"] * len(values), dtype=object)


class RecoveryClient:
    def __init__(self) -> None:
        self.tags: list[tuple[str, str, str]] = []
        self.terminated: tuple[str, str] | None = None

    def search_logged_models(self, **_: object) -> list[SimpleNamespace]:
        return [SimpleNamespace(status="READY", model_uri="models:/logged-fixture")]

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        self.tags.append((run_id, key, value))

    def set_terminated(self, run_id: str, *, status: str) -> None:
        self.terminated = (run_id, status)


class RecoveryRegistry:
    def __init__(self) -> None:
        self.registered = RegisteredVersion(
            "ticket-router",
            "4",
            "f" * 32,
            "models:/logged-fixture",
        )
        self.tag_updates: list[dict[str, str]] = []

    def resolve_alias(self, *, name: str, alias: str) -> None:
        del name, alias
        return None

    def register_candidate(self, **_: object) -> RegisteredVersion:
        return self.registered

    def set_model_version_tags(
        self,
        *,
        name: str,
        version: str,
        tags: dict[str, str],
    ) -> None:
        del name, version
        self.tag_updates.append(tags)


def _write_recovery_evidence(root: Path) -> tuple[Path, Path, str]:
    evaluation_run_id = "final-fixture"
    reports = root / "reports"
    reports.mkdir()
    (reports / "test_access_audit.json").write_text(
        json.dumps(
            {
                "status": "authorized_and_opened",
                "evaluation_run_id": evaluation_run_id,
                "test_access_timestamp_utc": "2026-08-11T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    models = root / "models"
    artifact_directory = models / evaluation_run_id / "final_model"
    artifact_directory.mkdir(parents=True)
    (artifact_directory / "metrics.json").write_text(
        json.dumps({"macro_f1": 0.72, "weighted_f1": 0.74}),
        encoding="utf-8",
    )
    (artifact_directory / "latency_distribution.json").write_text(
        json.dumps({"median_milliseconds_per_record": 0.2}),
        encoding="utf-8",
    )
    (artifact_directory / "training_metadata.json").write_text(
        json.dumps(
            {
                "split_manifest_sha256": "a" * 64,
                "combined_training_data_sha256": "b" * 64,
                "test_data_sha256": "c" * 64,
                "final_configuration_sha256": "d" * 64,
                "git_commit": None,
                "test_evaluated": True,
            }
        ),
        encoding="utf-8",
    )
    (artifact_directory / "per_class_metrics.csv").write_text(
        "class,recall\nBilling,0.61\nTechnical,0.55\n",
        encoding="utf-8",
    )
    return reports, models, evaluation_run_id


def test_recovery_reuses_logged_model_and_completes_audit_without_test_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports, models, evaluation_run_id = _write_recovery_evidence(tmp_path)
    client = RecoveryClient()
    registry = RecoveryRegistry()
    run = cast(
        Run,
        SimpleNamespace(info=SimpleNamespace(run_id="f" * 32, experiment_id="9")),
    )
    monkeypatch.setattr(
        recover_final,
        "configure_experiment_tracking",
        lambda **_: SimpleNamespace(resolved_uri="file:///fixture-mlruns"),
    )
    monkeypatch.setattr(
        recover_final,
        "mlflow",
        SimpleNamespace(
            MlflowClient=lambda: client,
            sklearn=SimpleNamespace(load_model=lambda _: FixturePredictor()),
        ),
    )
    monkeypatch.setattr(recover_final, "_find_final_run", lambda *args: run)
    monkeypatch.setattr(
        recover_final,
        "get_model_info",
        lambda _: SimpleNamespace(signature=object()),
    )
    monkeypatch.setattr(recover_final, "signature_matches_text_api", lambda *_, **__: True)
    monkeypatch.setattr(
        recover_final,
        "ModelRegistryService",
        lambda: cast(ModelRegistryService, registry),
    )

    result = recover_final.recover_final_registration(
        settings=Settings.load(env_file=None),
        final_config=FinalModelConfig.load(),
        reports_dir=reports,
        model_artifacts_dir=models,
        project_root=tmp_path,
    )

    audit = json.loads((reports / "test_access_audit.json").read_text(encoding="utf-8"))
    summary_path = reports / evaluation_run_id / "final_evaluation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert result["recovered_without_test_access"] is True
    assert summary["registered_model_version"] == "4"
    assert audit["status"] == "completed_after_post_evaluation_recovery"
    assert audit["recovery_reloaded_test_data"] is False
    assert client.terminated == ("f" * 32, "FINISHED")
    assert registry.tag_updates[0]["promotion_gates_passed"] == "true"


def test_recovery_helpers_reject_ambiguous_runs_and_non_numeric_metrics(
    tmp_path: Path,
) -> None:
    class MissingExperimentClient:
        def get_experiment_by_name(self, name: str) -> None:
            del name
            return None

    with pytest.raises(FinalRecoveryError, match="does not exist"):
        recover_final._find_final_run(
            cast(Any, MissingExperimentClient()),
            FinalModelConfig.load(),
            "final-fixture",
        )
    with pytest.raises(FinalRecoveryError, match="numeric metric"):
        recover_final._numeric_mapping({"macro_f1": "not-numeric"})

    invalid = tmp_path / "array.json"
    invalid.write_text("[]", encoding="utf-8")
    with pytest.raises(FinalRecoveryError, match="JSON object"):
        recover_final._read_json(invalid)
