"""One-time Stage 7 lifecycle on deterministic synthetic splits."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import polars as pl
import pytest

from ticket_router.config import Settings
from ticket_router.data.split_manifest import OutputFileManifest, SplitManifest
from ticket_router.hashing import sha256_file
from ticket_router.modeling.mlflow_tracking import LoggedCandidate, TrackingResolution
from ticket_router.registry import evaluate_final
from ticket_router.registry.config import FinalModelConfig
from ticket_router.registry.service import ModelRegistryService, RegisteredVersion


class FinalRegistry:
    def __init__(self) -> None:
        self.version = RegisteredVersion("ticket-router", "8", "e" * 32, "runs:/fixture/model")
        self.tags: list[dict[str, str]] = []

    def resolve_alias(self, *, name: str, alias: str) -> None:
        del name, alias
        return None

    def register_candidate(self, **_: object) -> RegisteredVersion:
        return self.version

    def set_model_version_tags(
        self,
        *,
        name: str,
        version: str,
        tags: dict[str, str],
    ) -> None:
        del name, version
        self.tags.append(tags)


class FinalTrackingClient:
    def __init__(self) -> None:
        self.tags: list[tuple[str, str, str]] = []

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        self.tags.append((run_id, key, value))


def test_final_fixture_is_evaluated_once_registered_and_never_promoted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings.load(env_file=None)
    processed = tmp_path / "processed"
    processed.mkdir()
    texts, labels = _tiny_text_classification_data()
    split_indices = {
        "train": [index for index in range(len(texts)) if index % 12 < 8],
        "validation": [index for index in range(len(texts)) if 8 <= index % 12 < 10],
        "test": [index for index in range(len(texts)) if index % 12 >= 10],
    }
    paths: dict[str, Path] = {}
    for split, indices in split_indices.items():
        path = processed / f"{split}.parquet"
        pl.DataFrame(
            {
                "model_text": [texts[index] for index in indices],
                "queue": [labels[index] for index in indices],
            }
        ).write_parquet(path)
        paths[split] = path
    manifest_path = processed / "split_manifest.json"
    _manifest(settings, paths).write(manifest_path)
    registry = FinalRegistry()
    client = FinalTrackingClient()
    tracking = TrackingResolution("fixture", "file:///fixture", True)
    logged = LoggedCandidate("e" * 32, "runs:/fixture/model")
    monkeypatch.setattr(evaluate_final, "configure_experiment_tracking", lambda **_: tracking)
    monkeypatch.setattr(evaluate_final, "log_final_model_to_mlflow", lambda **_: logged)
    monkeypatch.setattr(evaluate_final, "_verify_logged_model", lambda *_, **__: (True, True, True))
    monkeypatch.setattr(evaluate_final, "get_git_version", lambda _: None)
    monkeypatch.setattr(
        evaluate_final,
        "ModelRegistryService",
        lambda: cast(ModelRegistryService, registry),
    )
    monkeypatch.setattr(
        evaluate_final,
        "mlflow",
        SimpleNamespace(MlflowClient=lambda: client),
    )

    result = evaluate_final.run_final_evaluation(
        settings=settings,
        final_config=FinalModelConfig.load(),
        processed_dir=processed,
        split_manifest_path=manifest_path,
        model_artifacts_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        project_root=tmp_path,
        clock=lambda: datetime(2026, 8, 12, tzinfo=UTC),
    )

    audit = json.loads((tmp_path / "reports/test_access_audit.json").read_text(encoding="utf-8"))
    assert result.registered_version.version == "8"
    assert result.champion_version is None
    assert result.evaluation.metrics["macro_f1"] == pytest.approx(1.0)
    assert audit["status"] == "completed"
    assert audit["repeated_test_evaluation_allowed"] is False
    assert registry.tags[0]["candidate_alias"] == "candidate"
    assert len(client.tags) == 2
    assert (result.report_directory / "promotion_gate_results.json").is_file()
    assert (result.artifacts.artifact_directory / "test_predictions.parquet").is_file()

    with pytest.raises(evaluate_final.FinalEvaluationError, match="already been recorded"):
        evaluate_final.run_final_evaluation(
            settings=settings,
            final_config=FinalModelConfig.load(),
            processed_dir=processed,
            split_manifest_path=manifest_path,
            model_artifacts_dir=tmp_path / "models",
            reports_dir=tmp_path / "reports",
            project_root=tmp_path,
        )


def _manifest(settings: Settings, paths: dict[str, Path]) -> SplitManifest:
    counts = {split: pl.read_parquet(path).height for split, path in paths.items()}
    total = sum(counts.values())
    return SplitManifest(
        preparation_timestamp_utc="2026-08-12T00:00:00Z",
        data_source_hashes={"normalized_data": "a" * 64},
        configuration_hash="b" * 64,
        preprocessing=settings.project_config.preprocessing,
        splitting=settings.project_config.splitting,
        split_ratios=settings.project_config.split_ratios,
        random_seed=42,
        duplicate_group_column="normalized_text_hash",
        selected_input_rows=total,
        contradictory_group_count=0,
        contradictory_rows_excluded=0,
        final_row_count=total,
        split_counts=counts,
        split_percentages={split: count / total for split, count in counts.items()},
        per_class_counts={},
        maximum_class_proportion_deviation={},
        most_deviant_class={},
        label_mapping={"Billing": 0, "Technical": 1, "Returns": 2},
        model_feature_columns=("model_text",),
        target_column="queue",
        output_files={
            split: OutputFileManifest(
                path=path.as_posix(),
                sha256=sha256_file(path),
                row_count=counts[split],
            )
            for split, path in paths.items()
        },
        preparation_code_version=None,
    )


def _tiny_text_classification_data() -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for label, vocabulary in (
        ("Billing", "invoice payment refund"),
        ("Technical", "server network error"),
        ("Returns", "return exchange parcel"),
    ):
        for index in range(12):
            texts.append(f"{vocabulary} request number {index % 4}")
            labels.append(label)
    return texts, labels
