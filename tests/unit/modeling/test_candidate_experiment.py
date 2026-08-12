"""Small-fixture Stage 6 lifecycle without network, test data, or external MLflow."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from ticket_router.config import Settings
from ticket_router.data.split_manifest import OutputFileManifest, SplitManifest
from ticket_router.hashing import sha256_file
from ticket_router.modeling import train_candidates
from ticket_router.modeling.experiment_config import CandidateExperimentSettings
from ticket_router.modeling.mlflow_tracking import LoggedCandidate, TrackingResolution


class TagClient:
    def __init__(self) -> None:
        self.tags: list[tuple[str, str, str]] = []

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        self.tags.append((run_id, key, value))


def test_fixture_candidate_search_writes_selection_artifacts_without_test_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tiny_text_classification_data: tuple[list[str], list[str]],
    small_experiment_config: CandidateExperimentSettings,
) -> None:
    settings = Settings.load(env_file=None)
    processed = tmp_path / "processed"
    processed.mkdir()
    texts, labels = tiny_text_classification_data
    train_indices = [index for index in range(len(texts)) if index % 12 < 9]
    validation_indices = [index for index in range(len(texts)) if index % 12 >= 9]
    train_path = processed / "train.parquet"
    validation_path = processed / "validation.parquet"
    _frame(texts, labels, train_indices).write_parquet(train_path)
    _frame(texts, labels, validation_indices).write_parquet(validation_path)
    manifest_path = processed / "split_manifest.json"
    _manifest(settings, train_path, validation_path).write(manifest_path)
    tracking = TrackingResolution("fixture", "file:///fixture", True)
    client = TagClient()
    logged_names: list[str] = []

    def log_candidate(**kwargs: object) -> LoggedCandidate:
        run_name = str(kwargs["run_name"])
        logged_names.append(run_name)
        run_id = f"{len(logged_names):032x}"
        return LoggedCandidate(run_id, f"runs:/{run_id}/model")

    monkeypatch.setattr(train_candidates, "configure_experiment_tracking", lambda **_: tracking)
    monkeypatch.setattr(train_candidates, "log_candidate_to_mlflow", log_candidate)
    monkeypatch.setattr(
        train_candidates,
        "mlflow",
        SimpleNamespace(MlflowClient=lambda: client),
    )
    monkeypatch.setattr(train_candidates, "get_git_version", lambda _: None)

    result = train_candidates.run_candidate_experiments(
        settings=settings,
        experiment_config=small_experiment_config,
        processed_dir=processed,
        split_manifest_path=manifest_path,
        model_artifacts_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        leaderboard_path=tmp_path / "model_leaderboard.csv",
        project_root=tmp_path,
        experiment_run_id="candidate-fixture",
    )

    assert len(result.candidates) == 5
    assert result.selected_candidate in {record.spec.name for record in result.candidates}
    assert len(logged_names) == 5
    assert len(client.tags) == 10
    assert not (processed / "test.parquet").exists()
    assert pl.read_csv(tmp_path / "model_leaderboard.csv").height == 5
    expected_reports = {
        "calibration_plot.png",
        "calibration_points.json",
        "candidate_comparison.json",
        "confidence_distribution.json",
        "cross_validation_summary.json",
        "experiment_lineage.json",
        "mlflow_run_comparison.json",
        "validation_error_analysis.json",
    }
    assert expected_reports == {path.name for path in result.report_directory.iterdir()}
    assert all(record.artifacts.artifact_directory.is_dir() for record in result.candidates)


def _frame(texts: list[str], labels: list[str], indices: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model_text": [texts[index] for index in indices],
            "queue": [labels[index] for index in indices],
        }
    )


def _manifest(settings: Settings, train_path: Path, validation_path: Path) -> SplitManifest:
    train_rows = pl.read_parquet(train_path).height
    validation_rows = pl.read_parquet(validation_path).height
    return SplitManifest(
        preparation_timestamp_utc="2026-08-12T00:00:00Z",
        data_source_hashes={"normalized_data": "a" * 64},
        configuration_hash="b" * 64,
        preprocessing=settings.project_config.preprocessing,
        splitting=settings.project_config.splitting,
        split_ratios=settings.project_config.split_ratios,
        random_seed=42,
        duplicate_group_column="normalized_text_hash",
        selected_input_rows=train_rows + validation_rows,
        contradictory_group_count=0,
        contradictory_rows_excluded=0,
        final_row_count=train_rows + validation_rows,
        split_counts={"train": train_rows, "validation": validation_rows, "test": 0},
        split_percentages={"train": 0.75, "validation": 0.25, "test": 0.0},
        per_class_counts={},
        maximum_class_proportion_deviation={},
        most_deviant_class={},
        label_mapping={"Billing": 0, "Technical": 1, "Returns": 2},
        model_feature_columns=("model_text",),
        target_column="queue",
        output_files={
            "train": OutputFileManifest(
                path=train_path.as_posix(),
                sha256=sha256_file(train_path),
                row_count=train_rows,
            ),
            "validation": OutputFileManifest(
                path=validation_path.as_posix(),
                sha256=sha256_file(validation_path),
                row_count=validation_rows,
            ),
        },
        preparation_code_version=None,
    )
