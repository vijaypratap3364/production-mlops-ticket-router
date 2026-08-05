"""End-to-end unit test for a tiny validation-only baseline run."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from ticket_router.config import Settings
from ticket_router.data.load import ModelingDataset, TrainingSplitName, load_training_split
from ticket_router.data.split_manifest import OutputFileManifest, SplitManifest
from ticket_router.hashing import sha256_file
from ticket_router.modeling import train_baseline as training_module
from ticket_router.modeling.config import BaselineSettings


def test_training_run_writes_all_artifacts_without_loading_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tiny_text_classification_data: tuple[list[str], list[str]],
    small_baseline_config: BaselineSettings,
) -> None:
    settings = Settings.load(env_file=None)
    processed = tmp_path / "processed"
    processed.mkdir()
    texts, labels = tiny_text_classification_data
    train_indices = [index for index in range(len(texts)) if index % 12 < 9]
    validation_indices = [index for index in range(len(texts)) if index % 12 >= 9]
    train = _frame(texts, labels, train_indices)
    validation = _frame(texts, labels, validation_indices)
    train_path = processed / "train.parquet"
    validation_path = processed / "validation.parquet"
    train.write_parquet(train_path)
    validation.write_parquet(validation_path)
    manifest_path = processed / "split_manifest.json"
    _manifest(settings, train_path, validation_path, train.height, validation.height).write(
        manifest_path
    )
    requested_splits: list[str] = []

    def recording_loader(path: Path, split: TrainingSplitName) -> ModelingDataset:
        requested_splits.append(split)
        return load_training_split(path, split)

    monkeypatch.setattr(training_module, "load_training_split", recording_loader)
    result = training_module.train_baselines(
        settings=settings,
        baseline_config=small_baseline_config,
        processed_dir=processed,
        split_manifest_path=manifest_path,
        artifacts_dir=tmp_path / "models",
        leaderboard_path=tmp_path / "reports" / "model_leaderboard.csv",
        project_root=tmp_path,
        run_id="fixture-run",
    )

    assert requested_splits == ["train", "validation"]
    assert not (processed / "test.parquet").exists()
    assert result.leaderboard_path.is_file()
    assert pl.read_csv(result.leaderboard_path).height == 3
    assert (result.run_directory / "run_manifest.json").is_file()
    required = {
        "pipeline.joblib",
        "metrics.json",
        "classification_report.json",
        "confusion_matrix.png",
        "per_class_metrics.csv",
        "validation_predictions.parquet",
        "model_configuration.json",
        "training_metadata.json",
        "inference_benchmark.json",
        "error_analysis.json",
    }
    for summary in result.model_summaries:
        assert required == {path.name for path in summary.artifact_directory.iterdir()}


def _frame(texts: list[str], labels: list[str], indices: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model_text": [texts[index] for index in indices],
            "queue": [labels[index] for index in indices],
        }
    )


def _manifest(
    settings: Settings,
    train_path: Path,
    validation_path: Path,
    train_rows: int,
    validation_rows: int,
) -> SplitManifest:
    label_mapping = {"Billing": 0, "Technical": 1, "Returns": 2}
    return SplitManifest(
        preparation_timestamp_utc="2026-08-05T00:00:00Z",
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
        label_mapping=label_mapping,
        model_feature_columns=("model_text",),
        target_column="queue",
        output_files={
            "train": OutputFileManifest(
                path="processed/train.parquet",
                sha256=sha256_file(train_path),
                row_count=train_rows,
            ),
            "validation": OutputFileManifest(
                path="processed/validation.parquet",
                sha256=sha256_file(validation_path),
                row_count=validation_rows,
            ),
        },
        preparation_code_version=None,
    )
