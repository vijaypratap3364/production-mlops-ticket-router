"""Tests for split artifact generation and manifest integrity."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from ticket_router.config import Settings
from ticket_router.data.class_selection import ClassSelectionReport
from ticket_router.data.download import configuration_hash
from ticket_router.data.manifests import NormalizationManifest
from ticket_router.data.normalize import combine_ticket_text
from ticket_router.data.prepare import prepare_dataset
from ticket_router.hashing import sha256_file

HASH = "a" * 64


def _normalized_fixture() -> pl.DataFrame:
    records: list[dict[str, object]] = []
    record_number = 0
    for class_index, label in enumerate(("Queue A", "Queue B")):
        for group_index in range(30):
            repeats = 2 if group_index % 5 == 0 else 1
            subject = f"Class {class_index} request {group_index}"
            body = f"Fixture message {group_index}; contact fixture@example.com."
            for _ in range(repeats):
                record_number += 1
                records.append(
                    {
                        "ticket_record_id": f"{record_number:064x}",
                        "source_row_id": f"fixture.csv:{record_number}",
                        "source_file": "fixture.csv",
                        "source_row_number": record_number,
                        "language": "en",
                        "subject": subject,
                        "body": body,
                        "text": combine_ticket_text(subject, body),
                        "queue": label,
                    }
                )
    return pl.DataFrame(records)


def test_prepare_writes_hashed_outputs_and_reuses_verified_manifest(tmp_path: Path) -> None:
    base_settings = Settings.load(env_file=None)
    dataset_settings = base_settings.project_config.dataset.model_copy(
        update={"number_of_target_queues": 2, "minimum_class_count": 7}
    )
    splitting_settings = base_settings.project_config.splitting.model_copy(
        update={
            "class_proportion_tolerance": 0.05,
            "split_size_tolerance": 0.05,
        }
    )
    project_config = base_settings.project_config.model_copy(
        update={"dataset": dataset_settings, "splitting": splitting_settings}
    )
    settings = base_settings.model_copy(update={"project_config": project_config})

    frame = _normalized_fixture()
    normalized_path = tmp_path / "data" / "interim" / "normalized.parquet"
    normalized_path.parent.mkdir(parents=True)
    frame.write_parquet(normalized_path)
    normalized_hash = sha256_file(normalized_path)
    normalization_manifest = NormalizationManifest(
        normalization_timestamp_utc="2026-08-05T00:00:00Z",
        source_repository="owner/synthetic",
        requested_revision="b" * 40,
        resolved_revision="b" * 40,
        raw_manifest_path="data/raw/data_manifest.json",
        raw_manifest_sha256=HASH,
        input_file_sha256={},
        output_file_path="data/interim/normalized.parquet",
        output_file_sha256=normalized_hash,
        rows_read=frame.height,
        rows_dropped_malformed=0,
        english_rows=frame.height,
        rows_dropped_missing_queue=0,
        rows_dropped_missing_text=0,
        output_row_count=frame.height,
        output_column_names=tuple(frame.columns),
        configuration_hash=HASH,
        normalization_code_version=None,
    )
    normalization_manifest_path = normalized_path.parent / "normalization_manifest.json"
    normalization_manifest.write(normalization_manifest_path)
    class_count = frame.height // 2
    selected_classes = ClassSelectionReport(
        input_file_path="data/interim/normalized.parquet",
        input_file_sha256=normalized_hash,
        configuration_hash=HASH,
        requested_class_count=2,
        configured_minimum_class_count=7,
        minimum_count_for_stratified_split=7,
        effective_minimum_class_count=7,
        original_class_counts={"Queue A": class_count, "Queue B": class_count},
        selected_classes=("Queue A", "Queue B"),
        label_mapping={"Queue A": 0, "Queue B": 1},
        selected_class_counts={"Queue A": class_count, "Queue B": class_count},
        excluded_classes=(),
        final_row_count=frame.height,
        class_proportions={"Queue A": 0.5, "Queue B": 0.5},
        imbalance_ratio=1.0,
    )
    selected_classes_path = tmp_path / "artifacts" / "reports" / "selected_classes.json"
    selected_classes.write(selected_classes_path)

    first = prepare_dataset(
        settings=settings,
        normalized_path=normalized_path,
        normalization_manifest_path=normalization_manifest_path,
        selected_classes_path=selected_classes_path,
        processed_dir=tmp_path / "data" / "processed",
        reference_dir=tmp_path / "data" / "reference",
        reports_dir=tmp_path / "artifacts" / "reports",
        project_root=tmp_path,
        clock=lambda: datetime(2026, 8, 5, 12, 0, tzinfo=UTC),
    )
    second = prepare_dataset(
        settings=settings,
        normalized_path=normalized_path,
        normalization_manifest_path=normalization_manifest_path,
        selected_classes_path=selected_classes_path,
        processed_dir=tmp_path / "data" / "processed",
        reference_dir=tmp_path / "data" / "reference",
        reports_dir=tmp_path / "artifacts" / "reports",
        project_root=tmp_path,
    )

    assert first == second
    assert first.preparation_timestamp_utc == "2026-08-05T12:00:00Z"
    assert first.final_row_count == frame.height
    assert set(first.output_files) == {"train", "validation", "test", "training_reference"}
    for output in first.output_files.values():
        output_path = tmp_path / output.path
        assert output.sha256 == sha256_file(output_path)
        assert output.row_count == pl.read_parquet(output_path).height
    assert (tmp_path / "artifacts" / "reports" / "split_summary.json").is_file()


def test_prepare_reuses_semantically_identical_selection_after_serialization_change(
    tmp_path: Path,
) -> None:
    base_settings = Settings.load(env_file=None)
    dataset_settings = base_settings.project_config.dataset.model_copy(
        update={"number_of_target_queues": 2, "minimum_class_count": 7}
    )
    splitting_settings = base_settings.project_config.splitting.model_copy(
        update={"class_proportion_tolerance": 0.05, "split_size_tolerance": 0.05}
    )
    settings = base_settings.model_copy(
        update={
            "project_config": base_settings.project_config.model_copy(
                update={"dataset": dataset_settings, "splitting": splitting_settings}
            )
        }
    )
    frame = _normalized_fixture()
    normalized_path = tmp_path / "data" / "interim" / "normalized.parquet"
    normalized_path.parent.mkdir(parents=True)
    frame.write_parquet(normalized_path)
    normalized_hash = sha256_file(normalized_path)
    normalization_manifest_path = normalized_path.parent / "normalization_manifest.json"
    NormalizationManifest(
        normalization_timestamp_utc="2026-08-05T00:00:00Z",
        source_repository="owner/synthetic",
        requested_revision="b" * 40,
        resolved_revision="b" * 40,
        raw_manifest_path="data/raw/data_manifest.json",
        raw_manifest_sha256=HASH,
        input_file_sha256={},
        output_file_path="data/interim/normalized.parquet",
        output_file_sha256=normalized_hash,
        rows_read=frame.height,
        rows_dropped_malformed=0,
        english_rows=frame.height,
        rows_dropped_missing_queue=0,
        rows_dropped_missing_text=0,
        output_row_count=frame.height,
        output_column_names=tuple(frame.columns),
        configuration_hash=HASH,
        normalization_code_version=None,
    ).write(normalization_manifest_path)
    class_count = frame.height // 2
    selection = ClassSelectionReport(
        input_file_path="data/interim/normalized.parquet",
        input_file_sha256=normalized_hash,
        configuration_hash=configuration_hash(settings),
        requested_class_count=2,
        configured_minimum_class_count=7,
        minimum_count_for_stratified_split=7,
        effective_minimum_class_count=7,
        original_class_counts={"Queue A": class_count, "Queue B": class_count},
        selected_classes=("Queue A", "Queue B"),
        label_mapping={"Queue A": 0, "Queue B": 1},
        selected_class_counts={"Queue A": class_count, "Queue B": class_count},
        excluded_classes=(),
        final_row_count=frame.height,
        class_proportions={"Queue A": 0.5, "Queue B": 0.5},
        imbalance_ratio=1.0,
    )
    selection_path = tmp_path / "artifacts" / "reports" / "selected_classes.json"
    selection.write(selection_path)
    first = prepare_dataset(
        settings=settings,
        normalized_path=normalized_path,
        normalization_manifest_path=normalization_manifest_path,
        selected_classes_path=selection_path,
        processed_dir=tmp_path / "data" / "processed",
        reference_dir=tmp_path / "data" / "reference",
        reports_dir=tmp_path / "artifacts" / "reports",
        project_root=tmp_path,
    )

    selection_path.write_text(
        json.dumps(selection.model_dump(mode="json"), separators=(",", ":")),
        encoding="utf-8",
    )
    second = prepare_dataset(
        settings=settings,
        normalized_path=normalized_path,
        normalization_manifest_path=normalization_manifest_path,
        selected_classes_path=selection_path,
        processed_dir=tmp_path / "data" / "processed",
        reference_dir=tmp_path / "data" / "reference",
        reports_dir=tmp_path / "artifacts" / "reports",
        project_root=tmp_path,
    )

    assert second == first
