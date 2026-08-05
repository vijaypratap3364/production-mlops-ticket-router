"""Tests for reproducible Stage 3 artifact generation."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ticket_router.config import Settings
from ticket_router.data.analyze import run_analysis
from ticket_router.data.manifests import NormalizationManifest, RawDataManifest
from ticket_router.data.normalize import combine_ticket_text
from ticket_router.hashing import sha256_file

HASH = "a" * 64


def test_analysis_command_generates_expected_reports(tmp_path: Path) -> None:
    base_settings = Settings.load(env_file=None)
    dataset_settings = base_settings.project_config.dataset.model_copy(
        update={"number_of_target_queues": 2, "minimum_class_count": 7}
    )
    project_config = base_settings.project_config.model_copy(update={"dataset": dataset_settings})
    settings = base_settings.model_copy(update={"project_config": project_config})

    records: list[dict[str, object]] = []
    for index in range(14):
        subject = f"Synthetic request {index}"
        body = "A non-sensitive fixture body for report testing."
        records.append(
            {
                "ticket_record_id": f"{index + 1:064x}",
                "source_row_id": f"fixture.csv:{index + 1}",
                "source_file": "fixture.csv",
                "source_row_number": index + 1,
                "language": "en",
                "subject": subject,
                "body": body,
                "text": combine_ticket_text(subject, body),
                "queue": "Queue A" if index < 7 else "Queue B",
            }
        )
    normalized_path = tmp_path / "data" / "interim" / "normalized.parquet"
    normalized_path.parent.mkdir(parents=True)
    frame = pl.DataFrame(records)
    frame.write_parquet(normalized_path)

    raw_manifest = RawDataManifest(
        source_repository="owner/synthetic",
        requested_revision="b" * 40,
        resolved_revision="b" * 40,
        download_timestamp_utc="2026-08-04T00:00:00Z",
        raw_file_paths=(),
        row_count=14,
        column_names=("subject", "body", "answer", "queue", "language"),
        file_sha256={},
        file_size_bytes={},
        dataset_license="test-only",
        ingestion_code_version=None,
        configuration_hash=HASH,
    )
    raw_manifest_path = tmp_path / "data" / "raw" / "data_manifest.json"
    raw_manifest.write(raw_manifest_path)
    normalization_manifest = NormalizationManifest(
        normalization_timestamp_utc="2026-08-04T00:00:00Z",
        source_repository="owner/synthetic",
        requested_revision="b" * 40,
        resolved_revision="b" * 40,
        raw_manifest_path="data/raw/data_manifest.json",
        raw_manifest_sha256=sha256_file(raw_manifest_path),
        input_file_sha256={},
        output_file_path="data/interim/normalized.parquet",
        output_file_sha256=sha256_file(normalized_path),
        rows_read=14,
        rows_dropped_malformed=0,
        english_rows=14,
        rows_dropped_missing_queue=0,
        rows_dropped_missing_text=0,
        output_row_count=14,
        output_column_names=tuple(frame.columns),
        configuration_hash=HASH,
        normalization_code_version=None,
    )
    normalization_manifest_path = normalized_path.parent / "normalization_manifest.json"
    normalization_manifest.write(normalization_manifest_path)
    reports_dir = tmp_path / "artifacts" / "reports"

    result = run_analysis(
        settings=settings,
        normalized_path=normalized_path,
        normalization_manifest_path=normalization_manifest_path,
        raw_manifest_path=raw_manifest_path,
        reports_dir=reports_dir,
        project_root=tmp_path,
    )

    assert result["final_usable_row_count"] == 14
    assert sorted(path.name for path in reports_dir.iterdir()) == [
        "duplicate_analysis.json",
        "eda_report.html",
        "eda_report.json",
        "selected_classes.json",
    ]
    assert "Customer-Support Ticket Data Analysis" in (reports_dir / "eda_report.html").read_text(
        encoding="utf-8"
    )
