"""Tests for deterministic, English-only ticket normalization."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from ticket_router.data.manifests import GitVersion, NormalizationManifest, RawDataManifest
from ticket_router.data.normalize import normalize_dataset, stable_ticket_record_id
from ticket_router.hashing import sha256_file

REVISION = "a" * 40
CONFIGURATION_HASH = "c" * 64


def _raw_manifest(tmp_path: Path) -> Path:
    source_dir = tmp_path / "data" / "raw" / "source"
    source_dir.mkdir(parents=True)
    source_file = source_dir / "tickets.csv"
    shutil.copy2(Path("tests/fixtures/tickets.csv"), source_file)
    path_text = "data/raw/source/tickets.csv"
    manifest = RawDataManifest(
        source_repository="owner/tickets",
        requested_revision=REVISION,
        resolved_revision=REVISION,
        download_timestamp_utc="2026-08-03T12:00:00Z",
        raw_file_paths=(path_text,),
        row_count=5,
        column_names=(
            "Subject",
            "Body",
            "Answer",
            "Type",
            "Queue",
            "Priority",
            "Language",
            "Tags",
        ),
        file_sha256={path_text: sha256_file(source_file)},
        file_size_bytes={path_text: source_file.stat().st_size},
        dataset_license="cc-by-nc-4.0",
        ingestion_code_version=None,
        configuration_hash=CONFIGURATION_HASH,
    )
    manifest_path = tmp_path / "data" / "raw" / "data_manifest.json"
    manifest.write(manifest_path)
    return manifest_path


def test_stable_record_id_is_deterministic() -> None:
    def record_id(source_row_number: int) -> str:
        return stable_ticket_record_id(
            source_repository="owner/tickets",
            resolved_revision=REVISION,
            source_file="data/raw/source/tickets.csv",
            source_row_number=source_row_number,
            subject="Cannot sign in",
            body="Error",
            queue="Technical Support",
        )

    first = record_id(1)
    second = record_id(1)
    changed = record_id(2)

    assert first == second
    assert len(first) == 64
    assert changed != first


def test_normalization_filters_language_and_missing_values(tmp_path: Path) -> None:
    raw_manifest_path = _raw_manifest(tmp_path)

    manifest = normalize_dataset(
        raw_manifest_path=raw_manifest_path,
        interim_dir=tmp_path / "data" / "interim",
        project_root=tmp_path,
        language_filter="en",
        configuration_digest=CONFIGURATION_HASH,
        clock=lambda: datetime(2026, 8, 3, 13, 0, tzinfo=UTC),
        code_version=GitVersion(commit="b" * 40, dirty=False),
    )

    output_path = tmp_path / manifest.output_file_path
    normalized = pl.read_parquet(output_path)
    assert manifest.rows_read == 5
    assert manifest.rows_dropped_malformed == 0
    assert manifest.english_rows == 4
    assert manifest.rows_dropped_missing_text == 1
    assert manifest.rows_dropped_missing_queue == 1
    assert manifest.output_row_count == 2
    assert normalized.columns == list(manifest.output_column_names)
    assert normalized["language"].to_list() == ["en", "en"]
    assert normalized["subject"].to_list() == ["Cannot sign in", None]
    assert normalized["text"].to_list()[1] == (
        "[SUBJECT] \n[BODY] Please send a copy of my latest invoice."
    )
    assert normalized["ticket_record_id"].n_unique() == 2
    assert manifest.output_file_sha256 == sha256_file(output_path)

    saved_manifest = NormalizationManifest.read(
        tmp_path / "data" / "interim" / "normalization_manifest.json"
    )
    assert saved_manifest == manifest


def test_normalization_drops_structurally_malformed_rows(tmp_path: Path) -> None:
    source_dir = tmp_path / "data" / "raw" / "source"
    source_dir.mkdir(parents=True)
    source_file = source_dir / "tickets.csv"
    source_file.write_text(
        "subject,body,queue,language\n"
        "Valid,Usable,Support,en\n"
        "Broken,Unquoted comma,Support,en,unexpected\n",
        encoding="utf-8",
    )
    path_text = "data/raw/source/tickets.csv"
    raw_manifest = RawDataManifest(
        source_repository="owner/tickets",
        requested_revision=REVISION,
        resolved_revision=REVISION,
        download_timestamp_utc="2026-08-03T12:00:00Z",
        raw_file_paths=(path_text,),
        row_count=2,
        column_names=("subject", "body", "queue", "language"),
        file_sha256={path_text: sha256_file(source_file)},
        file_size_bytes={path_text: source_file.stat().st_size},
        dataset_license="cc-by-nc-4.0",
        ingestion_code_version=None,
        configuration_hash=CONFIGURATION_HASH,
    )
    raw_manifest_path = tmp_path / "data" / "raw" / "data_manifest.json"
    raw_manifest.write(raw_manifest_path)

    manifest = normalize_dataset(
        raw_manifest_path=raw_manifest_path,
        interim_dir=tmp_path / "data" / "interim",
        project_root=tmp_path,
        language_filter="en",
        configuration_digest=CONFIGURATION_HASH,
    )

    assert manifest.rows_read == 2
    assert manifest.rows_dropped_malformed == 1
    assert manifest.english_rows == 1
    assert manifest.output_row_count == 1
