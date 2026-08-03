"""Normalize raw support-ticket CSV files into leakage-aware Parquet data."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl

from ticket_router.config import Settings
from ticket_router.data.download import configuration_hash
from ticket_router.data.manifests import (
    GitVersion,
    NormalizationManifest,
    RawDataManifest,
    get_git_version,
)
from ticket_router.hashing import sha256_file, sha256_json
from ticket_router.logging_config import configure_logging, get_logger

NORMALIZED_DATA_FILENAME = "normalized_tickets.parquet"
NORMALIZATION_MANIFEST_FILENAME = "normalization_manifest.json"
NULL_REPRESENTATIONS = frozenset({"", "null", "none", "nan", "n/a", "na", "<na>"})
ENGLISH_LANGUAGE_ALIASES = frozenset({"en", "eng", "english"})
REQUIRED_SOURCE_COLUMNS = frozenset({"language", "subject", "body", "queue"})
OUTPUT_COLUMNS = (
    "ticket_record_id",
    "source_row_id",
    "source_file",
    "source_row_number",
    "language",
    "subject",
    "body",
    "text",
    "queue",
)
OUTPUT_SCHEMA = {
    "ticket_record_id": pl.String,
    "source_row_id": pl.String,
    "source_file": pl.String,
    "source_row_number": pl.Int64,
    "language": pl.String,
    "subject": pl.String,
    "body": pl.String,
    "text": pl.String,
    "queue": pl.String,
}


class NormalizationError(RuntimeError):
    """Raised when normalized data cannot be produced safely."""


def normalize_column_name(value: str) -> str:
    """Convert an upstream column name to deterministic snake_case."""
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip()).strip("_").casefold()
    if not normalized:
        raise NormalizationError(f"Column name normalizes to an empty value: {value!r}")
    return normalized


def normalize_nullable_text(value: object) -> str | None:
    """Strip text and map common explicit null markers to None."""
    if value is None:
        return None
    normalized = str(value).strip()
    if normalized.casefold() in NULL_REPRESENTATIONS:
        return None
    return normalized


def combine_ticket_text(subject: str | None, body: str | None) -> str:
    """Create combined text using only the two allowed source fields."""
    return f"[SUBJECT] {subject or ''}\n[BODY] {body or ''}".strip()


def stable_ticket_record_id(
    *,
    source_repository: str,
    resolved_revision: str | None,
    source_file: str,
    source_row_number: int,
    subject: str | None,
    body: str | None,
    queue: str,
) -> str:
    """Generate a stable, source-aware record identifier."""
    return sha256_json(
        {
            "body": body,
            "queue": queue,
            "resolved_revision": resolved_revision,
            "source_file": source_file,
            "source_repository": source_repository,
            "source_row_number": source_row_number,
            "subject": subject,
        }
    )


def normalize_dataset(
    *,
    raw_manifest_path: Path,
    interim_dir: Path,
    project_root: Path,
    language_filter: str,
    configuration_digest: str,
    force: bool = False,
    clock: Callable[[], datetime] | None = None,
    code_version: GitVersion | None = None,
) -> NormalizationManifest:
    """Normalize all raw CSV files named by a verified raw manifest."""
    resolved_project_root = project_root.resolve()
    resolved_manifest_path = raw_manifest_path.resolve()
    resolved_interim_dir = interim_dir.resolve()
    output_path = resolved_interim_dir / NORMALIZED_DATA_FILENAME
    normalization_manifest_path = resolved_interim_dir / NORMALIZATION_MANIFEST_FILENAME
    raw_manifest = RawDataManifest.read(resolved_manifest_path)
    raw_manifest_hash = sha256_file(resolved_manifest_path)

    if not force and output_path.exists() and normalization_manifest_path.exists():
        existing_manifest = NormalizationManifest.read(normalization_manifest_path)
        _validate_cached_normalization(
            existing_manifest,
            raw_manifest_hash=raw_manifest_hash,
            configuration_digest=configuration_digest,
            project_root=resolved_project_root,
        )
        return existing_manifest

    if not force and (output_path.exists() or normalization_manifest_path.exists()):
        raise NormalizationError(
            f"Incomplete normalized data already exists in {resolved_interim_dir}. "
            "Refusing to overwrite it; inspect the directory or rerun with --force."
        )

    csv_paths = _verified_csv_paths(raw_manifest, resolved_project_root)
    if not csv_paths:
        raise NormalizationError("Raw manifest contains no CSV source files.")

    records: list[dict[str, Any]] = []
    rows_read = 0
    rows_dropped_malformed = 0
    english_rows = 0
    rows_dropped_missing_queue = 0
    rows_dropped_missing_text = 0

    for path_text, source_path in csv_paths:
        for source_row_number, source_row in _csv_rows(source_path):
            rows_read += 1
            if None in source_row:
                rows_dropped_malformed += 1
                continue
            normalized_row = _normalized_row_keys(source_row, source_path)
            language = normalize_nullable_text(normalized_row.get("language"))
            if not _language_matches(language, language_filter):
                continue
            english_rows += 1

            subject = normalize_nullable_text(normalized_row.get("subject"))
            body = normalize_nullable_text(normalized_row.get("body"))
            queue = normalize_nullable_text(normalized_row.get("queue"))
            if queue is None:
                rows_dropped_missing_queue += 1
                continue
            if subject is None and body is None:
                rows_dropped_missing_text += 1
                continue

            source_row_id = f"{path_text}:{source_row_number}"
            records.append(
                {
                    "ticket_record_id": stable_ticket_record_id(
                        source_repository=raw_manifest.source_repository,
                        resolved_revision=raw_manifest.resolved_revision,
                        source_file=path_text,
                        source_row_number=source_row_number,
                        subject=subject,
                        body=body,
                        queue=queue,
                    ),
                    "source_row_id": source_row_id,
                    "source_file": path_text,
                    "source_row_number": source_row_number,
                    "language": language_filter.casefold(),
                    "subject": subject,
                    "body": body,
                    "text": combine_ticket_text(subject, body),
                    "queue": queue,
                }
            )

    normalized_frame = pl.DataFrame(records, schema=OUTPUT_SCHEMA)
    if normalized_frame["ticket_record_id"].n_unique() != normalized_frame.height:
        raise NormalizationError("Stable ticket_record_id values are not unique.")

    resolved_interim_dir.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")
    try:
        normalized_frame.write_parquet(temporary_output, compression="zstd", statistics=True)
        output_hash = sha256_file(temporary_output)
        os.replace(temporary_output, output_path)
    finally:
        temporary_output.unlink(missing_ok=True)

    input_file_hashes = {
        path_text: raw_manifest.file_sha256[path_text] for path_text, _ in csv_paths
    }
    manifest = NormalizationManifest(
        normalization_timestamp_utc=_format_utc((clock or _utc_now)()),
        source_repository=raw_manifest.source_repository,
        requested_revision=raw_manifest.requested_revision,
        resolved_revision=raw_manifest.resolved_revision,
        raw_manifest_path=_portable_project_path(resolved_manifest_path, resolved_project_root),
        raw_manifest_sha256=raw_manifest_hash,
        input_file_sha256=input_file_hashes,
        output_file_path=_portable_project_path(output_path, resolved_project_root),
        output_file_sha256=output_hash,
        rows_read=rows_read,
        rows_dropped_malformed=rows_dropped_malformed,
        english_rows=english_rows,
        rows_dropped_missing_queue=rows_dropped_missing_queue,
        rows_dropped_missing_text=rows_dropped_missing_text,
        output_row_count=normalized_frame.height,
        output_column_names=OUTPUT_COLUMNS,
        configuration_hash=configuration_digest,
        normalization_code_version=code_version or get_git_version(resolved_project_root),
    )
    manifest.write(normalization_manifest_path)
    return manifest


def _verified_csv_paths(
    manifest: RawDataManifest,
    project_root: Path,
) -> list[tuple[str, Path]]:
    csv_paths: list[tuple[str, Path]] = []
    for path_text in manifest.raw_file_paths:
        if Path(path_text).suffix.casefold() != ".csv":
            continue
        source_path = _resolve_project_path(path_text, project_root)
        if not source_path.is_file():
            raise NormalizationError(f"Raw CSV file is missing: {source_path}")
        actual_hash = sha256_file(source_path)
        expected_hash = manifest.file_sha256.get(path_text)
        if actual_hash != expected_hash:
            raise NormalizationError(f"Raw CSV hash does not match its manifest: {source_path}")
        csv_paths.append((path_text, source_path))
    return sorted(csv_paths)


def _csv_rows(path: Path) -> Sequence[tuple[int, Mapping[str | None, object]]]:
    rows: list[tuple[int, Mapping[str | None, object]]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        sample = csv_file.read(65536)
        csv_file.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(csv_file, dialect=dialect)
        if reader.fieldnames is None:
            raise NormalizationError(f"CSV file has no header: {path}")
        for source_row_number, row in enumerate(reader, start=1):
            rows.append((source_row_number, row))
    return rows


def _normalized_row_keys(
    source_row: Mapping[str | None, object],
    source_path: Path,
) -> dict[str, object]:
    normalized_row: dict[str, object] = {}
    for column, value in source_row.items():
        if column is None:
            raise NormalizationError(
                f"CSV row in {source_path} contains fields beyond its declared header."
            )
        normalized_column = normalize_column_name(column)
        if normalized_column in normalized_row:
            raise NormalizationError(
                f"Columns collide after snake_case normalization in {source_path}: {column!r}"
            )
        normalized_row[normalized_column] = value

    missing_columns = sorted(REQUIRED_SOURCE_COLUMNS - normalized_row.keys())
    if missing_columns:
        raise NormalizationError(
            f"CSV file {source_path} is missing required columns: {', '.join(missing_columns)}"
        )
    return normalized_row


def _language_matches(value: str | None, language_filter: str) -> bool:
    if value is None:
        return False
    normalized_value = value.casefold()
    normalized_filter = language_filter.casefold()
    if normalized_filter in ENGLISH_LANGUAGE_ALIASES:
        return normalized_value in ENGLISH_LANGUAGE_ALIASES
    return normalized_value == normalized_filter


def _validate_cached_normalization(
    manifest: NormalizationManifest,
    *,
    raw_manifest_hash: str,
    configuration_digest: str,
    project_root: Path,
) -> None:
    if (
        manifest.raw_manifest_sha256 != raw_manifest_hash
        or manifest.configuration_hash != configuration_digest
    ):
        raise NormalizationError(
            "Existing normalized data was created from different raw/configuration inputs. "
            "Refusing to overwrite it; use --force after reviewing the change."
        )
    output_path = _resolve_project_path(manifest.output_file_path, project_root)
    if not output_path.is_file() or sha256_file(output_path) != manifest.output_file_sha256:
        raise NormalizationError(
            "Existing normalized Parquet is missing or fails its recorded hash. "
            "Use --force only after inspection."
        )


def _portable_project_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _resolve_project_path(path_text: str, project_root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else project_root / path


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _format_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--raw-manifest", type=Path, default=Path("data/raw/data_manifest.json"))
    parser.add_argument("--interim-dir", type=Path, default=Path("data/interim"))
    parser.add_argument("--force", action="store_true", help="Replace normalized output.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic normalizer CLI."""
    args = _parser().parse_args(argv)
    settings = Settings.load(args.config)
    configure_logging(settings.log_level)
    logger = get_logger(__name__)

    try:
        manifest = normalize_dataset(
            raw_manifest_path=args.raw_manifest,
            interim_dir=args.interim_dir,
            project_root=Path.cwd(),
            language_filter=settings.project_config.dataset.language_filter,
            configuration_digest=configuration_hash(settings),
            force=args.force,
        )
    except (NormalizationError, FileNotFoundError) as exc:
        logger.error("dataset_normalization_failed", error=str(exc))
        return 1

    print(json.dumps(manifest.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
