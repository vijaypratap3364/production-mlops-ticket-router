"""Revision-pinned Hugging Face dataset download and raw manifest creation."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from huggingface_hub import HfApi, snapshot_download

from ticket_router.config import Settings
from ticket_router.data.manifests import GitVersion, RawDataManifest, get_git_version
from ticket_router.hashing import sha256_file, sha256_json
from ticket_router.logging_config import configure_logging, get_logger

RAW_MANIFEST_FILENAME = "data_manifest.json"
RAW_SOURCE_DIRECTORY_NAME = "source"
FULL_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class DatasetDownloadError(RuntimeError):
    """Raised when an immutable dataset snapshot cannot be prepared."""


class ExistingRawDataError(DatasetDownloadError):
    """Raised when existing raw data would be replaced without explicit force."""


@dataclass(frozen=True)
class SourceMetadata:
    """Metadata resolved from the remote dataset repository."""

    resolved_revision: str | None
    dataset_license: str | None


class DatasetSourceClient(Protocol):
    """Boundary used to isolate network access in tests."""

    def resolve(self, repository: str, revision: str) -> SourceMetadata:
        """Resolve immutable source metadata."""

    def download(self, repository: str, revision: str, destination: Path) -> Path:
        """Download a snapshot into destination."""


class HuggingFaceDatasetSource:
    """Public Hugging Face Hub implementation of the source boundary."""

    def __init__(self) -> None:
        self._api = HfApi()

    def resolve(self, repository: str, revision: str) -> SourceMetadata:
        info = self._api.dataset_info(repo_id=repository, revision=revision)
        card_data = getattr(info, "card_data", None)
        if isinstance(card_data, Mapping):
            license_value = card_data.get("license")
        else:
            license_value = getattr(card_data, "license", None) if card_data is not None else None
        return SourceMetadata(
            resolved_revision=getattr(info, "sha", None),
            dataset_license=str(license_value) if license_value else None,
        )

    def download(self, repository: str, revision: str, destination: Path) -> Path:
        downloaded_path = snapshot_download(
            repo_id=repository,
            repo_type="dataset",
            revision=revision,
            local_dir=destination,
            max_workers=1,
        )
        return Path(downloaded_path)


def configuration_hash(settings: Settings) -> str:
    """Hash version-controlled project settings used by ingestion."""
    return sha256_json(settings.project_config.model_dump(mode="json"))


def download_dataset(
    *,
    repository: str,
    revision: str,
    raw_dir: Path,
    project_root: Path,
    configuration_digest: str,
    force: bool = False,
    source_client: DatasetSourceClient | None = None,
    clock: Callable[[], datetime] | None = None,
    code_version: GitVersion | None = None,
) -> RawDataManifest:
    """Download or verify an immutable raw dataset snapshot."""
    _validate_revision(revision)
    resolved_project_root = project_root.resolve()
    resolved_raw_dir = raw_dir.resolve()
    resolved_raw_dir.mkdir(parents=True, exist_ok=True)
    source_dir = resolved_raw_dir / RAW_SOURCE_DIRECTORY_NAME
    manifest_path = resolved_raw_dir / RAW_MANIFEST_FILENAME

    if not force and manifest_path.exists():
        manifest = RawDataManifest.read(manifest_path)
        _validate_cached_manifest(
            manifest,
            repository=repository,
            revision=revision,
            configuration_digest=configuration_digest,
            project_root=resolved_project_root,
        )
        return manifest

    if not force and source_dir.exists() and any(source_dir.iterdir()):
        raise ExistingRawDataError(
            f"Raw data already exists at {source_dir} without a valid manifest. "
            "Refusing to replace it; inspect the directory or rerun with --force."
        )

    client = source_client or HuggingFaceDatasetSource()
    # Hugging Face creates nested cache metadata below ``local_dir``. Staging in
    # the ignored, shorter work path avoids Windows MAX_PATH failures while the
    # final immutable snapshot still lives under data/raw.
    staging_parent = resolved_project_root / "work" / "ingestion"
    staging_parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix="download-", dir=staging_parent))
    temporary_source = temporary_root / RAW_SOURCE_DIRECTORY_NAME

    try:
        try:
            source_metadata = client.resolve(repository, revision)
            downloaded_source = client.download(repository, revision, temporary_source).resolve()
        except Exception as exc:
            raise DatasetDownloadError(
                f"Unable to download dataset '{repository}' at revision '{revision}'. "
                "Check network availability, the repository name, and revision. "
                f"Underlying error: {_safe_exception_message(exc)}. "
                "Any existing raw snapshot was left unchanged."
            ) from exc

        if not downloaded_source.is_dir():
            raise DatasetDownloadError(
                f"Dataset source returned a non-directory snapshot: {downloaded_source}"
            )

        # Hub transfer metadata is neither source data nor required for the
        # immutable local snapshot; excluding it also prevents machine-local
        # cache details from entering manifests.
        shutil.rmtree(downloaded_source / ".cache", ignore_errors=True)

        manifest = _build_raw_manifest(
            downloaded_source=downloaded_source,
            final_source_dir=source_dir,
            project_root=resolved_project_root,
            repository=repository,
            revision=revision,
            source_metadata=source_metadata,
            configuration_digest=configuration_digest,
            downloaded_at=(clock or _utc_now)(),
            code_version=code_version or get_git_version(resolved_project_root),
        )
        _replace_raw_snapshot(
            downloaded_source=downloaded_source,
            source_dir=source_dir,
            manifest=manifest,
            manifest_path=manifest_path,
        )
        return manifest
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def _validate_revision(revision: str) -> None:
    if not FULL_GIT_SHA_PATTERN.fullmatch(revision):
        raise DatasetDownloadError(
            "Dataset revision must be a full 40-character lowercase Git SHA; "
            f"received {revision!r}. Mutable names such as 'main' are not allowed."
        )


def _safe_exception_message(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    message = re.sub(
        r"(?i)(token|authorization|password)=([^&\s]+)",
        r"\1=<redacted>",
        message,
    )
    message = re.sub(r"hf_[A-Za-z0-9]{20,}", "hf_<redacted>", message)
    return message[:500]


def _build_raw_manifest(
    *,
    downloaded_source: Path,
    final_source_dir: Path,
    project_root: Path,
    repository: str,
    revision: str,
    source_metadata: SourceMetadata,
    configuration_digest: str,
    downloaded_at: datetime,
    code_version: GitVersion | None,
) -> RawDataManifest:
    source_files = _source_files(downloaded_source)
    if not source_files:
        raise DatasetDownloadError("The downloaded dataset snapshot contained no source files.")

    csv_files = [path for path in source_files if path.suffix.casefold() == ".csv"]
    if not csv_files:
        raise DatasetDownloadError("The downloaded dataset snapshot contained no CSV data files.")

    row_count, column_names = _inspect_csv_files(csv_files)
    raw_file_paths: list[str] = []
    file_sha256: dict[str, str] = {}
    file_size_bytes: dict[str, int] = {}

    for source_file in source_files:
        relative_source_path = source_file.relative_to(downloaded_source)
        final_path = final_source_dir / relative_source_path
        manifest_path = _portable_project_path(final_path, project_root)
        raw_file_paths.append(manifest_path)
        file_sha256[manifest_path] = sha256_file(source_file)
        file_size_bytes[manifest_path] = source_file.stat().st_size

    return RawDataManifest(
        source_repository=repository,
        requested_revision=revision,
        resolved_revision=source_metadata.resolved_revision,
        download_timestamp_utc=_format_utc(downloaded_at),
        raw_file_paths=tuple(raw_file_paths),
        row_count=row_count,
        column_names=column_names,
        file_sha256=file_sha256,
        file_size_bytes=file_size_bytes,
        dataset_license=source_metadata.dataset_license,
        ingestion_code_version=code_version,
        configuration_hash=configuration_digest,
    )


def _source_files(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(source_dir).parts
    )


def _inspect_csv_files(csv_files: Sequence[Path]) -> tuple[int, tuple[str, ...]]:
    row_count = 0
    columns: list[str] = []

    for path in sorted(csv_files):
        with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            sample = csv_file.read(65536)
            csv_file.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            except csv.Error:
                dialect = csv.excel
            reader = csv.reader(csv_file, dialect=dialect)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise DatasetDownloadError(f"CSV file is empty: {path}") from exc
            for column in header:
                if column not in columns:
                    columns.append(column)
            row_count += sum(1 for row in reader if any(cell.strip() for cell in row))

    return row_count, tuple(columns)


def _replace_raw_snapshot(
    *,
    downloaded_source: Path,
    source_dir: Path,
    manifest: RawDataManifest,
    manifest_path: Path,
) -> None:
    backup_dir = source_dir.with_name(f".{source_dir.name}.backup-{uuid4().hex}")
    had_existing_source = source_dir.exists()

    try:
        if had_existing_source:
            source_dir.replace(backup_dir)
        shutil.move(str(downloaded_source), str(source_dir))
        manifest.write(manifest_path)
    except Exception:
        if source_dir.exists():
            shutil.rmtree(source_dir, ignore_errors=True)
        if backup_dir.exists():
            backup_dir.replace(source_dir)
        raise
    else:
        shutil.rmtree(backup_dir, ignore_errors=True)


def _validate_cached_manifest(
    manifest: RawDataManifest,
    *,
    repository: str,
    revision: str,
    configuration_digest: str,
    project_root: Path,
) -> None:
    if (
        manifest.source_repository != repository
        or manifest.requested_revision != revision
        or manifest.configuration_hash != configuration_digest
    ):
        raise ExistingRawDataError(
            "Existing raw data was created from different source/configuration settings. "
            "Refusing to replace it; use --force after reviewing the change."
        )

    for path_text, expected_hash in manifest.file_sha256.items():
        path = _resolve_project_path(path_text, project_root)
        if not path.is_file():
            raise ExistingRawDataError(
                f"Cached raw file is missing: {path}. Use --force to rebuild the snapshot."
            )
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ExistingRawDataError(
                f"Cached raw file hash mismatch: {path}. Use --force only after inspection."
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
    parser.add_argument("--repository", help="Override the configured Hub dataset repository.")
    parser.add_argument("--revision", help="Override the configured full Hub revision SHA.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true", help="Replace verified existing raw data.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the revision-pinned downloader CLI."""
    args = _parser().parse_args(argv)
    settings = Settings.load(args.config)
    configure_logging(settings.log_level)
    logger = get_logger(__name__)
    dataset_settings = settings.project_config.dataset
    repository = args.repository or dataset_settings.repository
    revision = args.revision or dataset_settings.revision

    try:
        manifest = download_dataset(
            repository=repository,
            revision=revision,
            raw_dir=args.raw_dir,
            project_root=Path.cwd(),
            configuration_digest=configuration_hash(settings),
            force=args.force,
        )
    except DatasetDownloadError as exc:
        logger.error("dataset_download_failed", error=str(exc))
        return 1

    print(
        json.dumps(
            {
                "manifest": str(args.raw_dir / RAW_MANIFEST_FILENAME),
                "repository": manifest.source_repository,
                "requested_revision": manifest.requested_revision,
                "resolved_revision": manifest.resolved_revision,
                "row_count": manifest.row_count,
                "column_names": manifest.column_names,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
