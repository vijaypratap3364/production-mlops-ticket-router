"""Typed manifests for immutable raw and normalized datasets."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

SOURCE_GIT_COMMIT_ENV = "SOURCE_GIT_COMMIT"
SOURCE_GIT_DIRTY_ENV = "SOURCE_GIT_DIRTY"


class GitVersion(BaseModel):
    """Best-effort source-code identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    dirty: bool


class RawDataManifest(BaseModel):
    """Lineage and integrity metadata for a raw Hub snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_repository: str
    requested_revision: str
    resolved_revision: str | None
    download_timestamp_utc: str
    raw_file_paths: tuple[str, ...]
    row_count: int = Field(ge=0)
    column_names: tuple[str, ...]
    file_sha256: dict[str, str]
    file_size_bytes: dict[str, int]
    dataset_license: str | None
    ingestion_code_version: GitVersion | None
    configuration_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def read(cls, path: Path) -> Self:
        """Load and validate a raw-data manifest."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def write(self, path: Path) -> None:
        """Write the manifest atomically."""
        atomic_write_json(path, self.model_dump(mode="json"))


class NormalizationManifest(BaseModel):
    """Lineage and integrity metadata for normalized Parquet data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    normalization_timestamp_utc: str
    source_repository: str
    requested_revision: str
    resolved_revision: str | None
    raw_manifest_path: str
    raw_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_file_sha256: dict[str, str]
    output_file_path: str
    output_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rows_read: int = Field(ge=0)
    rows_dropped_malformed: int = Field(ge=0)
    english_rows: int = Field(ge=0)
    rows_dropped_missing_queue: int = Field(ge=0)
    rows_dropped_missing_text: int = Field(ge=0)
    output_row_count: int = Field(ge=0)
    output_column_names: tuple[str, ...]
    configuration_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalization_code_version: GitVersion | None

    @classmethod
    def read(cls, path: Path) -> Self:
        """Load and validate a normalization manifest."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def write(self, path: Path) -> None:
        """Write the manifest atomically."""
        atomic_write_json(path, self.model_dump(mode="json"))


def atomic_write_json(path: Path, value: object) -> None:
    """Write JSON through a sibling temporary file and atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def get_git_version(repository_root: Path) -> GitVersion | None:
    """Return source identity from release metadata or the local Git worktree."""
    environment_commit = os.getenv(SOURCE_GIT_COMMIT_ENV, "").strip().lower()
    if environment_commit:
        dirty_text = os.getenv(SOURCE_GIT_DIRTY_ENV, "false").strip().casefold()
        if dirty_text not in {"true", "false"}:
            raise ValueError(f"{SOURCE_GIT_DIRTY_ENV} must be 'true' or 'false'")
        return GitVersion(commit=environment_commit, dirty=dirty_text == "true")
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    return GitVersion(
        commit=commit_result.stdout.strip(),
        dirty=bool(status_result.stdout.strip()),
    )
