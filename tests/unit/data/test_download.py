"""Tests for immutable, network-isolated raw dataset downloads."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ticket_router.data.download import (
    DatasetDownloadError,
    ExistingRawDataError,
    SourceMetadata,
    download_dataset,
)
from ticket_router.data.manifests import GitVersion, RawDataManifest
from ticket_router.hashing import sha256_file

REVISION = "a" * 40
CONFIGURATION_HASH = "c" * 64
FIXED_TIME = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)


class FakeSourceClient:
    def __init__(self, fixture_path: Path) -> None:
        self.fixture_path = fixture_path
        self.download_calls = 0

    def resolve(self, repository: str, revision: str) -> SourceMetadata:
        assert repository == "owner/tickets"
        assert revision == REVISION
        return SourceMetadata(resolved_revision=revision, dataset_license="cc-by-nc-4.0")

    def download(self, repository: str, revision: str, destination: Path) -> Path:
        self.download_calls += 1
        destination.mkdir(parents=True)
        shutil.copy2(self.fixture_path, destination / "tickets.csv")
        return destination


class OfflineSourceClient:
    def resolve(self, repository: str, revision: str) -> SourceMetadata:
        raise ConnectionError("offline")

    def download(self, repository: str, revision: str, destination: Path) -> Path:
        raise AssertionError("download must not be attempted")


@pytest.fixture
def fixture_path() -> Path:
    return Path("tests/fixtures/tickets.csv")


def test_download_generates_complete_manifest(tmp_path: Path, fixture_path: Path) -> None:
    client = FakeSourceClient(fixture_path)
    raw_dir = tmp_path / "data" / "raw"

    manifest = download_dataset(
        repository="owner/tickets",
        revision=REVISION,
        raw_dir=raw_dir,
        project_root=tmp_path,
        configuration_digest=CONFIGURATION_HASH,
        source_client=client,
        clock=lambda: FIXED_TIME,
        code_version=GitVersion(commit="b" * 40, dirty=False),
    )

    manifest_path = raw_dir / "data_manifest.json"
    raw_file = raw_dir / "source" / "tickets.csv"
    assert client.download_calls == 1
    assert manifest.source_repository == "owner/tickets"
    assert manifest.requested_revision == REVISION
    assert manifest.resolved_revision == REVISION
    assert manifest.download_timestamp_utc == "2026-08-03T12:00:00Z"
    assert manifest.row_count == 5
    assert manifest.column_names == (
        "Subject",
        "Body",
        "Answer",
        "Type",
        "Queue",
        "Priority",
        "Language",
        "Tags",
    )
    assert manifest.dataset_license == "cc-by-nc-4.0"
    assert manifest.file_sha256["data/raw/source/tickets.csv"] == sha256_file(raw_file)
    assert RawDataManifest.read(manifest_path) == manifest


def test_existing_verified_snapshot_is_reused_without_network(
    tmp_path: Path,
    fixture_path: Path,
) -> None:
    raw_dir = tmp_path / "data" / "raw"
    first_client = FakeSourceClient(fixture_path)
    first_manifest = download_dataset(
        repository="owner/tickets",
        revision=REVISION,
        raw_dir=raw_dir,
        project_root=tmp_path,
        configuration_digest=CONFIGURATION_HASH,
        source_client=first_client,
        clock=lambda: FIXED_TIME,
    )

    cached_manifest = download_dataset(
        repository="owner/tickets",
        revision=REVISION,
        raw_dir=raw_dir,
        project_root=tmp_path,
        configuration_digest=CONFIGURATION_HASH,
        source_client=OfflineSourceClient(),
    )

    assert cached_manifest == first_manifest
    assert first_client.download_calls == 1


def test_existing_snapshot_is_not_replaced_for_different_revision(
    tmp_path: Path,
    fixture_path: Path,
) -> None:
    raw_dir = tmp_path / "data" / "raw"
    download_dataset(
        repository="owner/tickets",
        revision=REVISION,
        raw_dir=raw_dir,
        project_root=tmp_path,
        configuration_digest=CONFIGURATION_HASH,
        source_client=FakeSourceClient(fixture_path),
    )

    with pytest.raises(ExistingRawDataError, match="Refusing to replace"):
        download_dataset(
            repository="owner/tickets",
            revision="d" * 40,
            raw_dir=raw_dir,
            project_root=tmp_path,
            configuration_digest=CONFIGURATION_HASH,
            source_client=OfflineSourceClient(),
        )


def test_network_failure_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(DatasetDownloadError, match="ConnectionError: offline"):
        download_dataset(
            repository="owner/tickets",
            revision=REVISION,
            raw_dir=tmp_path / "data" / "raw",
            project_root=tmp_path,
            configuration_digest=CONFIGURATION_HASH,
            source_client=OfflineSourceClient(),
        )
