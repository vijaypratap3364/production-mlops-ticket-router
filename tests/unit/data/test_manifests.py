"""Tests for atomic manifests and container-safe source lineage."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ticket_router.data.manifests import get_git_version


def test_git_version_uses_explicit_container_lineage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SOURCE_GIT_COMMIT", "A" * 40)
    monkeypatch.setenv("SOURCE_GIT_DIRTY", "false")

    version = get_git_version(tmp_path)

    assert version is not None
    assert version.commit == "a" * 40
    assert version.dirty is False


def test_git_version_rejects_invalid_explicit_lineage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SOURCE_GIT_COMMIT", "not-a-commit")

    with pytest.raises(ValidationError):
        get_git_version(tmp_path)

    monkeypatch.setenv("SOURCE_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("SOURCE_GIT_DIRTY", "unknown")
    with pytest.raises(ValueError, match="SOURCE_GIT_DIRTY"):
        get_git_version(tmp_path)
