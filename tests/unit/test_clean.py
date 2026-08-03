"""Tests for the bounded local cleanup utility."""

from __future__ import annotations

from pathlib import Path

from ticket_router.maintenance import (
    CACHE_DIRECTORIES,
    CACHE_FILES,
    clean_project_caches,
)


def test_clean_removes_only_known_generated_paths(tmp_path: Path) -> None:
    for relative_path in CACHE_DIRECTORIES:
        cache_directory = tmp_path / relative_path
        cache_directory.mkdir()
        (cache_directory / "generated.txt").write_text("generated", encoding="utf-8")

    for relative_path in CACHE_FILES:
        (tmp_path / relative_path).write_text("generated", encoding="utf-8")

    preserved_file = tmp_path / "keep.txt"
    preserved_file.write_text("preserve", encoding="utf-8")
    nested_cache = tmp_path / "src" / "package" / "__pycache__"
    nested_cache.mkdir(parents=True)
    (nested_cache / "module.pyc").write_bytes(b"generated")
    excluded_cache = tmp_path / ".venv" / "package" / "__pycache__"
    excluded_cache.mkdir(parents=True)
    (excluded_cache / "dependency.pyc").write_bytes(b"preserve")

    clean_project_caches(tmp_path)

    assert all(not (tmp_path / path).exists() for path in CACHE_DIRECTORIES)
    assert all(not (tmp_path / path).exists() for path in CACHE_FILES)
    assert not nested_cache.exists()
    assert excluded_cache.exists()
    assert preserved_file.read_text(encoding="utf-8") == "preserve"
