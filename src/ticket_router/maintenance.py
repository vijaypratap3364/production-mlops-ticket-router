"""Bounded maintenance operations for the local repository."""

from __future__ import annotations

import shutil
from pathlib import Path

CACHE_DIRECTORIES = (
    ".mypy_cache",
    ".pytest_cache",
    ".pytest-tmp",
    ".ruff_cache",
    "htmlcov",
)
CACHE_FILES = (
    ".coverage",
    "coverage.xml",
)
SOURCE_DIRECTORIES = ("src", "tests", "scripts")
RECURSIVE_CACHE_DIRECTORY_NAMES = ("__pycache__",)


def clean_project_caches(root: Path) -> None:
    """Delete known project-local caches without touching data or artifacts."""
    resolved_root = root.resolve()

    for relative_path in CACHE_DIRECTORIES:
        target = (resolved_root / relative_path).resolve()
        if target.parent != resolved_root:
            raise ValueError(f"refusing to clean path outside project root: {target}")
        shutil.rmtree(target, ignore_errors=True)

    for relative_path in CACHE_FILES:
        target = (resolved_root / relative_path).resolve()
        if target.parent != resolved_root:
            raise ValueError(f"refusing to clean path outside project root: {target}")
        target.unlink(missing_ok=True)

    for source_directory in SOURCE_DIRECTORIES:
        search_root = resolved_root / source_directory
        if not search_root.is_dir():
            continue
        for directory_name in RECURSIVE_CACHE_DIRECTORY_NAMES:
            for target in search_root.rglob(directory_name):
                resolved_target = target.resolve()
                if resolved_root not in resolved_target.parents:
                    raise ValueError(
                        f"refusing to clean path outside project root: {resolved_target}"
                    )
                shutil.rmtree(resolved_target, ignore_errors=True)
