"""Command wrapper for bounded project-cache cleanup."""

from __future__ import annotations

from pathlib import Path

from ticket_router.maintenance import clean_project_caches

if __name__ == "__main__":
    clean_project_caches(Path.cwd())
