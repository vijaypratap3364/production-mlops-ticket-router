"""CLI compatibility wrapper for monitoring reference generation."""

from __future__ import annotations

from ticket_router.monitoring.reference import main

if __name__ == "__main__":
    raise SystemExit(main())
