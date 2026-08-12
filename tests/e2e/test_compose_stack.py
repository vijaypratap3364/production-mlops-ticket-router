"""Opt-in end-to-end verification against an already-running local Compose stack."""

from __future__ import annotations

import os

import pytest
from scripts.compose_smoke import main


def test_live_compose_prediction_feedback_and_persistence() -> None:
    if os.getenv("RUN_COMPOSE_E2E") != "1":
        pytest.skip("set RUN_COMPOSE_E2E=1 after starting the prepared Compose stack")
    database_url = os.getenv("COMPOSE_E2E_DATABASE_URL")
    if not database_url:
        pytest.fail("COMPOSE_E2E_DATABASE_URL is required for live persistence verification")

    exit_code = main(
        [
            "--api-url",
            os.getenv("COMPOSE_E2E_API_URL", "http://127.0.0.1:8000"),
            "--dashboard-url",
            os.getenv("COMPOSE_E2E_DASHBOARD_URL", "http://127.0.0.1:8501"),
            "--database-url",
            database_url,
            "--ready-timeout-seconds",
            "30",
        ]
    )

    assert exit_code == 0
