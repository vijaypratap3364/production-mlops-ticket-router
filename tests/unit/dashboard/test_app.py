"""Headless Streamlit navigation smoke test."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


def test_streamlit_app_has_unique_pages_and_no_render_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DASHBOARD_API_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("DASHBOARD_REQUEST_TIMEOUT_SECONDS", "0.1")
    app_path = Path(__file__).parents[3] / "src/ticket_router/dashboard/app.py"

    app = AppTest.from_file(str(app_path)).run(timeout=30)

    assert not app.exception
