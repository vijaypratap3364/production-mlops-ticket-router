"""Tests for structured logging setup."""

from __future__ import annotations

import json

import pytest

from ticket_router.logging_config import configure_logging, get_logger


def test_json_logging_contains_structured_context(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO", json_logs=True)

    get_logger("ticket-router-test").info("settings_loaded", component="unit-test")

    event = json.loads(capsys.readouterr().out)
    assert event["event"] == "settings_loaded"
    assert event["component"] == "unit-test"
    assert event["level"] == "info"
    assert event["logger"] == "ticket-router-test"
    assert "timestamp" in event


def test_unknown_log_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown log level"):
        configure_logging("VERBOSE")
