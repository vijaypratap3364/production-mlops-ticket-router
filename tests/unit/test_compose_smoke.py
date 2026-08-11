"""End-to-end smoke protocol tests without requiring Docker or network access."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from scripts.compose_smoke import SmokeTestError, execute_smoke_test


class FakeTransport:
    def __init__(self, *, ready: bool = True) -> None:
        self.ready = ready
        self.dashboard_checked = False
        self.posts: list[tuple[str, Mapping[str, object]]] = []

    def get_json(self, path: str) -> dict[str, Any]:
        if path == "/health":
            return {"status": "ok"}
        if path == "/ready":
            return {
                "ready": self.ready,
                "model_ready": self.ready,
                "database_ready": True,
            }
        raise AssertionError(path)

    def post_json(self, path: str, payload: Mapping[str, object]) -> dict[str, Any]:
        self.posts.append((path, payload))
        if path == "/predict":
            return {
                "request_id": "request-1",
                "predicted_queue": "Billing",
                "model_version": "7",
            }
        if path == "/feedback":
            return {"feedback_id": "feedback-1"}
        raise AssertionError(path)

    def dashboard_health(self) -> None:
        self.dashboard_checked = True


def test_smoke_protocol_checks_prediction_feedback_persistence_and_dashboard() -> None:
    transport = FakeTransport()
    persisted: list[str] = []

    result = execute_smoke_test(
        transport=transport,
        persistence_check=persisted.append,
        ready_timeout_seconds=1.0,
        poll_interval_seconds=0.0,
    )

    assert result.request_id == "request-1"
    assert persisted == ["request-1"]
    assert transport.dashboard_checked is True
    assert [path for path, _ in transport.posts] == ["/predict", "/feedback"]
    assert transport.posts[1][1]["corrected_queue"] == "Billing"


def test_smoke_protocol_fails_when_champion_never_becomes_ready() -> None:
    with pytest.raises(SmokeTestError, match="did not become ready"):
        execute_smoke_test(
            transport=FakeTransport(ready=False),
            persistence_check=lambda _: None,
            ready_timeout_seconds=0.0,
            poll_interval_seconds=0.0,
        )
