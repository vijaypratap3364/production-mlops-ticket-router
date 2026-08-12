"""Safe operational-validation protocol tests without Docker disruptions."""

from __future__ import annotations

import json

import httpx
from scripts.operational_validation import (
    _baseline_http_scenarios,
    _insufficient_monitoring_scenario,
    _missing_champion_scenario,
)


def test_baseline_operational_protocol_checks_validation_and_duplicate_feedback() -> None:
    feedback_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal feedback_calls
        if request.url.path == "/ready":
            return httpx.Response(200, json={"ready": True})
        if request.url.path == "/predict/batch":
            return httpx.Response(422, json={"error": {"code": "invalid_request"}})
        if request.url.path == "/predict" and request.content == b"{not-valid-json":
            return httpx.Response(422, json={"error": {"code": "invalid_request"}})
        if request.url.path == "/predict":
            return httpx.Response(
                200,
                json={"request_id": "fixture-request", "predicted_queue": "Billing"},
            )
        if request.url.path == "/feedback":
            feedback_calls += 1
            return httpx.Response(201 if feedback_calls == 1 else 409, json={})
        return httpx.Response(404, json={})

    with httpx.Client(
        base_url="http://127.0.0.1:8000",
        transport=httpx.MockTransport(handler),
    ) as client:
        scenarios = _baseline_http_scenarios(client, maximum_batch_size=2)

    assert len(scenarios) == 4
    assert all(scenario.passed for scenario in scenarios)
    assert [scenario.name for scenario in scenarios] == [
        "api_ready_before_validation",
        "malformed_request",
        "oversized_batch",
        "duplicate_feedback",
    ]


def test_local_missing_champion_and_insufficient_monitoring_fail_gracefully() -> None:
    missing = _missing_champion_scenario()
    insufficient = _insufficient_monitoring_scenario()

    assert missing.passed is True
    assert "readiness HTTP 503" in missing.observed
    assert insufficient.passed is True
    assert insufficient.observed == "status insufficient_data"


def test_operational_fixture_contains_no_ticket_payload_in_results() -> None:
    scenario = _insufficient_monitoring_scenario()
    serialized = json.dumps(scenario.model_dump(mode="json"))

    assert "subject" not in serialized
    assert "body" not in serialized
