"""Typed FastAPI client parsing, feedback, and safe failure tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

from ticket_router.api.schemas import TicketRequest
from ticket_router.dashboard.api_client import DashboardAPIError, TicketRouterAPIClient


def _prediction(*, request_id: str = "request-1") -> dict[str, object]:
    return {
        "request_id": request_id,
        "predicted_queue": "Billing",
        "confidence": 0.8,
        "top_k": [
            {"queue": "Billing", "confidence": 0.8},
            {"queue": "Technical", "confidence": 0.2},
        ],
        "model_name": "fixture-router",
        "model_version": "7",
        "prediction_timestamp": datetime(2026, 8, 8, tzinfo=UTC).isoformat(),
        "warning": None,
    }


def _client(transport: httpx.MockTransport) -> TicketRouterAPIClient:
    return TicketRouterAPIClient(
        base_url="http://api.test",
        timeout_seconds=1.0,
        transport=transport,
    )


def test_prediction_and_batch_responses_are_parsed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/predict":
            return httpx.Response(200, json=_prediction(), request=request)
        return httpx.Response(
            200,
            json={"predictions": [_prediction(request_id="batch-1")]},
            request=request,
        )

    with _client(httpx.MockTransport(handler)) as client:
        single = client.predict(subject="Invoice", body="Please review")
        batch = client.predict_batch((TicketRequest(subject="Invoice", body="Review"),))

    assert single.prediction.predicted_queue == "Billing"
    assert single.api_latency_ms >= 0.0
    assert batch.response.predictions[0].request_id == "batch-1"
    assert batch.api_latency_ms >= 0.0


def test_feedback_submission_sends_only_the_declared_contract() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            201,
            json={
                "request_id": "request-1",
                "feedback_id": "feedback-1",
                "corrected_queue": "Technical",
                "recorded_at": datetime(2026, 8, 8, tzinfo=UTC).isoformat(),
            },
            request=request,
        )

    with _client(httpx.MockTransport(handler)) as client:
        response = client.submit_feedback(
            request_id="request-1",
            corrected_queue="Technical",
            accepted=False,
            comment="Reviewed fixture",
        )

    assert response.feedback_id == "feedback-1"
    assert captured == {
        "request_id": "request-1",
        "corrected_queue": "Technical",
        "accepted": False,
        "comment": "Reviewed fixture",
        "source": "demo",
    }


def test_api_error_uses_sanitized_server_message() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            503,
            json={"error": {"message": "The champion model is unavailable."}},
            request=request,
        )
    )

    with _client(transport) as client, pytest.raises(DashboardAPIError, match="champion model"):
        client.model_metadata()


def test_invalid_response_contract_is_rejected() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"unexpected": True}, request=request)
    )

    with _client(transport) as client, pytest.raises(DashboardAPIError, match="response contract"):
        client.health()


def test_readiness_parses_expected_503_response() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            503,
            json={"ready": False, "model_ready": False, "database_ready": True},
            request=request,
        )
    )

    with _client(transport) as client:
        readiness = client.readiness()

    assert readiness.ready is False
