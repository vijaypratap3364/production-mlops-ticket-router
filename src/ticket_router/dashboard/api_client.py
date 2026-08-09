"""Reusable HTTP client for the Streamlit presentation adapter."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from types import TracebackType
from typing import Any, Self, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from ticket_router.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelMetadataResponse,
    MonitoringHistoryResponse,
    PredictionResponse,
    ReadinessResponse,
    SystemStatusResponse,
    TicketRequest,
)

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class DashboardAPIError(RuntimeError):
    """Safe client-facing error with no request payload or response body attached."""


@dataclass(frozen=True)
class TimedPrediction:
    prediction: PredictionResponse
    api_latency_ms: float


@dataclass(frozen=True)
class TimedBatchPrediction:
    response: BatchPredictionResponse
    api_latency_ms: float


class TicketRouterAPIClient:
    """Typed, bounded access to the FastAPI service used by every dashboard page."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
            headers={"User-Agent": "ticket-router-streamlit/0.1"},
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def health(self) -> HealthResponse:
        return self._request("GET", "/health", response_model=HealthResponse)

    def readiness(self) -> ReadinessResponse:
        return self._request(
            "GET",
            "/ready",
            response_model=ReadinessResponse,
            accepted_statuses=(200, 503),
        )

    def model_metadata(self) -> ModelMetadataResponse:
        return self._request("GET", "/model", response_model=ModelMetadataResponse)

    def predict(self, *, subject: str, body: str) -> TimedPrediction:
        payload = TicketRequest(subject=subject, body=body)
        started = perf_counter()
        prediction = self._request(
            "POST",
            "/predict",
            response_model=PredictionResponse,
            json=payload.model_dump(mode="json", exclude_none=True),
        )
        return TimedPrediction(
            prediction=prediction,
            api_latency_ms=(perf_counter() - started) * 1000.0,
        )

    def predict_batch(self, items: Sequence[TicketRequest]) -> TimedBatchPrediction:
        payload = BatchPredictionRequest(items=list(items))
        started = perf_counter()
        response = self._request(
            "POST",
            "/predict/batch",
            response_model=BatchPredictionResponse,
            json=payload.model_dump(mode="json", exclude_none=True),
        )
        return TimedBatchPrediction(
            response=response,
            api_latency_ms=(perf_counter() - started) * 1000.0,
        )

    def submit_feedback(
        self,
        *,
        request_id: str,
        corrected_queue: str,
        accepted: bool,
        comment: str | None,
    ) -> FeedbackResponse:
        payload = FeedbackRequest(
            request_id=request_id,
            corrected_queue=corrected_queue,
            accepted=accepted,
            comment=comment,
            source="demo",
        )
        return self._request(
            "POST",
            "/feedback",
            response_model=FeedbackResponse,
            json=payload.model_dump(mode="json", exclude_none=True),
            accepted_statuses=(201,),
        )

    def monitoring_history(self, *, limit: int = 30) -> MonitoringHistoryResponse:
        return self._request(
            "GET",
            "/monitoring/history",
            response_model=MonitoringHistoryResponse,
            params={"limit": limit},
        )

    def system_status(self) -> SystemStatusResponse:
        return self._request("GET", "/system/status", response_model=SystemStatusResponse)

    def _request(
        self,
        method: str,
        path: str,
        *,
        response_model: type[ResponseModel],
        accepted_statuses: tuple[int, ...] = (200,),
        **kwargs: Any,
    ) -> ResponseModel:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise DashboardAPIError("The API request timed out. Please try again.") from exc
        except httpx.RequestError as exc:
            raise DashboardAPIError(
                "The local API is unavailable. Confirm that FastAPI is running."
            ) from exc
        if response.status_code not in accepted_statuses:
            raise DashboardAPIError(_safe_error_message(response))
        try:
            return response_model.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise DashboardAPIError("The API returned an unexpected response contract.") from exc


def _safe_error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
    return f"The API request failed with status {response.status_code}."
