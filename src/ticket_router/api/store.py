"""Prediction/feedback persistence port and local non-persistent adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Protocol

from ticket_router.api.errors import DatabaseUnavailableError, UnknownPredictionError


@dataclass(frozen=True)
class PredictionEvent:
    """Privacy-safe prediction record; raw subject/body are deliberately absent."""

    request_id: str
    created_at: datetime
    predicted_queue: str
    confidence: float
    model_name: str
    model_version: str
    subject_length: int
    body_length: int
    combined_length: int


@dataclass(frozen=True)
class FeedbackEvent:
    feedback_id: str
    request_id: str
    corrected_queue: str
    accepted: bool | None
    comment: str | None
    created_at: datetime


class PredictionFeedbackStore(Protocol):
    """Persistence behavior required by prediction and feedback use cases."""

    def save_predictions(self, events: tuple[PredictionEvent, ...]) -> None: ...

    def save_feedback(self, event: FeedbackEvent) -> None: ...

    def close(self) -> None: ...


class InMemoryPredictionFeedbackStore:
    """Thread-safe local adapter used until PostgreSQL migrations are introduced."""

    def __init__(self) -> None:
        self._predictions: dict[str, PredictionEvent] = {}
        self._feedback: dict[str, FeedbackEvent] = {}
        self._available = True
        self._lock = Lock()

    def save_predictions(self, events: tuple[PredictionEvent, ...]) -> None:
        with self._lock:
            self._require_available()
            self._predictions.update((event.request_id, event) for event in events)

    def save_feedback(self, event: FeedbackEvent) -> None:
        with self._lock:
            self._require_available()
            if event.request_id not in self._predictions:
                raise UnknownPredictionError(event.request_id)
            self._feedback[event.feedback_id] = event

    def close(self) -> None:
        with self._lock:
            self._available = False

    def _require_available(self) -> None:
        if not self._available:
            raise DatabaseUnavailableError("prediction store is closed")
