"""Framework-neutral persistence records and repository ports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class PredictionEvent:
    """Privacy-safe prediction metadata; raw subject/body are deliberately absent."""

    request_id: str
    created_at: datetime
    model_name: str
    model_version: str
    predicted_queue: str
    confidence: float
    top_k: tuple[dict[str, str | float], ...]
    subject_length: int
    body_length: int
    word_count: int
    combined_length: int
    uppercase_ratio: float
    digit_ratio: float
    punctuation_ratio: float
    url_count: int
    email_marker_count: int
    language_indicator: str | None
    low_confidence: bool
    latency_ms: float
    redacted_text: str | None
    text_hash: str
    text_hash_algorithm: str
    request_metadata: dict[str, str]


@dataclass(frozen=True)
class FeedbackEvent:
    feedback_id: str
    request_id: str
    corrected_queue: str
    accepted: bool | None
    comment: str | None
    source: str
    created_at: datetime


@dataclass(frozen=True)
class StoredFeedback:
    feedback_id: str
    request_id: str
    corrected_queue: str
    accepted: bool | None
    comment: str | None
    source: str
    model_version: str
    created_at: datetime


@dataclass(frozen=True)
class MonitoringRun:
    run_id: str
    started_at: datetime
    completed_at: datetime | None
    reference_period_start: datetime
    reference_period_end: datetime
    current_period_start: datetime
    current_period_end: datetime
    drift_status: str
    report_paths: tuple[str, ...]
    summary: dict[str, object]


@dataclass(frozen=True)
class RetrainingRun:
    run_id: str
    trigger_reason: str
    source_data_period_start: datetime
    source_data_period_end: datetime
    status: str
    mlflow_run_id: str | None
    candidate_model_version: str | None
    gate_results: dict[str, object]
    started_at: datetime
    completed_at: datetime | None


class PredictionFeedbackRepository(Protocol):
    def save_predictions(self, events: tuple[PredictionEvent, ...]) -> None: ...

    def get_prediction(self, request_id: str) -> PredictionEvent | None: ...

    def save_feedback(self, event: FeedbackEvent) -> StoredFeedback: ...

    def get_feedback_for_request(self, request_id: str) -> StoredFeedback | None: ...

    def close(self) -> None: ...


class MonitoringRunRepository(Protocol):
    def save(self, run: MonitoringRun) -> None: ...

    def get(self, run_id: str) -> MonitoringRun | None: ...

    def list_recent(self, *, limit: int) -> tuple[MonitoringRun, ...]: ...


class RetrainingRunRepository(Protocol):
    def save(self, run: RetrainingRun) -> None: ...

    def get(self, run_id: str) -> RetrainingRun | None: ...

    def list_recent(self, *, limit: int) -> tuple[RetrainingRun, ...]: ...
