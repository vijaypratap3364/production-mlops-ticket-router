"""Framework-neutral monitoring records and repository ports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class CurrentPrediction:
    request_id: str
    created_at: datetime
    model_version: str
    predicted_queue: str
    prediction_confidence: float
    low_confidence: bool
    subject_length: int
    body_length: int
    combined_length: int
    word_count: int
    uppercase_ratio: float
    digit_ratio: float
    punctuation_ratio: float
    url_count: int
    email_marker_count: int

    def monitoring_values(self) -> dict[str, str | int | float | bool]:
        return {
            "subject_length": self.subject_length,
            "body_length": self.body_length,
            "combined_length": self.combined_length,
            "word_count": self.word_count,
            "uppercase_ratio": self.uppercase_ratio,
            "digit_ratio": self.digit_ratio,
            "punctuation_ratio": self.punctuation_ratio,
            "url_count": self.url_count,
            "email_marker_count": self.email_marker_count,
            "predicted_queue": self.predicted_queue,
            "prediction_confidence": self.prediction_confidence,
            "low_confidence": self.low_confidence,
            "model_version": self.model_version,
        }


@dataclass(frozen=True)
class LabeledPrediction:
    predicted_queue: str
    corrected_queue: str
    confidence: float
    accepted: bool | None
    model_version: str


class MonitoringDataRepository(Protocol):
    def load_predictions(
        self,
        *,
        start: datetime,
        end: datetime,
        model_version: str | None = None,
    ) -> tuple[CurrentPrediction, ...]: ...

    def load_labeled_predictions(
        self,
        *,
        start: datetime,
        end: datetime,
        model_version: str | None = None,
    ) -> tuple[LabeledPrediction, ...]: ...
