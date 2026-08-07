"""Delayed-label model quality metrics kept separate from unlabeled drift."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import f1_score, recall_score  # type: ignore[import-untyped]

from ticket_router.monitoring.contracts import LabeledPrediction


@dataclass(frozen=True)
class DelayedQualityResult:
    available: bool
    sample_count: int
    minimum_sample_count: int
    macro_f1: float | None
    weighted_f1: float | None
    per_class_recall: dict[str, float]
    correction_rate: float | None
    acceptance_rate: float | None
    mean_correct_confidence: float | None
    mean_incorrect_confidence: float | None
    quality_by_model_version: dict[str, dict[str, float | int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "sample_count": self.sample_count,
            "minimum_sample_count": self.minimum_sample_count,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "per_class_recall": self.per_class_recall,
            "correction_rate": self.correction_rate,
            "acceptance_rate": self.acceptance_rate,
            "mean_correct_confidence": self.mean_correct_confidence,
            "mean_incorrect_confidence": self.mean_incorrect_confidence,
            "quality_by_model_version": self.quality_by_model_version,
        }


def calculate_delayed_quality(
    events: tuple[LabeledPrediction, ...],
    *,
    minimum_sample_count: int,
) -> DelayedQualityResult:
    """Calculate quality only when the delayed-label sample is policy-sufficient."""
    if len(events) < minimum_sample_count:
        return DelayedQualityResult(
            available=False,
            sample_count=len(events),
            minimum_sample_count=minimum_sample_count,
            macro_f1=None,
            weighted_f1=None,
            per_class_recall={},
            correction_rate=None,
            acceptance_rate=_acceptance_rate(events),
            mean_correct_confidence=None,
            mean_incorrect_confidence=None,
            quality_by_model_version={},
        )
    predicted = [event.predicted_queue for event in events]
    actual = [event.corrected_queue for event in events]
    labels = sorted(set(predicted) | set(actual))
    recalls = recall_score(actual, predicted, labels=labels, average=None, zero_division=0)
    correct = cast(
        NDArray[np.bool_],
        np.asarray(
            [prediction == target for prediction, target in zip(predicted, actual, strict=True)],
            dtype=np.bool_,
        ),
    )
    confidences = np.asarray([event.confidence for event in events], dtype=np.float64)
    versions = sorted({event.model_version for event in events})
    return DelayedQualityResult(
        available=True,
        sample_count=len(events),
        minimum_sample_count=minimum_sample_count,
        macro_f1=float(f1_score(actual, predicted, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(actual, predicted, average="weighted", zero_division=0)),
        per_class_recall={
            label: float(value) for label, value in zip(labels, recalls, strict=True)
        },
        correction_rate=float(1.0 - correct.mean()),
        acceptance_rate=_acceptance_rate(events),
        mean_correct_confidence=_conditional_mean(confidences, correct),
        mean_incorrect_confidence=_conditional_mean(confidences, ~correct),
        quality_by_model_version={
            version: _version_quality(
                tuple(event for event in events if event.model_version == version)
            )
            for version in versions
        },
    )


def _version_quality(events: tuple[LabeledPrediction, ...]) -> dict[str, float | int]:
    predicted = [event.predicted_queue for event in events]
    actual = [event.corrected_queue for event in events]
    return {
        "sample_count": len(events),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(actual, predicted, average="weighted", zero_division=0)),
        "correction_rate": float(
            sum(prediction != target for prediction, target in zip(predicted, actual, strict=True))
            / len(events)
        ),
    }


def _acceptance_rate(events: tuple[LabeledPrediction, ...]) -> float | None:
    observed = [event.accepted for event in events if event.accepted is not None]
    return float(sum(observed) / len(observed)) if observed else None


def _conditional_mean(
    values: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> float | None:
    selected = values[mask]
    return float(selected.mean()) if selected.size else None
