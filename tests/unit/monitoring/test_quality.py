"""Delayed-label quality tests."""

import pytest

from ticket_router.monitoring.contracts import LabeledPrediction
from ticket_router.monitoring.quality import calculate_delayed_quality


def _event(
    predicted: str,
    corrected: str,
    *,
    confidence: float,
    accepted: bool | None,
    version: str,
) -> LabeledPrediction:
    return LabeledPrediction(predicted, corrected, confidence, accepted, version)


def test_delayed_label_quality_and_model_versions() -> None:
    events = (
        _event("A", "A", confidence=0.9, accepted=True, version="7"),
        _event("A", "B", confidence=0.8, accepted=False, version="7"),
        _event("B", "B", confidence=0.7, accepted=True, version="8"),
        _event("B", "A", confidence=0.4, accepted=None, version="8"),
    )

    result = calculate_delayed_quality(events, minimum_sample_count=4)

    assert result.available is True
    assert result.macro_f1 == pytest.approx(0.5)
    assert result.weighted_f1 == pytest.approx(0.5)
    assert result.correction_rate == pytest.approx(0.5)
    assert result.acceptance_rate == pytest.approx(2 / 3)
    assert set(result.quality_by_model_version) == {"7", "8"}
    assert result.mean_correct_confidence == pytest.approx(0.8)
    assert result.mean_incorrect_confidence == pytest.approx(0.6)


def test_quality_requires_minimum_delayed_labels() -> None:
    result = calculate_delayed_quality((), minimum_sample_count=5)

    assert result.available is False
    assert result.macro_f1 is None
    assert result.sample_count == 0
