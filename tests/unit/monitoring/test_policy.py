"""Multi-signal alert policy tests."""

from ticket_router.monitoring.config import MonitoringConfig
from ticket_router.monitoring.drift import ColumnDrift, DriftResult
from ticket_router.monitoring.policy import evaluate_alert_policy
from ticket_router.monitoring.quality import DelayedQualityResult, calculate_delayed_quality


def test_insufficient_data_status_is_explicit(monitoring_config: MonitoringConfig) -> None:
    result = evaluate_alert_policy(
        event_count=10,
        minimum_event_count=100,
        drift=None,
        quality=calculate_delayed_quality((), minimum_sample_count=50),
        reference_macro_f1=0.8,
        settings=monitoring_config.alerts,
    )

    assert result.status == "insufficient_data"
    assert result.signals == ()


def test_two_warning_signals_produce_warning(monitoring_config: MonitoringConfig) -> None:
    result = evaluate_alert_policy(
        event_count=100,
        minimum_event_count=100,
        drift=_drift(feature_share=0.30, distribution_score=0.11, confidence_change=0.0),
        quality=_quality(available=False, macro_f1=None),
        reference_macro_f1=0.8,
        settings=monitoring_config.alerts,
    )

    assert result.status == "warning"
    assert {signal.name for signal in result.signals} == {
        "drifted_input_feature_share",
        "prediction_distribution_score",
    }
    assert all(signal.severity == "warning" for signal in result.signals)


def test_two_critical_signals_produce_critical(monitoring_config: MonitoringConfig) -> None:
    result = evaluate_alert_policy(
        event_count=100,
        minimum_event_count=100,
        drift=_drift(feature_share=0.60, distribution_score=0.25, confidence_change=-0.2),
        quality=_quality(available=True, macro_f1=0.78),
        reference_macro_f1=0.8,
        settings=monitoring_config.alerts,
    )

    assert result.status == "critical"
    assert sum(signal.severity == "critical" for signal in result.signals) == 2
    assert "low_confidence_rate_increase" not in {signal.name for signal in result.signals}


def test_one_signal_does_not_raise_composite_alert(monitoring_config: MonitoringConfig) -> None:
    result = evaluate_alert_policy(
        event_count=100,
        minimum_event_count=100,
        drift=_drift(feature_share=0.0, distribution_score=0.0, confidence_change=0.11),
        quality=_quality(available=True, macro_f1=0.8),
        reference_macro_f1=0.8,
        settings=monitoring_config.alerts,
    )

    assert result.status == "healthy"
    assert [signal.name for signal in result.signals] == ["low_confidence_rate_increase"]
    assert result.to_dict()["policy"] == (
        "at least two independent signals are required for warning or critical"
    )


def _drift(
    *,
    feature_share: float,
    distribution_score: float,
    confidence_change: float,
) -> DriftResult:
    return DriftResult(
        columns=(
            ColumnDrift(
                column="predicted_queue",
                score=distribution_score,
                threshold=0.10,
                drifted=distribution_score >= 0.10,
                method="jensenshannon",
            ),
        ),
        drifted_input_feature_count=1 if feature_share else 0,
        input_feature_count=10,
        drifted_input_feature_share=feature_share,
        new_predicted_classes=(),
        missing_predicted_classes=(),
        reference_low_confidence_rate=0.2,
        current_low_confidence_rate=0.2 + confidence_change,
        low_confidence_rate_change=confidence_change,
        reference_mean_combined_length=100.0,
        current_mean_combined_length=100.0,
        combined_length_relative_change=0.0,
    )


def _quality(*, available: bool, macro_f1: float | None) -> DelayedQualityResult:
    return DelayedQualityResult(
        available=available,
        sample_count=50 if available else 0,
        minimum_sample_count=50,
        macro_f1=macro_f1,
        weighted_f1=macro_f1,
        per_class_recall={},
        correction_rate=None,
        acceptance_rate=None,
        mean_correct_confidence=None,
        mean_incorrect_confidence=None,
        quality_by_model_version={},
    )
