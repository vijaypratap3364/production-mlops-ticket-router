"""Multi-signal alert policy for drift and delayed-label quality."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from ticket_router.monitoring.config import AlertSettings
from ticket_router.monitoring.drift import DriftResult
from ticket_router.monitoring.quality import DelayedQualityResult

MonitoringStatus = Literal["healthy", "warning", "critical", "insufficient_data"]
SignalSeverity = Literal["warning", "critical"]


@dataclass(frozen=True)
class AlertSignal:
    name: str
    severity: SignalSeverity
    observed: float
    threshold: float


@dataclass(frozen=True)
class AlertDecision:
    status: MonitoringStatus
    signals: tuple[AlertSignal, ...]
    event_count: int
    minimum_event_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "signals": [asdict(signal) for signal in self.signals],
            "event_count": self.event_count,
            "minimum_event_count": self.minimum_event_count,
            "policy": "at least two independent signals are required for warning or critical",
        }


def evaluate_alert_policy(
    *,
    event_count: int,
    minimum_event_count: int,
    drift: DriftResult | None,
    quality: DelayedQualityResult,
    reference_macro_f1: float,
    settings: AlertSettings,
) -> AlertDecision:
    if event_count < minimum_event_count or drift is None:
        return AlertDecision(
            status="insufficient_data",
            signals=(),
            event_count=event_count,
            minimum_event_count=minimum_event_count,
        )
    signals = [
        *_threshold_signal(
            "drifted_input_feature_share",
            drift.drifted_input_feature_share,
            settings.warning_drifted_feature_share,
            settings.critical_drifted_feature_share,
        ),
        *_threshold_signal(
            "prediction_distribution_score",
            drift.column("predicted_queue").score,
            settings.warning_prediction_distribution_score,
            settings.critical_prediction_distribution_score,
        ),
        *_threshold_signal(
            "low_confidence_rate_increase",
            max(drift.low_confidence_rate_change, 0.0),
            settings.warning_low_confidence_rate_increase,
            settings.critical_low_confidence_rate_increase,
        ),
    ]
    if quality.available and quality.macro_f1 is not None:
        signals.extend(
            _threshold_signal(
                "labeled_macro_f1_decline",
                max(reference_macro_f1 - quality.macro_f1, 0.0),
                settings.warning_macro_f1_decline,
                settings.critical_macro_f1_decline,
            )
        )
    critical_count = sum(signal.severity == "critical" for signal in signals)
    if critical_count >= settings.minimum_critical_signals:
        status: MonitoringStatus = "critical"
    elif len(signals) >= settings.minimum_warning_signals:
        status = "warning"
    else:
        status = "healthy"
    return AlertDecision(
        status=status,
        signals=tuple(signals),
        event_count=event_count,
        minimum_event_count=minimum_event_count,
    )


def _threshold_signal(
    name: str,
    observed: float,
    warning: float,
    critical: float,
) -> tuple[AlertSignal, ...]:
    if observed >= critical:
        return (AlertSignal(name, "critical", observed, critical),)
    if observed >= warning:
        return (AlertSignal(name, "warning", observed, warning),)
    return ()
