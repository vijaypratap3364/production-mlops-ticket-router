"""Multi-signal alert policy tests."""

from ticket_router.monitoring.config import MonitoringConfig
from ticket_router.monitoring.policy import evaluate_alert_policy
from ticket_router.monitoring.quality import calculate_delayed_quality


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
