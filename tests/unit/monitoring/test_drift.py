"""Evidently report and planted-drift tests."""

from pathlib import Path

from ticket_router.monitoring.config import MonitoringConfig
from ticket_router.monitoring.drift import generate_drift_report
from ticket_router.monitoring.simulate_drift import simulated_monitoring_frames


def test_identical_batch_has_no_drift(
    tmp_path: Path,
    monitoring_config: MonitoringConfig,
) -> None:
    reference, _ = simulated_monitoring_frames(rows=200)
    result = generate_drift_report(
        reference=reference,
        current=reference.clone(),
        settings=monitoring_config.drift,
        html_path=tmp_path / "no-drift.html",
        json_path=tmp_path / "no-drift.json",
    )

    assert result.drifted_input_feature_share == 0.0
    assert result.column("predicted_queue").drifted is False
    assert (tmp_path / "no-drift.html").is_file()
    assert (tmp_path / "no-drift.json").is_file()


def test_planted_drift_is_detected_and_reports_exist(
    tmp_path: Path,
    monitoring_config: MonitoringConfig,
) -> None:
    reference, current = simulated_monitoring_frames()
    result = generate_drift_report(
        reference=reference,
        current=current,
        settings=monitoring_config.drift,
        html_path=tmp_path / "drift.html",
        json_path=tmp_path / "drift.json",
    )

    assert result.drifted_input_feature_share >= 0.5
    assert result.column("predicted_queue").drifted is True
    assert result.column("prediction_confidence").drifted is True
    assert result.low_confidence_rate_change > 0.5
    assert (tmp_path / "drift.html").stat().st_size > 0
    assert (tmp_path / "drift.json").stat().st_size > 0
