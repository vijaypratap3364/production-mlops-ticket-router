"""Deterministic planted-drift simulation regression tests."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from ticket_router.monitoring.config import MonitoringConfig
from ticket_router.monitoring.simulate_drift import run_simulation, simulated_monitoring_frames


def test_planted_drift_simulation_writes_auditable_critical_reports(tmp_path: Path) -> None:
    summary = run_simulation(config=MonitoringConfig.load(), output_dir=tmp_path)
    drift = cast(dict[str, object], summary["drift"])

    assert summary["status"] == "critical"
    assert summary["reference_rows"] == 500
    assert summary["current_rows"] == 500
    assert cast(float, drift["drifted_input_feature_share"]) > 0.5
    columns = cast(list[dict[str, object]], drift["columns"])
    predicted_queue = next(column for column in columns if column["column"] == "predicted_queue")
    assert predicted_queue["drifted"] is True
    assert (tmp_path / "drift_report.html").is_file()
    assert (tmp_path / "drift_report.json").is_file()
    assert (tmp_path / "simulation_summary.json").is_file()


def test_planted_drift_simulation_rejects_an_unrepresentative_tiny_batch() -> None:
    with pytest.raises(ValueError, match="at least 100 rows"):
        simulated_monitoring_frames(rows=99)
