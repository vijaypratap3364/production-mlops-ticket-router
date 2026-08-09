"""Privacy-safe operational read-model projection tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from ticket_router.api.operations import OperationsService
from ticket_router.db.contracts import MonitoringRun, RetrainingRun


class MonitoringRuns:
    def __init__(self, records: tuple[MonitoringRun, ...]) -> None:
        self.records = records

    def save(self, run: MonitoringRun) -> None:
        raise AssertionError("read-only test")

    def get(self, run_id: str) -> MonitoringRun | None:
        return next((run for run in self.records if run.run_id == run_id), None)

    def list_recent(self, *, limit: int) -> tuple[MonitoringRun, ...]:
        return self.records[:limit]


class RetrainingRuns:
    def __init__(self, records: tuple[RetrainingRun, ...]) -> None:
        self.records = records

    def save(self, run: RetrainingRun) -> None:
        raise AssertionError("read-only test")

    def get(self, run_id: str) -> RetrainingRun | None:
        return next((run for run in self.records if run.run_id == run_id), None)

    def list_recent(self, *, limit: int) -> tuple[RetrainingRun, ...]:
        return self.records[:limit]


def test_operational_records_are_projected_for_dashboard() -> None:
    now = datetime(2026, 8, 8, tzinfo=UTC)
    monitoring = MonitoringRun(
        run_id=str(uuid4()),
        started_at=now,
        completed_at=now + timedelta(minutes=1),
        reference_period_start=now - timedelta(days=14),
        reference_period_end=now - timedelta(days=7),
        current_period_start=now - timedelta(days=7),
        current_period_end=now,
        drift_status="warning",
        report_paths=("reports/drift.json", "reports/drift.html"),
        summary={
            "model_version": "7",
            "event_count": 120,
            "feedback_count": 25,
            "drift_without_labels": {"current_low_confidence_rate": 0.25},
            "performance_with_delayed_labels": {"available": True, "macro_f1": 0.7},
            "current_predicted_class_distribution": {"Billing": 80, "Technical": 40},
        },
    )
    retraining = RetrainingRun(
        run_id=str(uuid4()),
        trigger_reason="manual review",
        source_data_period_start=now - timedelta(days=30),
        source_data_period_end=now,
        status="candidate_registered",
        mlflow_run_id="run-1",
        candidate_model_version="8",
        gate_results={"passed": True},
        started_at=now,
        completed_at=now + timedelta(minutes=5),
    )
    service = OperationsService(
        monitoring_runs=MonitoringRuns((monitoring,)),
        retraining_runs=RetrainingRuns((retraining,)),
    )

    history = service.monitoring_history(limit=10)
    latest_retraining = service.latest_retraining_status()

    assert history.runs[0].predicted_class_distribution == {"Billing": 80, "Technical": 40}
    assert history.runs[0].report_path == "reports/drift.html"
    assert latest_retraining is not None
    assert latest_retraining.details["candidate_model_version"] == "8"
