"""Application service for one bounded batch-monitoring run."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import polars as pl

from ticket_router.data.manifests import atomic_write_json
from ticket_router.db.contracts import MonitoringRun, MonitoringRunRepository
from ticket_router.monitoring.config import MonitoringConfig
from ticket_router.monitoring.contracts import MonitoringDataRepository
from ticket_router.monitoring.drift import DriftResult, generate_drift_report
from ticket_router.monitoring.policy import AlertDecision, evaluate_alert_policy
from ticket_router.monitoring.quality import (
    DelayedQualityResult,
    calculate_delayed_quality,
)
from ticket_router.monitoring.reference import MonitoringReferenceManifest


@dataclass(frozen=True)
class MonitoringExecution:
    run_id: str
    summary_path: Path
    html_report_path: Path | None
    json_report_path: Path | None
    decision: AlertDecision
    drift: DriftResult | None
    quality: DelayedQualityResult


def execute_monitoring_run(
    *,
    reference_path: Path,
    reference_manifest_path: Path,
    data_repository: MonitoringDataRepository,
    run_repository: MonitoringRunRepository | None,
    config: MonitoringConfig,
    start: datetime,
    end: datetime,
    output_root: Path,
    model_version: str | None = None,
    minimum_event_count: int | None = None,
    run_id: str | None = None,
) -> MonitoringExecution:
    """Load one time window, separate drift from labels, report, and persist summary."""
    started_at = datetime.now(UTC)
    if start.tzinfo is None or end.tzinfo is None or end <= start:
        raise ValueError("monitoring start/end must be timezone-aware and ordered")
    reference_manifest = MonitoringReferenceManifest.read(reference_manifest_path)
    if reference_manifest.reference_data_sha256 != _hash(reference_path):
        raise ValueError("monitoring reference hash does not match its manifest")
    reference = pl.read_parquet(reference_path)
    expected_version = model_version or reference_manifest.champion_model_version
    current_events = data_repository.load_predictions(
        start=start,
        end=end,
        model_version=expected_version,
    )
    labeled_events = data_repository.load_labeled_predictions(
        start=start,
        end=end,
        model_version=expected_version,
    )
    minimum = minimum_event_count or config.current_window.minimum_event_count
    quality = calculate_delayed_quality(
        labeled_events,
        minimum_sample_count=config.quality.minimum_feedback_count,
    )
    reference_quality = reference_manifest.champion_baseline_macro_f1
    identifier = run_id or str(uuid4())
    report_directory = output_root / identifier
    report_directory.mkdir(parents=True, exist_ok=True)
    html_path: Path | None = None
    json_path: Path | None = None
    drift: DriftResult | None = None
    if len(current_events) >= minimum:
        current = pl.DataFrame([event.monitoring_values() for event in current_events])
        html_path = report_directory / "drift_report.html"
        json_path = report_directory / "drift_report.json"
        drift = generate_drift_report(
            reference=reference,
            current=current,
            settings=config.drift,
            html_path=html_path,
            json_path=json_path,
        )
    decision = evaluate_alert_policy(
        event_count=len(current_events),
        minimum_event_count=minimum,
        drift=drift,
        quality=quality,
        reference_macro_f1=reference_quality,
        settings=config.alerts,
    )
    completed_at = datetime.now(UTC)
    summary = {
        "run_id": identifier,
        "status": decision.status,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "model_version": expected_version,
        "reference_data_sha256": reference_manifest.reference_data_sha256,
        "event_count": len(current_events),
        "feedback_count": len(labeled_events),
        "current_predicted_class_distribution": dict(
            sorted(Counter(event.predicted_queue for event in current_events).items())
        ),
        "drift_without_labels": drift.to_dict() if drift is not None else None,
        "performance_with_delayed_labels": quality.to_dict(),
        "reference_macro_f1": reference_quality,
        "alert_decision": decision.to_dict(),
        "completed_at": completed_at.isoformat(),
    }
    summary_path = report_directory / "monitoring_summary.json"
    atomic_write_json(summary_path, summary)
    if run_repository is not None:
        created_at = _parse_utc(reference_manifest.creation_timestamp_utc)
        report_paths = tuple(
            path.as_posix() for path in (html_path, json_path, summary_path) if path is not None
        )
        run_repository.save(
            MonitoringRun(
                run_id=identifier,
                started_at=started_at,
                completed_at=completed_at,
                reference_period_start=created_at,
                reference_period_end=created_at,
                current_period_start=start,
                current_period_end=end,
                drift_status=decision.status,
                report_paths=report_paths,
                summary=summary,
            )
        )
    return MonitoringExecution(
        run_id=identifier,
        summary_path=summary_path,
        html_report_path=html_path,
        json_report_path=json_path,
        decision=decision,
        drift=drift,
        quality=quality,
    )


def _hash(path: Path) -> str:
    from ticket_router.hashing import sha256_file

    return sha256_file(path)


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
