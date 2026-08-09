"""Read-only operational projections for API and dashboard presentation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ticket_router.api.schemas import (
    MonitoringHistoryResponse,
    MonitoringRunResponse,
    OperationalRunResponse,
)
from ticket_router.db.contracts import (
    MonitoringRun,
    MonitoringRunRepository,
    RetrainingRun,
    RetrainingRunRepository,
)

MODEL_CARD_SUMMARY = (
    "A local, CPU-oriented sparse-text classifier that routes English customer-support "
    "tickets to one of the registered support queues."
)
MODEL_LIMITATIONS = (
    "The training data is synthetic and template-heavy, so offline results may not transfer to "
    "organization-specific tickets.",
    "The model is intended for English tickets and must not be used for safety-critical or fully "
    "autonomous decisions.",
    "Minority queues have weaker recall and low-confidence predictions require human review.",
)


class OperationsService:
    """Project persistence records into privacy-safe read models."""

    def __init__(
        self,
        *,
        monitoring_runs: MonitoringRunRepository | None,
        retraining_runs: RetrainingRunRepository | None,
    ) -> None:
        self._monitoring_runs = monitoring_runs
        self._retraining_runs = retraining_runs

    def monitoring_history(self, *, limit: int) -> MonitoringHistoryResponse:
        if limit <= 0 or limit > 100:
            raise ValueError("monitoring history limit must be between 1 and 100")
        records = self._monitoring_runs.list_recent(limit=limit) if self._monitoring_runs else ()
        return MonitoringHistoryResponse(runs=[_monitoring_response(record) for record in records])

    def latest_monitoring_status(self) -> OperationalRunResponse | None:
        records = self._monitoring_runs.list_recent(limit=1) if self._monitoring_runs else ()
        return _monitoring_status(records[0]) if records else None

    def latest_retraining_status(self) -> OperationalRunResponse | None:
        records = self._retraining_runs.list_recent(limit=1) if self._retraining_runs else ()
        return _retraining_status(records[0]) if records else None


def _monitoring_response(record: MonitoringRun) -> MonitoringRunResponse:
    summary = record.summary
    return MonitoringRunResponse(
        run_id=record.run_id,
        status=record.drift_status,
        completed_at=record.completed_at,
        model_version=_optional_string(summary.get("model_version")),
        event_count=_nonnegative_int(summary.get("event_count")),
        feedback_count=_nonnegative_int(summary.get("feedback_count")),
        drift_without_labels=_optional_mapping(summary.get("drift_without_labels")),
        performance_with_delayed_labels=_optional_mapping(
            summary.get("performance_with_delayed_labels")
        ),
        predicted_class_distribution=_integer_mapping(
            summary.get("current_predicted_class_distribution")
        ),
        report_path=_preferred_report_path(record.report_paths),
    )


def _monitoring_status(record: MonitoringRun) -> OperationalRunResponse:
    return OperationalRunResponse(
        run_id=record.run_id,
        status=record.drift_status,
        started_at=record.started_at,
        completed_at=record.completed_at,
        details={
            "event_count": _nonnegative_int(record.summary.get("event_count")),
            "feedback_count": _nonnegative_int(record.summary.get("feedback_count")),
            "model_version": _optional_string(record.summary.get("model_version")),
        },
    )


def _retraining_status(record: RetrainingRun) -> OperationalRunResponse:
    return OperationalRunResponse(
        run_id=record.run_id,
        status=record.status,
        started_at=record.started_at,
        completed_at=record.completed_at,
        details={
            "trigger_reason": record.trigger_reason,
            "candidate_model_version": record.candidate_model_version,
            "mlflow_run_id": record.mlflow_run_id,
            "promotion_gates_passed": record.gate_results.get("passed"),
        },
    )


def _preferred_report_path(paths: tuple[str, ...]) -> str | None:
    return next(
        (path for path in paths if path.casefold().endswith(".html")), paths[0] if paths else None
    )


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if not isinstance(value, (str, int, float)):
        return 0
    try:
        result = int(value)
    except (TypeError, ValueError):
        return 0
    return max(result, 0)


def _optional_mapping(value: object) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _integer_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int] = {}
    for key, raw_count in value.items():
        count = _nonnegative_int(raw_count)
        result[str(key)] = count
    return result
