"""Read-only monitoring overview rendered from FastAPI responses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import streamlit as st

from ticket_router.api.schemas import MonitoringRunResponse
from ticket_router.dashboard.api_client import DashboardAPIError
from ticket_router.dashboard.ui import api_client, page_intro, status_text


def render() -> None:
    page_intro(
        "Monitoring",
        "Privacy-safe drift and delayed-label quality from completed batch-monitoring runs.",
    )
    try:
        with st.spinner("Loading monitoring history…"):
            history = api_client().monitoring_history(limit=30)
    except DashboardAPIError as exc:
        st.error(str(exc))
        return
    if not history.runs:
        st.info(
            "No monitoring runs are available yet. Run the monitoring flow after PostgreSQL has "
            "enough prediction events."
        )
        return
    latest = history.runs[0]
    _render_latest(latest)
    _render_low_confidence_trend(history.runs)
    _render_distribution(latest)
    _render_quality(latest)
    _render_drift_details(latest)


def _render_latest(run: MonitoringRunResponse) -> None:
    drift = run.drift_without_labels or {}
    share = _float_value(drift.get("drifted_input_feature_share"))
    columns = st.columns(4)
    columns[0].metric("Latest status", status_text(run.status))
    columns[1].metric("Prediction events", f"{run.event_count:,}")
    columns[2].metric("Feedback labels", f"{run.feedback_count:,}")
    columns[3].metric(
        "Drifted input features",
        f"{share:.1%}" if share is not None else "Not available",
    )
    st.caption(
        f"Run `{run.run_id}` · model version {run.model_version or 'not recorded'} · "
        f"completed {run.completed_at.isoformat() if run.completed_at else 'not recorded'}"
    )
    if run.report_path:
        st.markdown("**Detailed report path**")
        st.code(run.report_path, language=None)


def _render_low_confidence_trend(runs: list[MonitoringRunResponse]) -> None:
    rows: list[dict[str, object]] = []
    for run in reversed(runs):
        drift = run.drift_without_labels or {}
        rate = _float_value(drift.get("current_low_confidence_rate"))
        if run.completed_at is not None and rate is not None:
            rows.append({"completed_at": run.completed_at, "low_confidence_rate": rate})
    st.subheader("Low-confidence trend")
    if not rows:
        st.info("A low-confidence trend will appear after drift-capable monitoring runs complete.")
        return
    frame = pd.DataFrame(rows).set_index("completed_at")
    st.line_chart(frame, y="low_confidence_rate")


def _render_distribution(run: MonitoringRunResponse) -> None:
    st.subheader("Predicted-class distribution")
    if not run.predicted_class_distribution:
        st.info("The latest stored monitoring summary does not contain class counts.")
        return
    frame = pd.DataFrame(
        [
            {"Queue": queue, "Predictions": count}
            for queue, count in sorted(
                run.predicted_class_distribution.items(), key=lambda item: (-item[1], item[0])
            )
        ]
    ).set_index("Queue")
    st.bar_chart(frame, y="Predictions", horizontal=True)


def _render_quality(run: MonitoringRunResponse) -> None:
    st.subheader("Delayed-label quality")
    quality = run.performance_with_delayed_labels
    if not quality or not bool(quality.get("available")):
        sample_count = int(quality.get("sample_count", 0)) if quality else 0
        minimum = int(quality.get("minimum_sample_count", 0)) if quality else 0
        st.info(
            f"Quality is not yet available: {sample_count} feedback labels observed; "
            f"{minimum} required by policy."
        )
        return
    macro_f1 = _float_value(quality.get("macro_f1"))
    weighted_f1 = _float_value(quality.get("weighted_f1"))
    correction_rate = _float_value(quality.get("correction_rate"))
    metrics = st.columns(3)
    metrics[0].metric("Macro F1", f"{macro_f1:.3f}" if macro_f1 is not None else "Not recorded")
    metrics[1].metric(
        "Weighted F1",
        f"{weighted_f1:.3f}" if weighted_f1 is not None else "Not recorded",
    )
    metrics[2].metric(
        "Correction rate",
        f"{correction_rate:.1%}" if correction_rate is not None else "Not recorded",
    )
    by_version = quality.get("quality_by_model_version")
    if isinstance(by_version, Mapping) and by_version:
        rows = [
            {"Model version": str(version), **dict(values)}
            for version, values in by_version.items()
            if isinstance(values, Mapping)
        ]
        if rows:
            st.markdown("**Model-version comparison**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_drift_details(run: MonitoringRunResponse) -> None:
    drift = run.drift_without_labels
    if not drift:
        return
    columns = drift.get("columns")
    if isinstance(columns, list) and columns:
        st.subheader("Feature drift details")
        st.dataframe(pd.DataFrame(columns), hide_index=True, use_container_width=True)


def _float_value(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
