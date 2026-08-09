"""Operational dependency status page."""

from __future__ import annotations

import streamlit as st

from ticket_router.api.schemas import OperationalRunResponse
from ticket_router.dashboard.api_client import DashboardAPIError
from ticket_router.dashboard.ui import api_client, page_intro, status_text


def render() -> None:
    page_intro(
        "System status",
        "Live process, model, database, monitoring, and retraining readiness from FastAPI.",
    )
    try:
        with st.spinner("Checking local services…"):
            health = api_client().health()
            readiness = api_client().readiness()
            status = api_client().system_status()
    except DashboardAPIError as exc:
        st.error(str(exc))
        return
    first, second, third, fourth = st.columns(4)
    first.metric("API health", status_text(health.status))
    second.metric("API readiness", status_text("ready" if readiness.ready else "unavailable"))
    third.metric("Database", status_text(status.database_status))
    fourth.metric(
        "Champion model",
        status_text("ready" if status.mlflow_model_available else "unavailable"),
    )
    st.caption(
        f"Database mode: {status.database_mode.replace('_', ' ')} · "
        f"model ready: {readiness.model_ready} · database ready: {readiness.database_ready}"
    )
    left, right = st.columns(2)
    with left:
        _render_run("Latest monitoring run", status.latest_monitoring_run)
    with right:
        _render_run("Latest retraining run", status.latest_retraining_run)


def _render_run(title: str, run: OperationalRunResponse | None) -> None:
    st.subheader(title)
    if run is None:
        st.info("No run has been recorded in the configured database.")
        return
    st.markdown(status_text(run.status))
    st.caption(
        f"Run `{run.run_id}` · started {run.started_at.isoformat()} · "
        f"completed {run.completed_at.isoformat() if run.completed_at else 'in progress'}"
    )
    st.json(run.details)
