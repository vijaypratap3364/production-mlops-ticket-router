"""Validated CSV batch-routing page."""

from __future__ import annotations

import streamlit as st

from ticket_router.dashboard.api_client import DashboardAPIError
from ticket_router.dashboard.ui import api_client, dashboard_settings, page_intro
from ticket_router.dashboard.validation import (
    DashboardValidationError,
    batch_results_frame,
    parse_ticket_csv,
)


def render() -> None:
    page_intro(
        "Batch routing",
        "Upload a UTF-8 CSV containing subject and body columns. Only those approved fields are "
        "sent to the inference API.",
    )
    settings = dashboard_settings().api_settings
    uploaded = st.file_uploader("Ticket CSV", type=("csv",), accept_multiple_files=False)
    if uploaded is None:
        st.info("Choose a CSV to validate and preview it before routing.")
        return
    try:
        batch = parse_ticket_csv(
            uploaded.getvalue(),
            maximum_batch_size=settings.maximum_batch_size,
            maximum_subject_characters=settings.maximum_subject_characters,
            maximum_body_characters=settings.maximum_body_characters,
        )
    except DashboardValidationError as exc:
        st.error(str(exc))
        return
    if batch.ignored_columns:
        st.warning(
            "These extra columns will be ignored and never sent to the API: "
            + ", ".join(batch.ignored_columns)
        )
    st.success(f"Validated {len(batch.items)} ticket rows.")
    st.caption("Preview is visible only in this browser session; ticket text is not logged.")
    st.dataframe(batch.frame.head(20), hide_index=True, use_container_width=True)
    if not st.button("Route validated batch", type="primary"):
        return
    try:
        with st.spinner(f"Routing {len(batch.items)} tickets through FastAPI…"):
            result = api_client().predict_batch(batch.items)
    except DashboardAPIError as exc:
        st.error(str(exc))
        return
    frame = batch_results_frame(result.response)
    st.subheader("Batch results")
    left, right = st.columns(2)
    left.metric("Predictions", len(frame.index))
    right.metric("API round trip", f"{result.api_latency_ms:,.1f} ms")
    st.dataframe(frame, hide_index=True, use_container_width=True)
    st.download_button(
        "Download routing results",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name="ticket-routing-results.csv",
        mime="text/csv",
    )
