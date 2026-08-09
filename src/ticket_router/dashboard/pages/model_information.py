"""Champion model identity, lineage, metrics, and limitations page."""

from __future__ import annotations

import streamlit as st

from ticket_router.dashboard.api_client import DashboardAPIError
from ticket_router.dashboard.ui import api_client, format_bytes, format_optional, page_intro


def render() -> None:
    page_intro(
        "Model information",
        "Registry-backed champion identity and evaluation lineage exposed by FastAPI.",
    )
    try:
        with st.spinner("Loading champion metadata…"):
            model = api_client().model_metadata()
    except DashboardAPIError as exc:
        st.error(str(exc))
        return
    first, second, third, fourth = st.columns(4)
    first.metric("Champion", model.model_name)
    second.metric("Version", model.model_version)
    third.metric(
        "Macro F1", f"{model.macro_f1:.4f}" if model.macro_f1 is not None else "Not recorded"
    )
    fourth.metric("Model size", format_bytes(model.model_size_bytes))
    st.markdown("### Lineage")
    st.markdown(f"**Alias:** `{model.alias}`")
    st.markdown(f"**Created:** {format_optional(model.created_at)}")
    st.markdown(f"**Training-data hash:** `{format_optional(model.training_data_hash)}`")
    st.markdown("**Selected classes:**")
    st.write(", ".join(model.labels))
    st.markdown("### Purpose")
    st.write(model.model_card_summary)
    st.markdown("### Limitations")
    for limitation in model.limitations:
        st.markdown(f"- {limitation}")
    with st.expander("Input contract"):
        st.json(model.input_contract)
