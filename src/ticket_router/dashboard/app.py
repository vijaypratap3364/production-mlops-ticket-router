"""Streamlit entry point for the local ticket-routing demonstration."""

from __future__ import annotations

import streamlit as st

from ticket_router.dashboard.pages import (
    batch_routing,
    model_information,
    monitoring,
    route_ticket,
    system_status,
)

st.set_page_config(
    page_title="Ticket Routing Control Room",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --router-accent: #0f766e; --router-ink: #102a43; }
      .stApp { background: linear-gradient(180deg, #f5fbfa 0%, #ffffff 18rem); }
      [data-testid="stSidebar"] { border-right: 1px solid #d8e8e5; }
      [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid #d8e8e5;
        border-radius: 0.75rem;
        padding: 0.8rem 1rem;
      }
      h1, h2, h3 { color: var(--router-ink); letter-spacing: -0.02em; }
      .stButton > button, .stDownloadButton > button { border-radius: 0.6rem; }
      .stButton > button[kind="primary"] { background: var(--router-accent); }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Ticket Router")
    st.caption("Local, privacy-aware MLOps demonstration")

navigation = st.navigation(
    [
        st.Page(
            route_ticket.render,
            title="Route a ticket",
            icon="🎫",
            url_path="route-ticket",
            default=True,
        ),
        st.Page(
            batch_routing.render,
            title="Batch routing",
            icon="📄",
            url_path="batch-routing",
        ),
        st.Page(monitoring.render, title="Monitoring", icon="📈", url_path="monitoring"),
        st.Page(
            model_information.render,
            title="Model information",
            icon="🧠",
            url_path="model-information",
        ),
        st.Page(
            system_status.render,
            title="System status",
            icon="🩺",
            url_path="system-status",
        ),
    ]
)
navigation.run()
