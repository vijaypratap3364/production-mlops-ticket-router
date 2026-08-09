"""Shared Streamlit presentation helpers."""

from __future__ import annotations

from typing import Literal

import streamlit as st

from ticket_router.config import Settings
from ticket_router.dashboard.api_client import TicketRouterAPIClient

STATUS_ICON: dict[str, str] = {
    "healthy": "🟢",
    "ready": "🟢",
    "ok": "🟢",
    "warning": "🟠",
    "critical": "🔴",
    "unavailable": "🔴",
    "not_configured": "⚪",
    "required_unavailable": "🔴",
    "insufficient_data": "⚪",
    "memory": "🔵",
    "postgresql": "🟢",
}


@st.cache_resource
def dashboard_settings() -> Settings:
    return Settings.load()


@st.cache_resource
def api_client() -> TicketRouterAPIClient:
    settings = dashboard_settings()
    return TicketRouterAPIClient(
        base_url=settings.dashboard_api_url,
        timeout_seconds=settings.dashboard_request_timeout_seconds,
    )


def page_intro(title: str, description: str) -> None:
    st.title(title)
    st.caption(description)


def status_text(status: str) -> str:
    normalized = status.casefold()
    icon = STATUS_ICON.get(normalized, "⚪")
    return f"{icon} {status.replace('_', ' ').title()}"


def format_optional(value: object | None, *, fallback: str = "Not recorded") -> str:
    return fallback if value is None else str(value)


def format_bytes(value: int | None) -> str:
    if value is None:
        return "Not recorded"
    units = ("B", "KiB", "MiB", "GiB")
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:,.2f} {unit}"
        size /= 1024.0
    return f"{size:,.2f} GiB"


def render_notice(message: str, *, kind: Literal["info", "warning", "error"] = "info") -> None:
    getattr(st, kind)(message)
