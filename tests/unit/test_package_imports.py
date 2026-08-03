"""Smoke tests for package and architecture-boundary imports."""

from __future__ import annotations

from importlib import import_module

import pytest

PACKAGE_MODULES = (
    "ticket_router",
    "ticket_router.api",
    "ticket_router.config",
    "ticket_router.dashboard",
    "ticket_router.data",
    "ticket_router.db",
    "ticket_router.features",
    "ticket_router.logging_config",
    "ticket_router.maintenance",
    "ticket_router.modeling",
    "ticket_router.monitoring",
    "ticket_router.orchestration",
    "ticket_router.registry",
)


@pytest.mark.parametrize("module_name", PACKAGE_MODULES)
def test_package_module_imports(module_name: str) -> None:
    assert import_module(module_name) is not None
