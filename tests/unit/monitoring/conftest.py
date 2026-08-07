"""Shared Stage 10 monitoring configuration fixture."""

import pytest

from ticket_router.monitoring.config import MonitoringConfig


@pytest.fixture
def monitoring_config() -> MonitoringConfig:
    return MonitoringConfig.load()
