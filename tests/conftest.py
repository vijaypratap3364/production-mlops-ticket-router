"""Repository-wide pytest classification hooks."""

from __future__ import annotations

from pathlib import Path

import pytest

TEST_ROOT = Path(__file__).parent


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Classify infrastructure and live-stack tests from their directory boundary."""
    del config
    for item in items:
        relative_parts = item.path.relative_to(TEST_ROOT).parts
        if relative_parts[0] == "integration":
            item.add_marker(pytest.mark.integration)
        elif relative_parts[0] == "e2e":
            item.add_marker(pytest.mark.e2e)
