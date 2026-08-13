"""Regression tests for the repository-local pytest environment."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_pytest_uses_project_local_base_temp(pytestconfig: pytest.Config) -> None:
    configured_base_temp = Path(str(pytestconfig.option.basetemp)).resolve()
    expected_base_temp = (Path.cwd() / ".pytest-run").resolve()

    assert configured_base_temp == expected_base_temp
    assert configured_base_temp.parent == Path.cwd().resolve()
    assert pytestconfig.getini("cache_dir") == "tmp/pytest-cache-uv"
