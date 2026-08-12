"""Typed benchmark configuration and safety-cap tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ticket_router.benchmarking.config import BenchmarkConfig


def test_versioned_benchmark_configuration_is_bounded() -> None:
    config = BenchmarkConfig.load()

    assert config.random_seed == 42
    assert config.batch_sizes == (8, 32)
    assert config.load_test.default_users <= config.load_test.maximum_users
    assert config.load_test.default_duration_seconds <= config.load_test.maximum_duration_seconds
    assert config.load_test.allow_remote_host is False


def test_benchmark_configuration_rejects_duplicate_batch_sizes(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/benchmark.yaml").read_text(encoding="utf-8"))
    raw["batch_sizes"] = [8, 8]
    path = tmp_path / "benchmark.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="batch_sizes must be unique"):
        BenchmarkConfig.load(path)
