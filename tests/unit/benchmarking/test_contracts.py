"""Latency summary and machine-readable benchmark contract tests."""

from __future__ import annotations

import pytest

from ticket_router.benchmarking.contracts import summarize_latencies


def test_latency_summary_preserves_percentiles_and_batch_throughput() -> None:
    result = summarize_latencies(
        [0.001, 0.002, 0.003, 0.004, 0.005],
        operations_per_observation=8,
    )

    assert result.observation_count == 5
    assert result.operation_count == 40
    assert result.p50_ms == pytest.approx(3.0)
    assert result.p95_ms == pytest.approx(4.8)
    assert result.p99_ms == pytest.approx(4.96)
    assert result.throughput_per_second == pytest.approx(40 / 0.015)


@pytest.mark.parametrize(
    ("observations", "message"),
    [
        ([], "at least one"),
        ([-0.001], "cannot be negative"),
        ([0.0], "must be positive"),
    ],
)
def test_latency_summary_rejects_invalid_observations(
    observations: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        summarize_latencies(observations)
