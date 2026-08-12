"""Machine-readable benchmark result contracts and latency calculations."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LatencySummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    observation_count: int = Field(gt=0)
    operation_count: int = Field(gt=0)
    minimum_ms: float = Field(ge=0.0)
    mean_ms: float = Field(ge=0.0)
    p50_ms: float = Field(ge=0.0)
    p95_ms: float = Field(ge=0.0)
    p99_ms: float = Field(ge=0.0)
    maximum_ms: float = Field(ge=0.0)
    throughput_per_second: float = Field(gt=0.0)


class BatchBenchmark(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: int = Field(gt=1)
    batch_latency: LatencySummary
    p95_ms_per_item: float = Field(ge=0.0)


class LoadTestSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    failure_rate: float = Field(ge=0.0, le=1.0)
    p50_ms: float = Field(ge=0.0)
    p95_ms: float = Field(ge=0.0)
    p99_ms: float = Field(ge=0.0)
    throughput_per_second: float = Field(ge=0.0)
    source_path: str


class TargetEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str
    target: float
    measured: float
    comparison: str
    unit: str
    passed: bool


class ModelContractResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    checks: dict[str, bool]
    labels: tuple[str, ...]
    label_mapping: dict[str, int]
    probability_dimensions: tuple[int, int]
    top_k_size: int
    deterministic_fixed_sample: bool
    serialized_roundtrip_size_bytes: int = Field(gt=0)


class ReliabilityScenario(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    passed: bool
    expected: str
    observed: str


def summarize_latencies(
    elapsed_seconds: list[float],
    *,
    operations_per_observation: int = 1,
) -> LatencySummary:
    """Summarize raw observations without rounding away measured precision."""
    if not elapsed_seconds:
        raise ValueError("at least one latency observation is required")
    if operations_per_observation <= 0:
        raise ValueError("operations_per_observation must be positive")
    ordered_ms = sorted(value * 1000.0 for value in elapsed_seconds)
    if ordered_ms[0] < 0.0:
        raise ValueError("latency observations cannot be negative")
    total_seconds = sum(elapsed_seconds)
    if total_seconds <= 0.0:
        raise ValueError("total measured duration must be positive")
    observations = len(ordered_ms)
    operations = observations * operations_per_observation
    return LatencySummary(
        observation_count=observations,
        operation_count=operations,
        minimum_ms=ordered_ms[0],
        mean_ms=sum(ordered_ms) / observations,
        p50_ms=_percentile(ordered_ms, 0.50),
        p95_ms=_percentile(ordered_ms, 0.95),
        p99_ms=_percentile(ordered_ms, 0.99),
        maximum_ms=ordered_ms[-1],
        throughput_per_second=operations / total_seconds,
    )


def _percentile(ordered_values: list[float], quantile: float) -> float:
    position = (len(ordered_values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered_values) - 1)
    fraction = position - lower
    return ordered_values[lower] + (ordered_values[upper] - ordered_values[lower]) * fraction
