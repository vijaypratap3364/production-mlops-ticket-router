"""Typed, versioned Stage 15 benchmark configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_BENCHMARK_CONFIG_PATH = Path("configs/benchmark.yaml")


class BenchmarkTargets(BaseModel):
    """Predeclared local guardrails; values are not measured results."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    maximum_cold_load_ms: float = Field(gt=0.0)
    maximum_single_request_p95_ms: float = Field(gt=0.0)
    maximum_batch_p95_ms_per_item: float = Field(gt=0.0)
    minimum_direct_throughput_per_second: float = Field(gt=0.0)
    maximum_process_rss_after_load_mib: float = Field(gt=0.0)
    maximum_serialized_model_size_mib: float = Field(gt=0.0)
    maximum_api_p95_ms: float = Field(gt=0.0)
    maximum_api_response_overhead_p50_ms: float = Field(gt=0.0)
    maximum_api_error_rate: float = Field(ge=0.0, le=1.0)
    maximum_load_test_p95_ms: float = Field(gt=0.0)
    minimum_load_test_throughput_per_second: float = Field(gt=0.0)
    maximum_load_test_failure_rate: float = Field(ge=0.0, le=1.0)


class LoadTestConfig(BaseModel):
    """Bounded Locust defaults and hard safety caps."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_users: int = Field(gt=0)
    default_spawn_rate: float = Field(gt=0.0)
    default_duration_seconds: int = Field(gt=0)
    maximum_users: int = Field(gt=0, le=100)
    maximum_spawn_rate: float = Field(gt=0.0, le=50.0)
    maximum_duration_seconds: int = Field(gt=0, le=3600)
    allow_remote_host: bool = False
    output_prefix: Path
    html_report_path: Path

    @model_validator(mode="after")
    def defaults_must_fit_caps(self) -> Self:
        if self.default_users > self.maximum_users:
            raise ValueError("default_users exceeds maximum_users")
        if self.default_spawn_rate > self.maximum_spawn_rate:
            raise ValueError("default_spawn_rate exceeds maximum_spawn_rate")
        if self.default_duration_seconds > self.maximum_duration_seconds:
            raise ValueError("default_duration_seconds exceeds maximum_duration_seconds")
        return self


class BenchmarkConfig(BaseModel):
    """Complete reproducible inference and load-test protocol."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    random_seed: int = Field(ge=0)
    warmup_iterations: int = Field(ge=1, le=100)
    single_request_iterations: int = Field(ge=10, le=10_000)
    batch_sizes: tuple[int, ...] = Field(min_length=1)
    batch_repetitions: int = Field(ge=3, le=1_000)
    api_iterations: int = Field(ge=10, le=10_000)
    api_url: str
    api_timeout_seconds: float = Field(gt=0.0, le=60.0)
    split_manifest_path: Path
    output_json_path: Path
    output_markdown_path: Path
    locust_stats_path: Path
    reliability_results_path: Path
    load_test: LoadTestConfig
    targets: BenchmarkTargets

    @field_validator("batch_sizes")
    @classmethod
    def batch_sizes_must_be_unique_and_bounded(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(size <= 1 or size > 1000 for size in value):
            raise ValueError("batch_sizes must contain values from 2 through 1000")
        if len(set(value)) != len(value):
            raise ValueError("batch_sizes must be unique")
        return tuple(sorted(value))

    @field_validator("api_url")
    @classmethod
    def api_url_must_be_http(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("api_url must be an absolute HTTP(S) URL")
        return value.rstrip("/")

    @classmethod
    def load(cls, path: Path = DEFAULT_BENCHMARK_CONFIG_PATH) -> BenchmarkConfig:
        with path.open(encoding="utf-8") as config_file:
            raw: Any = yaml.safe_load(config_file)
        if not isinstance(raw, dict):
            raise ValueError(f"benchmark configuration root must be a mapping: {path}")
        return cls.model_validate(raw)
