"""Typed, versioned Prefect orchestration configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_ORCHESTRATION_CONFIG_PATH = Path("configs/orchestration.yaml")


class RuntimeSettings(BaseModel):
    """Retry and local worker configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    temporary_failure_retries: int = Field(ge=0, le=10)
    retry_delay_seconds: int = Field(ge=0, le=3600)
    work_pool_name: str = Field(min_length=1)
    work_queue_name: str = Field(min_length=1)


class OrchestrationPaths(BaseModel):
    """All flow inputs and generated-output roots."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_config: Path
    experiment_config: Path
    final_model_config: Path
    monitoring_config: Path
    raw_manifest: Path
    normalized_data: Path
    normalization_manifest: Path
    selected_classes: Path
    processed_directory: Path
    reference_directory: Path
    reports_directory: Path
    model_artifacts_directory: Path
    leaderboard: Path
    monitoring_output_directory: Path
    approved_feedback_input: Path
    retraining_output_directory: Path
    orchestration_output_directory: Path


class RetrainingSettings(BaseModel):
    """Multi-signal controls that prevent reflexive retraining."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_new_feedback_labels: int = Field(gt=0)
    required_consecutive_critical_windows: int = Field(ge=2)
    macro_f1_decline_tolerance: float = Field(gt=0.0, le=1.0)
    sustained_low_confidence_rate_increase: float = Field(gt=0.0, le=1.0)


class ScheduleSettings(BaseModel):
    """Disabled-by-default local schedule definitions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    timezone: str = Field(min_length=1)
    monitoring_cron: str = Field(min_length=1)
    retraining_evaluation_cron: str = Field(min_length=1)


class OrchestrationConfig(BaseModel):
    """Complete Stage 11 workflow configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime: RuntimeSettings
    paths: OrchestrationPaths
    retraining: RetrainingSettings
    schedules: ScheduleSettings

    @model_validator(mode="after")
    def reject_invalid_approved_input(self) -> Self:
        if self.paths.approved_feedback_input.suffix != ".parquet":
            raise ValueError("approved_feedback_input must be a Parquet file")
        return self

    @classmethod
    def load(cls, path: str | Path = DEFAULT_ORCHESTRATION_CONFIG_PATH) -> Self:
        config_path = Path(path)
        with config_path.open(encoding="utf-8") as config_file:
            raw: Any = yaml.safe_load(config_file)
        if not isinstance(raw, dict):
            raise ValueError(f"orchestration configuration must be a mapping: {config_path}")
        return cls.model_validate(raw)


def orchestration_configuration_hash(config: OrchestrationConfig) -> str:
    """Return a stable hash for workflow configuration lineage."""
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
