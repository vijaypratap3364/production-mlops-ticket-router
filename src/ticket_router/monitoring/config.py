"""Typed configuration for privacy-safe batch monitoring."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_MONITORING_CONFIG_PATH = Path("configs/monitoring.yaml")


class ReferenceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    output_path: Path
    manifest_path: Path
    source_splits: tuple[Literal["train", "validation"], ...]
    inference_batch_size: int = Field(gt=0)

    @model_validator(mode="after")
    def source_splits_must_be_complete_and_unique(self) -> Self:
        if self.source_splits != ("train", "validation"):
            raise ValueError("monitoring reference must use train and validation exactly once")
        return self


class CurrentWindowSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    default_lookback_days: int = Field(gt=0)
    minimum_event_count: int = Field(gt=0)


class DriftSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    numeric_method: Literal["wasserstein"]
    categorical_method: Literal["jensenshannon"]
    numeric_feature_threshold: float = Field(gt=0.0, le=1.0)
    prediction_distribution_threshold: float = Field(gt=0.0, le=1.0)
    confidence_distribution_threshold: float = Field(gt=0.0, le=1.0)
    low_confidence_distribution_threshold: float = Field(gt=0.0, le=1.0)


class QualitySettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_feedback_count: int = Field(gt=0)


class AlertSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    warning_drifted_feature_share: float = Field(ge=0.0, le=1.0)
    critical_drifted_feature_share: float = Field(ge=0.0, le=1.0)
    warning_prediction_distribution_score: float = Field(ge=0.0, le=1.0)
    critical_prediction_distribution_score: float = Field(ge=0.0, le=1.0)
    warning_low_confidence_rate_increase: float = Field(ge=0.0, le=1.0)
    critical_low_confidence_rate_increase: float = Field(ge=0.0, le=1.0)
    warning_macro_f1_decline: float = Field(ge=0.0, le=1.0)
    critical_macro_f1_decline: float = Field(ge=0.0, le=1.0)
    minimum_warning_signals: int = Field(gt=1)
    minimum_critical_signals: int = Field(gt=1)

    @model_validator(mode="after")
    def critical_thresholds_must_not_be_weaker(self) -> Self:
        pairs = (
            (
                self.warning_drifted_feature_share,
                self.critical_drifted_feature_share,
                "drifted feature share",
            ),
            (
                self.warning_prediction_distribution_score,
                self.critical_prediction_distribution_score,
                "prediction distribution",
            ),
            (
                self.warning_low_confidence_rate_increase,
                self.critical_low_confidence_rate_increase,
                "low-confidence increase",
            ),
            (
                self.warning_macro_f1_decline,
                self.critical_macro_f1_decline,
                "macro F1 decline",
            ),
        )
        for warning, critical, name in pairs:
            if critical < warning:
                raise ValueError(
                    f"critical {name} threshold must be at least its warning threshold"
                )
        return self


class MonitoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reference: ReferenceSettings
    current_window: CurrentWindowSettings
    drift: DriftSettings
    quality: QualitySettings
    alerts: AlertSettings

    @classmethod
    def load(cls, path: Path = DEFAULT_MONITORING_CONFIG_PATH) -> Self:
        with path.open(encoding="utf-8") as config_file:
            value = yaml.safe_load(config_file)
        return cls.model_validate(value)
