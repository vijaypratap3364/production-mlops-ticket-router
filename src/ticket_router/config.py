"""Typed application and experiment configuration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CONFIG_PATH = Path("configs/base.yaml")


class DatasetSettings(BaseModel):
    """Versioned dataset-selection settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repository: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    language_filter: str = Field(min_length=2)
    target_column: str = Field(min_length=1)
    text_columns: tuple[str, ...] = Field(min_length=1)
    number_of_target_queues: int = Field(gt=1)
    minimum_class_count: int = Field(gt=0)

    @field_validator("text_columns")
    @classmethod
    def text_columns_must_be_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject duplicate text features before any data code is introduced."""
        if len(set(value)) != len(value):
            raise ValueError("text_columns must contain unique column names")
        return value


class SplitRatios(BaseModel):
    """Train, validation, and sealed-test allocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    train: float = Field(gt=0.0, lt=1.0)
    validation: float = Field(gt=0.0, lt=1.0)
    test: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def ratios_must_sum_to_one(self) -> Self:
        """Require a complete, non-overlapping split allocation."""
        total = self.train + self.validation + self.test
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"split ratios must sum to 1.0; received {total}")
        return self


class MLflowSettings(BaseModel):
    """Versioned local MLflow identifiers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tracking_uri: str = Field(min_length=1)
    model_name: str = Field(min_length=1)


class AnalysisSettings(BaseModel):
    """Versioned validation and privacy-safe EDA thresholds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_subject_characters: int = Field(gt=0)
    max_body_characters: int = Field(gt=0)
    max_text_characters: int = Field(gt=0)
    near_empty_word_threshold: int = Field(ge=0)
    template_min_group_size: int = Field(gt=1)
    token_min_document_frequency: int = Field(gt=1)
    common_tokens_per_class: int = Field(gt=0)


class ProjectSettings(BaseModel):
    """Non-secret, version-controlled project configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    global_random_seed: int = Field(ge=0)
    dataset: DatasetSettings
    split_ratios: SplitRatios
    analysis: AnalysisSettings
    mlflow: MLflowSettings


class Settings(BaseSettings):
    """Central settings assembled from versioned YAML and local environment values."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    project_config: ProjectSettings
    project_name: str = "production-mlops-ticket-router"
    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    global_random_seed: int | None = Field(default=None, ge=0)
    mlflow_tracking_uri: str | None = None
    store_raw_ticket_content: bool = False
    input_hmac_secret: SecretStr | None = None

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        """Normalize and validate configured standard-library log levels."""
        normalized = value.upper()
        allowed_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        if normalized not in allowed_levels:
            allowed = ", ".join(sorted(allowed_levels))
            raise ValueError(f"log_level must be one of: {allowed}")
        return normalized

    @property
    def random_seed(self) -> int:
        """Return the environment override or the reproducible project default."""
        return (
            self.global_random_seed
            if self.global_random_seed is not None
            else self.project_config.global_random_seed
        )

    @property
    def effective_mlflow_tracking_uri(self) -> str:
        """Return the deployment override or the versioned local MLflow URI."""
        return self.mlflow_tracking_uri or self.project_config.mlflow.tracking_uri

    @classmethod
    def load(
        cls,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        *,
        env_file: str | Path | None = ".env",
    ) -> Settings:
        """Load and validate YAML before applying environment-specific overrides."""
        path = Path(config_path)
        with path.open(encoding="utf-8") as config_file:
            raw_config: Any = yaml.safe_load(config_file)

        if not isinstance(raw_config, dict):
            raise ValueError(f"configuration root must be a mapping: {path}")

        project_config = ProjectSettings.model_validate(raw_config)
        return cls(project_config=project_config, _env_file=env_file)
