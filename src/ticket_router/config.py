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


class APISettings(BaseModel):
    """Typed deployment inference validation and response limits."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    confidence_warning_threshold: float = Field(ge=0.0, le=1.0)
    default_top_k: int = Field(gt=0, le=20)
    maximum_batch_size: int = Field(gt=0, le=1000)
    maximum_subject_characters: int = Field(gt=0, le=2000)
    maximum_body_characters: int = Field(gt=0, le=20000)
    minimum_usable_characters: int = Field(gt=0)

    @model_validator(mode="after")
    def usable_text_limit_must_fit_inputs(self) -> Self:
        """Ensure at least one valid request can satisfy the usable-text rule."""
        maximum_combined = self.maximum_subject_characters + self.maximum_body_characters
        if self.minimum_usable_characters > maximum_combined:
            raise ValueError("minimum_usable_characters exceeds the combined input limits")
        return self


class TextPreprocessingSettings(BaseModel):
    """Conservative, stateless model-text preprocessing policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unicode_normalization: Literal["NFC", "NFKC"] = "NFKC"
    mask_email_addresses: bool = True
    mask_urls: bool = True
    mask_phone_numbers: bool = True
    email_mask: str = Field(min_length=1)
    url_mask: str = Field(min_length=1)
    phone_mask: str = Field(min_length=1)

    @model_validator(mode="after")
    def masks_must_be_unique(self) -> Self:
        """Keep redaction categories distinguishable in sparse text features."""
        masks = (self.email_mask, self.url_mask, self.phone_mask)
        if len(set(masks)) != len(masks):
            raise ValueError("preprocessing mask tokens must be unique")
        return self


class SplittingSettings(BaseModel):
    """Versioned duplicate and grouped-stratification validation policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    duplicate_policy: Literal["exclude_contradictory_group_exact_duplicates"]
    class_proportion_tolerance: float = Field(ge=0.0, lt=1.0)
    split_size_tolerance: float = Field(ge=0.0, lt=1.0)


class ProjectSettings(BaseModel):
    """Non-secret, version-controlled project configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    global_random_seed: int = Field(ge=0)
    dataset: DatasetSettings
    split_ratios: SplitRatios
    preprocessing: TextPreprocessingSettings
    splitting: SplittingSettings
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
    mlflow_registered_model_name: str | None = None
    mlflow_model_alias: str = "champion"
    database_url: SecretStr | None = None
    database_required: bool = False
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_confidence_warning_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    api_default_top_k: int = Field(default=3, gt=0, le=20)
    api_maximum_batch_size: int = Field(default=100, gt=0, le=1000)
    api_maximum_subject_characters: int = Field(default=2000, gt=0, le=2000)
    api_maximum_body_characters: int = Field(default=20000, gt=0, le=20000)
    api_minimum_usable_characters: int = Field(default=1, gt=0)
    dashboard_api_url: str = "http://127.0.0.1:8000"
    dashboard_request_timeout_seconds: float = Field(default=10.0, gt=0.0, le=60.0)
    store_raw_ticket_content: bool = False
    store_redacted_ticket_text: bool = False
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

    @property
    def effective_registered_model_name(self) -> str:
        """Return the deployment model-name override or the versioned default."""
        return self.mlflow_registered_model_name or self.project_config.mlflow.model_name

    @property
    def api_settings(self) -> APISettings:
        """Assemble the typed serving limits from deployment environment values."""
        return APISettings(
            confidence_warning_threshold=self.api_confidence_warning_threshold,
            default_top_k=self.api_default_top_k,
            maximum_batch_size=self.api_maximum_batch_size,
            maximum_subject_characters=self.api_maximum_subject_characters,
            maximum_body_characters=self.api_maximum_body_characters,
            minimum_usable_characters=self.api_minimum_usable_characters,
        )

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
