"""Typed configuration for restrained candidate-model experimentation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_EXPERIMENT_CONFIG_PATH = Path("configs/experiments.yaml")


class CandidateSearchSettings(BaseModel):
    """Small configuration-driven search spaces shared by candidate families."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cv_folds: int = Field(ge=2, le=5)
    iterations_per_candidate: int = Field(ge=1, le=10)
    n_jobs: int = Field(ge=1)
    word_ngram_ranges: tuple[tuple[int, int], ...] = Field(min_length=1)
    character_ngram_ranges: tuple[tuple[int, int], ...] = Field(min_length=1)
    min_df_values: tuple[int, ...] = Field(min_length=1)
    max_df_values: tuple[float, ...] = Field(min_length=1)
    max_features_values: tuple[int, ...] = Field(min_length=1)
    regularization_c_values: tuple[float, ...] = Field(min_length=1)
    class_weight_values: tuple[Literal["balanced"] | None, ...] = Field(min_length=1)
    complement_nb_alpha_values: tuple[float, ...] = Field(min_length=1)
    calibration_cv_folds: int = Field(ge=2, le=5)

    @field_validator("word_ngram_ranges")
    @classmethod
    def validate_word_ngrams(
        cls, value: tuple[tuple[int, int], ...]
    ) -> tuple[tuple[int, int], ...]:
        if any(item not in {(1, 1), (1, 2)} for item in value):
            raise ValueError("word n-grams must be (1, 1) or (1, 2)")
        return value

    @field_validator("character_ngram_ranges")
    @classmethod
    def validate_character_ngrams(
        cls, value: tuple[tuple[int, int], ...]
    ) -> tuple[tuple[int, int], ...]:
        if any(minimum < 2 or maximum < minimum or maximum > 8 for minimum, maximum in value):
            raise ValueError("character n-grams must be ordered and remain between 2 and 8")
        return value

    @field_validator("min_df_values", "max_features_values")
    @classmethod
    def validate_positive_integers(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(item <= 0 for item in value):
            raise ValueError("search integer values must be positive")
        return value

    @field_validator(
        "regularization_c_values",
        "complement_nb_alpha_values",
    )
    @classmethod
    def validate_positive_floats(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(item <= 0.0 for item in value):
            raise ValueError("search float values must be positive")
        return value

    @field_validator("max_df_values")
    @classmethod
    def validate_max_df(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(item <= 0.0 or item > 1.0 for item in value):
            raise ValueError("max_df values must be in the interval (0, 1]")
        return value


class CandidateSelectionSettings(BaseModel):
    """Operational and quality guardrails applied before ranking by macro F1."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_per_class_recall: float = Field(ge=0.0, le=1.0)
    major_class_minimum_support: int = Field(gt=0)
    major_class_minimum_recall: float = Field(ge=0.0, le=1.0)
    maximum_inference_milliseconds_per_record: float = Field(gt=0.0)
    maximum_serialized_model_size_mb: float = Field(gt=0.0)
    maximum_cv_macro_f1_standard_deviation: float = Field(ge=0.0)
    maximum_cv_to_validation_macro_f1_drop: float = Field(ge=0.0)


class CandidateEvaluationSettings(BaseModel):
    """Bounded runtime and report-generation settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    inference_batch_size: int = Field(gt=0)
    inference_repeats: int = Field(gt=0, le=10)
    error_sample_size: int = Field(gt=0, le=100)
    confused_pair_count: int = Field(gt=0, le=100)
    confidence_bins: int = Field(ge=5, le=50)


class CandidateExperimentSettings(BaseModel):
    """Complete MLflow candidate-search configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_name: str = Field(min_length=1)
    local_tracking_directory: Path
    allow_local_tracking_fallback: bool = True
    primary_metric: Literal["macro_f1"] = "macro_f1"
    search: CandidateSearchSettings
    selection: CandidateSelectionSettings
    evaluation: CandidateEvaluationSettings

    @classmethod
    def load(cls, path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH) -> Self:
        config_path = Path(path)
        with config_path.open(encoding="utf-8") as config_file:
            raw_config: Any = yaml.safe_load(config_file)
        if not isinstance(raw_config, dict):
            raise ValueError(f"experiment configuration root must be a mapping: {config_path}")
        return cls.model_validate(raw_config)


def experiment_configuration_hash(
    config: CandidateExperimentSettings,
    *,
    random_seed: int,
) -> str:
    """Hash every versioned choice that can alter search or selection."""
    payload = {"experiment": config.model_dump(mode="json"), "random_seed": random_seed}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
