"""Typed, versioned configuration for sparse-text baseline experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_BASELINE_CONFIG_PATH = Path("configs/baseline.yaml")


class TfidfSettings(BaseModel):
    """Vocabulary settings fitted exclusively on the training split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ngram_range: tuple[int, int]
    min_df: int = Field(ge=1)
    max_features: int | None = Field(default=None, gt=0)
    sublinear_tf: bool = True

    @field_validator("ngram_range")
    @classmethod
    def allow_conservative_word_ngrams(cls, value: tuple[int, int]) -> tuple[int, int]:
        """Limit the baseline search surface to unigrams or unigram/bigrams."""
        if value not in {(1, 1), (1, 2)}:
            raise ValueError("ngram_range must be either (1, 1) or (1, 2)")
        return value


class LogisticRegressionSettings(BaseModel):
    """Deterministic linear classifier settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    c: float = Field(gt=0.0)
    class_weight: Literal["balanced"] = "balanced"
    max_iter: int = Field(ge=100)
    solver: Literal["liblinear"] = "liblinear"


class ComplementNBSettings(BaseModel):
    """Complement Naive Bayes settings for non-negative TF-IDF features."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    alpha: float = Field(gt=0.0)


class EvaluationSettings(BaseModel):
    """Bounded validation benchmark and privacy-safe error-report settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    inference_batch_size: int = Field(gt=0)
    inference_repeats: int = Field(gt=0, le=10)
    error_sample_size: int = Field(gt=0, le=100)
    confused_pair_count: int = Field(gt=0, le=100)


class BaselineSettings(BaseModel):
    """Complete baseline experiment configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    primary_metric: Literal["macro_f1"] = "macro_f1"
    tfidf: TfidfSettings
    logistic_regression: LogisticRegressionSettings
    complement_nb: ComplementNBSettings
    evaluation: EvaluationSettings

    @classmethod
    def load(cls, path: str | Path = DEFAULT_BASELINE_CONFIG_PATH) -> Self:
        """Read and validate a baseline YAML file."""
        config_path = Path(path)
        with config_path.open(encoding="utf-8") as config_file:
            raw_config: Any = yaml.safe_load(config_file)
        if not isinstance(raw_config, dict):
            raise ValueError(f"baseline configuration root must be a mapping: {config_path}")
        return cls.model_validate(raw_config)


def baseline_configuration_hash(config: BaselineSettings, *, random_seed: int) -> str:
    """Hash every versioned setting that can influence a baseline run."""
    payload = {
        "baseline": config.model_dump(mode="json"),
        "random_seed": random_seed,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
