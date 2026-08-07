"""Typed frozen configuration for final evaluation and registry promotion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_FINAL_MODEL_CONFIG_PATH = Path("configs/final_model.yaml")


class SelectedCandidateSettings(BaseModel):
    """Stage 6 decision frozen before the final test boundary is opened."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["word_tfidf_calibrated_linear_svc"]
    stage6_run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    stage6_configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_macro_f1: float = Field(ge=0.0, le=1.0)
    validation_weighted_f1: float = Field(ge=0.0, le=1.0)
    word_ngram_range: tuple[Literal[1], Literal[2]]
    min_df: int = Field(gt=0)
    max_df: float = Field(gt=0.0, le=1.0)
    max_features: int = Field(gt=0)
    sublinear_tf: bool
    regularization_c: float = Field(gt=0.0)
    class_weight: Literal["balanced"] | None
    max_iter: int = Field(ge=100)
    calibration_cv_folds: int = Field(ge=2, le=5)
    calibration_method: Literal["sigmoid"]


class FinalEvaluationSettings(BaseModel):
    """Bounded test reporting and latency benchmark settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    inference_batch_size: int = Field(gt=0)
    inference_repeats: int = Field(ge=3, le=20)
    error_sample_size: int = Field(gt=0, le=100)
    confused_pair_count: int = Field(gt=0, le=100)


class RegistrySettings(BaseModel):
    """Stable registered-model and alias names."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str = Field(min_length=1)
    candidate_alias: str = Field(min_length=1)
    champion_alias: str = Field(min_length=1)


class PromotionSettings(BaseModel):
    """Frozen absolute and champion-relative promotion gates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_macro_f1: float = Field(ge=0.0, le=1.0)
    maximum_macro_f1_regression: float = Field(ge=0.0, le=1.0)
    minimum_per_class_recall: float = Field(ge=0.0, le=1.0)
    maximum_inference_milliseconds_per_record: float = Field(gt=0.0)
    required_input_dtype: Literal["str"]
    required_output_dtype: Literal["str"]
    require_model_load: bool = True
    require_prediction_contract: bool = True
    require_signature_compatibility: bool = True


class FinalModelConfig(BaseModel):
    """Complete Stage 7 configuration loaded before test access."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_name: str = Field(min_length=1)
    local_tracking_directory: Path
    allow_local_tracking_fallback: bool = True
    selected_candidate: SelectedCandidateSettings
    evaluation: FinalEvaluationSettings
    registry: RegistrySettings
    promotion: PromotionSettings

    @classmethod
    def load(cls, path: str | Path = DEFAULT_FINAL_MODEL_CONFIG_PATH) -> Self:
        config_path = Path(path)
        with config_path.open(encoding="utf-8") as config_file:
            raw_config: Any = yaml.safe_load(config_file)
        if not isinstance(raw_config, dict):
            raise ValueError(f"final-model configuration root must be a mapping: {config_path}")
        return cls.model_validate(raw_config)


def final_model_configuration_hash(config: FinalModelConfig, *, random_seed: int) -> str:
    """Hash the frozen model, evaluation, registry, and gate decisions."""
    payload = {"final_model": config.model_dump(mode="json"), "random_seed": random_seed}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
