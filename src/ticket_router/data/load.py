"""Split-aware loaders that keep the final test artifact sealed by default."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl

from ticket_router.features.contracts import validate_model_feature_columns

TrainingSplitName = Literal["train", "validation"]
MODEL_FEATURE_COLUMNS = ("model_text",)
TARGET_COLUMN = "queue"


class DatasetAccessError(ValueError):
    """Raised when code attempts unauthorized split or feature access."""


@dataclass(frozen=True)
class ModelingDataset:
    """Physically separated model inputs and target labels."""

    features: pl.DataFrame
    target: pl.Series


def load_training_split(
    processed_dir: Path,
    split: TrainingSplitName,
) -> ModelingDataset:
    """Load only train or validation data for routine model development."""
    if split not in {"train", "validation"}:
        raise DatasetAccessError(
            "Routine training utilities may load only 'train' or 'validation'; "
            "the test split is sealed for explicit final evaluation."
        )
    return build_modeling_dataset(pl.read_parquet(processed_dir / f"{split}.parquet"))


def load_final_evaluation_split(
    processed_dir: Path,
    *,
    final_evaluation_authorized: bool = False,
) -> ModelingDataset:
    """Load test data only through an explicit final-evaluation authorization."""
    if not final_evaluation_authorized:
        raise DatasetAccessError(
            "Test access denied. Set final_evaluation_authorized=True only after candidate "
            "selection and configuration are frozen."
        )
    return build_modeling_dataset(pl.read_parquet(processed_dir / "test.parquet"))


def build_modeling_dataset(
    frame: pl.DataFrame,
    *,
    feature_columns: tuple[str, ...] = MODEL_FEATURE_COLUMNS,
) -> ModelingDataset:
    """Expose only allowlisted predictors and keep the target physically separate."""
    validate_model_feature_columns(feature_columns)
    if feature_columns != MODEL_FEATURE_COLUMNS:
        raise DatasetAccessError(
            "Prepared sparse-text models must receive exactly one feature: model_text."
        )
    required = {*feature_columns, TARGET_COLUMN}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DatasetAccessError("Prepared split is missing columns: " + ", ".join(missing))
    features = frame.select(feature_columns)
    validate_model_feature_columns(features.columns)
    return ModelingDataset(features=features, target=frame[TARGET_COLUMN].clone())
