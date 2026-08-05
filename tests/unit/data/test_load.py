"""Tests for feature isolation and sealed test-split access."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import polars as pl
import pytest

from ticket_router.data.load import (
    DatasetAccessError,
    TrainingSplitName,
    build_modeling_dataset,
    load_final_evaluation_split,
    load_training_split,
)
from ticket_router.features.contracts import FeatureLeakageError


def _write_splits(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(
        {
            "model_text": ["safe model input"],
            "queue": ["Queue A"],
            "answer": ["forbidden agent response"],
        }
    )
    for split in ("train", "validation", "test"):
        frame.write_parquet(path / f"{split}.parquet")


def test_training_loader_exposes_only_model_text(tmp_path: Path) -> None:
    _write_splits(tmp_path)

    dataset = load_training_split(tmp_path, "train")

    assert dataset.features.columns == ["model_text"]
    assert dataset.target.to_list() == ["Queue A"]


def test_training_loader_cannot_access_sealed_test_split(tmp_path: Path) -> None:
    _write_splits(tmp_path)

    with pytest.raises(DatasetAccessError, match="test split is sealed"):
        load_training_split(tmp_path, cast(TrainingSplitName, "test"))
    with pytest.raises(DatasetAccessError, match="Test access denied"):
        load_final_evaluation_split(tmp_path)

    authorized = load_final_evaluation_split(tmp_path, final_evaluation_authorized=True)
    assert authorized.features.columns == ["model_text"]


def test_forbidden_feature_column_is_rejected() -> None:
    frame = pl.DataFrame({"model_text": ["safe"], "queue": ["Queue A"]})

    with pytest.raises(FeatureLeakageError, match="forbidden"):
        build_modeling_dataset(frame, feature_columns=("model_text", "queue"))
