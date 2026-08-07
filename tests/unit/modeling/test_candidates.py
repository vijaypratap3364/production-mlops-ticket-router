"""Tests for candidate-family construction and train-only data access."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from ticket_router.data.load import ModelingDataset, TrainingSplitName
from ticket_router.modeling import train_candidates as training_module
from ticket_router.modeling.candidates import build_candidate_specs
from ticket_router.modeling.experiment_config import CandidateExperimentSettings


def test_all_required_candidate_families_are_constructed(
    small_experiment_config: CandidateExperimentSettings,
) -> None:
    specs = build_candidate_specs(small_experiment_config.search, random_seed=42)

    assert {spec.name for spec in specs} == {
        "word_tfidf_logistic_regression",
        "character_tfidf_logistic_regression",
        "combined_word_character_logistic_regression",
        "word_tfidf_calibrated_linear_svc",
        "word_tfidf_complement_nb",
    }
    assert sum(spec.requires_calibration for spec in specs) == 1


def test_word_search_guarantees_stage5_incumbent_configuration() -> None:
    config = CandidateExperimentSettings.load("configs/experiments.yaml")
    spec = build_candidate_specs(config.search, random_seed=42)[0]

    assert isinstance(spec.parameter_distributions, list)
    incumbent = spec.parameter_distributions[0]
    assert incumbent == {
        "tfidf__ngram_range": [(1, 2)],
        "tfidf__min_df": [2],
        "tfidf__max_df": [1.0],
        "tfidf__max_features": [50000],
        "classifier__estimator__C": [1.0],
        "classifier__estimator__class_weight": ["balanced"],
    }


def test_candidate_dataset_boundary_never_requests_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[str] = []
    fixture = ModelingDataset(
        features=pl.DataFrame({"model_text": ["fixture"]}),
        target=pl.Series("queue", ["Queue"]),
    )

    def recording_loader(_path: Path, split: TrainingSplitName) -> ModelingDataset:
        requested.append(split)
        return fixture

    monkeypatch.setattr(training_module, "load_training_split", recording_loader)

    training, validation = training_module.load_candidate_datasets(Path("processed"))

    assert requested == ["train", "validation"]
    assert training is fixture
    assert validation is fixture
