"""Small deterministic text fixtures for baseline-model unit tests."""

from __future__ import annotations

import pytest

from ticket_router.modeling.config import BaselineSettings
from ticket_router.modeling.experiment_config import CandidateExperimentSettings


@pytest.fixture
def tiny_text_classification_data() -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for label, vocabulary in (
        ("Billing", "invoice payment refund"),
        ("Technical", "server network error"),
        ("Returns", "return exchange parcel"),
    ):
        for index in range(12):
            texts.append(f"{vocabulary} request number {index % 4}")
            labels.append(label)
    return texts, labels


@pytest.fixture
def small_baseline_config() -> BaselineSettings:
    return BaselineSettings.model_validate(
        {
            "primary_metric": "macro_f1",
            "tfidf": {
                "ngram_range": [1, 2],
                "min_df": 1,
                "max_features": 1000,
                "sublinear_tf": True,
            },
            "logistic_regression": {
                "c": 1.0,
                "class_weight": "balanced",
                "max_iter": 500,
                "solver": "liblinear",
            },
            "complement_nb": {"alpha": 1.0},
            "evaluation": {
                "inference_batch_size": 5,
                "inference_repeats": 1,
                "error_sample_size": 2,
                "confused_pair_count": 3,
            },
        }
    )


@pytest.fixture
def small_experiment_config() -> CandidateExperimentSettings:
    return CandidateExperimentSettings.model_validate(
        {
            "experiment_name": "fixture-candidate-search",
            "local_tracking_directory": "mlruns",
            "allow_local_tracking_fallback": True,
            "primary_metric": "macro_f1",
            "search": {
                "cv_folds": 2,
                "iterations_per_candidate": 1,
                "n_jobs": 1,
                "word_ngram_ranges": [[1, 1]],
                "character_ngram_ranges": [[3, 5]],
                "min_df_values": [1],
                "max_df_values": [1.0],
                "max_features_values": [1000],
                "regularization_c_values": [1.0],
                "class_weight_values": ["balanced"],
                "complement_nb_alpha_values": [1.0],
                "calibration_cv_folds": 2,
            },
            "selection": {
                "minimum_per_class_recall": 0.20,
                "major_class_minimum_support": 5,
                "major_class_minimum_recall": 0.20,
                "maximum_inference_milliseconds_per_record": 10.0,
                "maximum_serialized_model_size_mb": 10.0,
                "maximum_cv_macro_f1_standard_deviation": 0.20,
                "maximum_cv_to_validation_macro_f1_drop": 0.20,
            },
            "evaluation": {
                "inference_batch_size": 5,
                "inference_repeats": 1,
                "error_sample_size": 2,
                "confused_pair_count": 3,
                "confidence_bins": 5,
            },
        }
    )
