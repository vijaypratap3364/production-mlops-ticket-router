"""Small deterministic text fixtures for baseline-model unit tests."""

from __future__ import annotations

import pytest

from ticket_router.modeling.config import BaselineSettings


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
