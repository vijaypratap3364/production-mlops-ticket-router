"""Pipeline fitting, prediction, determinism, serialization, and batching tests."""

from __future__ import annotations

from pathlib import Path

import joblib  # type: ignore[import-untyped]
import numpy as np
import pytest

from ticket_router.modeling.baselines import build_baseline_pipelines
from ticket_router.modeling.config import BaselineSettings
from ticket_router.modeling.evaluation import predict_in_batches


@pytest.mark.parametrize(
    "model_name",
    ["dummy_most_frequent", "tfidf_logistic_regression", "tfidf_complement_nb"],
)
def test_pipeline_fit_predict_shape_and_labels(
    model_name: str,
    tiny_text_classification_data: tuple[list[str], list[str]],
    small_baseline_config: BaselineSettings,
) -> None:
    texts, labels = tiny_text_classification_data
    pipeline = build_baseline_pipelines(small_baseline_config, random_seed=42)[model_name]

    predictions = pipeline.fit(texts, labels).predict(texts[:7])

    assert predictions.shape == (7,)
    assert set(predictions).issubset(set(labels))


def test_logistic_predictions_are_reproducible(
    tiny_text_classification_data: tuple[list[str], list[str]],
    small_baseline_config: BaselineSettings,
) -> None:
    texts, labels = tiny_text_classification_data
    first = build_baseline_pipelines(small_baseline_config, random_seed=42)[
        "tfidf_logistic_regression"
    ]
    second = build_baseline_pipelines(small_baseline_config, random_seed=42)[
        "tfidf_logistic_regression"
    ]

    first_predictions = first.fit(texts, labels).predict(texts)
    second_predictions = second.fit(texts, labels).predict(texts)

    assert np.array_equal(first_predictions, second_predictions)


def test_serialization_reload_and_batch_inference(
    tmp_path: Path,
    tiny_text_classification_data: tuple[list[str], list[str]],
    small_baseline_config: BaselineSettings,
) -> None:
    texts, labels = tiny_text_classification_data
    pipeline = build_baseline_pipelines(small_baseline_config, random_seed=42)[
        "tfidf_logistic_regression"
    ].fit(texts, labels)
    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)

    reloaded = joblib.load(path)
    expected = pipeline.predict(texts)
    observed = predict_in_batches(reloaded, texts, batch_size=5)

    assert observed.shape == (len(texts),)
    assert np.array_equal(observed, expected)
