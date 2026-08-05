"""Tests for validation metrics and privacy-safe error analysis."""

from __future__ import annotations

import numpy as np

from ticket_router.modeling.artifacts import build_error_analysis
from ticket_router.modeling.baselines import build_baseline_pipelines
from ticket_router.modeling.config import BaselineSettings
from ticket_router.modeling.evaluation import evaluate_classifier


def test_metric_calculation_has_complete_shapes(
    tiny_text_classification_data: tuple[list[str], list[str]],
    small_baseline_config: BaselineSettings,
) -> None:
    texts, labels = tiny_text_classification_data
    label_order = ("Billing", "Technical", "Returns")
    pipeline = build_baseline_pipelines(small_baseline_config, random_seed=42)[
        "tfidf_complement_nb"
    ].fit(texts, labels)

    result = evaluate_classifier(
        pipeline,
        validation_texts=texts,
        validation_labels=labels,
        label_order=label_order,
    )

    assert result.predictions.shape == (len(texts),)
    assert result.confidences.shape == (len(texts),)
    assert result.confusion_matrix.shape == (3, 3)
    assert {row["class"] for row in result.per_class_metrics} == set(label_order)
    assert {
        "macro_f1",
        "weighted_f1",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "log_loss",
    }.issubset(result.metrics)
    assert np.all((result.confidences >= 0.0) & (result.confidences <= 1.0))


def test_error_analysis_does_not_write_raw_text(
    tiny_text_classification_data: tuple[list[str], list[str]],
    small_baseline_config: BaselineSettings,
) -> None:
    texts, labels = tiny_text_classification_data
    pipeline = build_baseline_pipelines(small_baseline_config, random_seed=42)[
        "dummy_most_frequent"
    ].fit(texts, labels)
    result = evaluate_classifier(
        pipeline,
        validation_texts=texts,
        validation_labels=labels,
        label_order=("Billing", "Technical", "Returns"),
    )

    report = build_error_analysis(
        result,
        validation_texts=texts,
        validation_labels=labels,
        label_order=("Billing", "Technical", "Returns"),
        sample_size=3,
        confused_pair_count=3,
    )
    serialized = str(report)

    assert "invoice" not in serialized
    assert "network" not in serialized
    assert "[TOKEN]" in serialized
