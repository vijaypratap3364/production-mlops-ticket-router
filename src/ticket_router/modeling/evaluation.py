"""Validation metrics and bounded batch-inference benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from time import perf_counter
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]


@dataclass(frozen=True)
class EvaluationResult:
    """All deterministic validation outputs needed by artifact writers."""

    predictions: NDArray[np.object_]
    confidences: NDArray[np.float64]
    metrics: dict[str, float]
    classification_report: dict[str, object]
    confusion_matrix: NDArray[np.int64]
    per_class_metrics: tuple[dict[str, float | int | str], ...]


def evaluate_classifier(
    pipeline: Pipeline,
    *,
    validation_texts: list[str],
    validation_labels: list[str],
    label_order: tuple[str, ...],
) -> EvaluationResult:
    """Evaluate one already-fitted classifier on validation data only."""
    if not validation_texts or len(validation_texts) != len(validation_labels):
        raise ValueError("Validation text and labels must be non-empty and aligned.")
    predictions = cast(
        NDArray[np.object_],
        np.asarray(pipeline.predict(validation_texts), dtype=object),
    )
    probabilities = _predict_probabilities(pipeline, validation_texts)
    model_classes = tuple(str(value) for value in cast(NDArray[np.object_], pipeline.classes_))
    confidences = probabilities.max(axis=1)
    report_raw = cast(
        dict[str, Any],
        classification_report(
            validation_labels,
            predictions,
            labels=list(label_order),
            output_dict=True,
            zero_division=0,
        ),
    )
    report = cast(dict[str, object], _to_builtin(report_raw))
    matrix = cast(
        NDArray[np.int64],
        confusion_matrix(validation_labels, predictions, labels=list(label_order)).astype(np.int64),
    )
    correct = predictions == np.asarray(validation_labels, dtype=object)
    metrics = {
        "macro_f1": float(
            f1_score(validation_labels, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(validation_labels, predictions, average="weighted", zero_division=0)
        ),
        "accuracy": float(accuracy_score(validation_labels, predictions)),
        "macro_precision": float(
            precision_score(validation_labels, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(validation_labels, predictions, average="macro", zero_division=0)
        ),
        "log_loss": float(log_loss(validation_labels, probabilities, labels=list(model_classes))),
        "mean_prediction_confidence": float(confidences.mean()),
        "mean_correct_confidence": _conditional_mean(confidences, correct),
        "mean_incorrect_confidence": _conditional_mean(confidences, ~correct),
    }
    per_class = tuple(_per_class_row(label, report_raw[label]) for label in label_order)
    return EvaluationResult(
        predictions=predictions,
        confidences=confidences,
        metrics=metrics,
        classification_report=report,
        confusion_matrix=matrix,
        per_class_metrics=per_class,
    )


def predict_in_batches(
    pipeline: Pipeline,
    texts: list[str],
    *,
    batch_size: int,
) -> NDArray[np.object_]:
    """Predict a bounded list of texts without changing fitted state."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not texts:
        return np.asarray([], dtype=object)
    batches = [
        np.asarray(pipeline.predict(texts[start : start + batch_size]), dtype=object)
        for start in range(0, len(texts), batch_size)
    ]
    return cast(NDArray[np.object_], np.concatenate(batches))


def benchmark_batch_inference(
    pipeline: Pipeline,
    texts: list[str],
    *,
    batch_size: int,
    repeats: int,
) -> dict[str, float | int]:
    """Measure warm inference locally and report median elapsed time."""
    if not texts:
        raise ValueError("Cannot benchmark an empty validation dataset")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    pipeline.predict(texts[: min(batch_size, len(texts))])
    durations: list[float] = []
    for _ in range(repeats):
        started = perf_counter()
        predictions = predict_in_batches(pipeline, texts, batch_size=batch_size)
        elapsed = perf_counter() - started
        if len(predictions) != len(texts):
            raise RuntimeError("Batch inference returned an unexpected prediction count.")
        durations.append(elapsed)
    median_seconds = median(durations)
    return {
        "records": len(texts),
        "batch_size": batch_size,
        "repeats": repeats,
        "median_total_seconds": median_seconds,
        "median_milliseconds_per_record": median_seconds * 1000.0 / len(texts),
        "median_records_per_second": len(texts) / median_seconds,
    }


def _predict_probabilities(pipeline: Pipeline, texts: list[str]) -> NDArray[np.float64]:
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError("Baseline pipeline does not provide valid prediction probabilities.")
    probabilities = np.asarray(pipeline.predict_proba(texts), dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(texts):
        raise ValueError("predict_proba returned an invalid shape.")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0.0).any()
        or (probabilities > 1.0).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-7)
    ):
        raise ValueError("predict_proba returned invalid probabilities.")
    return cast(NDArray[np.float64], probabilities)


def _conditional_mean(values: NDArray[np.float64], mask: NDArray[np.bool_]) -> float:
    return float(values[mask].mean()) if mask.any() else 0.0


def _per_class_row(label: str, metrics: dict[str, Any]) -> dict[str, float | int | str]:
    return {
        "class": label,
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1-score"]),
        "support": int(metrics["support"]),
    }


def _to_builtin(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
