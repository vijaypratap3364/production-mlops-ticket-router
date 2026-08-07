"""Run-scoped, privacy-safe artifact writers for baseline experiments."""

from __future__ import annotations

import os
import platform
import re
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import joblib  # type: ignore[import-untyped]
import matplotlib
import numpy as np
import polars as pl
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]

from ticket_router.data.manifests import atomic_write_json
from ticket_router.hashing import sha256_file
from ticket_router.modeling.evaluation import EvaluationResult

matplotlib.use("Agg")
from matplotlib import pyplot as plt

TOKEN_PATTERN = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)


@dataclass(frozen=True)
class ModelArtifactSummary:
    """Values needed for the cross-model leaderboard and run manifest."""

    model_name: str
    artifact_directory: Path
    metrics: dict[str, float]
    training_duration_seconds: float
    serialized_model_size_bytes: int
    serialized_model_sha256: str
    inference_benchmark: dict[str, float | int]


def build_error_analysis(
    result: EvaluationResult,
    *,
    validation_texts: list[str],
    validation_labels: list[str],
    label_order: tuple[str, ...],
    sample_size: int,
    confused_pair_count: int,
    evaluation_split: Literal["validation", "test"] = "validation",
) -> dict[str, object]:
    """Create aggregate confusions and redacted evaluation examples."""
    confused_pairs: list[dict[str, int | str]] = []
    for actual_index, actual_label in enumerate(label_order):
        for predicted_index, predicted_label in enumerate(label_order):
            if actual_index == predicted_index:
                continue
            count = int(result.confusion_matrix[actual_index, predicted_index])
            if count:
                confused_pairs.append(
                    {
                        "actual_class": actual_label,
                        "predicted_class": predicted_label,
                        "count": count,
                    }
                )
    confused_pairs.sort(
        key=lambda row: (
            -int(row["count"]),
            str(row["actual_class"]).casefold(),
            str(row["predicted_class"]).casefold(),
        )
    )
    lowest_recall = sorted(
        result.per_class_metrics,
        key=lambda row: (float(row["recall"]), str(row["class"]).casefold()),
    )
    incorrect_indices = [
        index
        for index, (actual, predicted) in enumerate(
            zip(validation_labels, result.predictions, strict=True)
        )
        if actual != str(predicted)
    ]
    correct_indices = [
        index
        for index, (actual, predicted) in enumerate(
            zip(validation_labels, result.predictions, strict=True)
        )
        if actual == str(predicted)
    ]
    incorrect_indices.sort(key=lambda index: (-result.confidences[index], index))
    correct_indices.sort(key=lambda index: (result.confidences[index], index))
    return {
        "privacy_note": (
            "Examples contain token-redacted structural excerpts only; raw ticket content is "
            "not written to this report."
        ),
        "most_frequently_confused_class_pairs": confused_pairs[:confused_pair_count],
        "lowest_recall_classes": list(lowest_recall[: min(5, len(lowest_recall))]),
        "high_confidence_incorrect_predictions": [
            _example(index, result, validation_texts, validation_labels, evaluation_split)
            for index in incorrect_indices[:sample_size]
        ],
        "low_confidence_correct_predictions": [
            _example(index, result, validation_texts, validation_labels, evaluation_split)
            for index in correct_indices[:sample_size]
        ],
    }


def write_model_artifacts(
    *,
    run_directory: Path,
    model_name: str,
    pipeline: Pipeline,
    result: EvaluationResult,
    validation_texts: list[str],
    validation_labels: list[str],
    label_order: tuple[str, ...],
    model_configuration: dict[str, object],
    training_duration_seconds: float,
    inference_benchmark: dict[str, float | int],
    lineage: dict[str, object],
    error_sample_size: int,
    confused_pair_count: int,
    evaluation_split: Literal["validation", "test"] = "validation",
) -> ModelArtifactSummary:
    """Serialize one fitted pipeline and every required evaluation artifact."""
    model_directory = run_directory / model_name
    model_directory.mkdir(parents=True, exist_ok=False)
    model_path = model_directory / "pipeline.joblib"
    _atomic_joblib_dump(pipeline, model_path)
    model_size = model_path.stat().st_size
    model_sha256 = sha256_file(model_path)
    combined_metrics = {
        **result.metrics,
        "training_duration_seconds": training_duration_seconds,
        "inference_milliseconds_per_record": float(
            inference_benchmark["median_milliseconds_per_record"]
        ),
        "serialized_model_size_bytes": float(model_size),
    }
    atomic_write_json(model_directory / "metrics.json", combined_metrics)
    atomic_write_json(
        model_directory / "classification_report.json",
        result.classification_report,
    )
    _write_per_class_metrics(
        model_directory / "per_class_metrics.csv",
        result.per_class_metrics,
    )
    _write_confusion_matrix(
        model_directory / "confusion_matrix.png",
        result.confusion_matrix,
        label_order,
        evaluation_split,
    )
    _write_predictions(
        model_directory / f"{evaluation_split}_predictions.parquet",
        result,
        validation_labels,
        evaluation_split,
    )
    atomic_write_json(model_directory / "model_configuration.json", model_configuration)
    atomic_write_json(model_directory / "inference_benchmark.json", inference_benchmark)
    atomic_write_json(
        model_directory / "training_metadata.json",
        {
            **lineage,
            "training_duration_seconds": training_duration_seconds,
            "serialized_model_size_bytes": model_size,
            "serialized_model_sha256": model_sha256,
            "environment": environment_versions(),
        },
    )
    atomic_write_json(
        model_directory / "error_analysis.json",
        build_error_analysis(
            result,
            validation_texts=validation_texts,
            validation_labels=validation_labels,
            label_order=label_order,
            sample_size=error_sample_size,
            confused_pair_count=confused_pair_count,
            evaluation_split=evaluation_split,
        ),
    )
    return ModelArtifactSummary(
        model_name=model_name,
        artifact_directory=model_directory,
        metrics=combined_metrics,
        training_duration_seconds=training_duration_seconds,
        serialized_model_size_bytes=model_size,
        serialized_model_sha256=model_sha256,
        inference_benchmark=inference_benchmark,
    )


def environment_versions() -> dict[str, str]:
    """Capture runtime and direct numerical/model package versions."""
    packages = (
        "cloudpickle",
        "joblib",
        "matplotlib",
        "numpy",
        "polars",
        "scikit-learn",
        "scipy",
    )
    result = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _example(
    index: int,
    result: EvaluationResult,
    texts: list[str],
    labels: list[str],
    evaluation_split: Literal["validation", "test"],
) -> dict[str, float | int | str]:
    return {
        f"{evaluation_split}_row_index": index,
        "actual_class": labels[index],
        "predicted_class": str(result.predictions[index]),
        "confidence": float(result.confidences[index]),
        "redacted_excerpt": _redacted_excerpt(texts[index]),
    }


def _redacted_excerpt(text: str, *, maximum_characters: int = 180) -> str:
    excerpt = TOKEN_PATTERN.sub("[TOKEN]", text[:maximum_characters])
    if len(text) > maximum_characters:
        excerpt += "…"
    return excerpt


def _write_predictions(
    path: Path,
    result: EvaluationResult,
    validation_labels: list[str],
    evaluation_split: Literal["validation", "test"],
) -> None:
    frame = pl.DataFrame(
        {
            f"{evaluation_split}_row_index": range(len(validation_labels)),
            "actual_class": validation_labels,
            "predicted_class": [str(value) for value in result.predictions],
            "confidence": result.confidences,
            "correct": [
                actual == str(predicted)
                for actual, predicted in zip(
                    validation_labels,
                    result.predictions,
                    strict=True,
                )
            ],
        }
    )
    _atomic_polars_write(frame, path, parquet=True)


def _write_per_class_metrics(
    path: Path,
    rows: tuple[dict[str, float | int | str], ...],
) -> None:
    _atomic_polars_write(pl.DataFrame(rows), path, parquet=False)


def _write_confusion_matrix(
    path: Path,
    matrix: np.ndarray[Any, np.dtype[np.int64]],
    labels: tuple[str, ...],
    evaluation_split: Literal["validation", "test"],
) -> None:
    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title=f"{evaluation_split.title()} confusion matrix",
        xlabel="Predicted queue",
        ylabel="Actual queue",
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    figure.tight_layout()
    temporary = path.with_name(f".tmp-{uuid4().hex[:8]}")
    try:
        figure.savefig(temporary, format="png", dpi=140, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)


def _atomic_joblib_dump(pipeline: Pipeline, path: Path) -> None:
    temporary = path.with_name(f".tmp-{uuid4().hex[:8]}")
    try:
        joblib.dump(pipeline, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_polars_write(frame: pl.DataFrame, path: Path, *, parquet: bool) -> None:
    temporary = path.with_name(f".tmp-{uuid4().hex[:8]}")
    try:
        if parquet:
            frame.write_parquet(temporary, compression="zstd")
        else:
            frame.write_csv(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
