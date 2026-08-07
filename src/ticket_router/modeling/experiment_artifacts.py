"""Comparison, confidence, calibration, and leaderboard artifacts."""

from __future__ import annotations

import csv
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from uuid import uuid4

import matplotlib
import numpy as np

from ticket_router.modeling.evaluation import EvaluationResult

matplotlib.use("Agg")
from matplotlib import pyplot as plt

LEADERBOARD_COLUMNS = (
    "rank",
    "stage",
    "run_id",
    "model_name",
    "primary_metric",
    "macro_f1",
    "weighted_f1",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "log_loss",
    "cv_macro_f1_mean",
    "cv_macro_f1_standard_deviation",
    "selection_eligible",
    "rejection_reasons",
    "mlflow_run_id",
    "inference_milliseconds_per_record",
    "training_duration_seconds",
    "serialized_model_size_bytes",
    "data_manifest_sha256",
    "configuration_sha256",
    "artifact_directory",
)


def confidence_distribution(
    result: EvaluationResult,
    *,
    actual_labels: Sequence[str],
    bins: int,
) -> dict[str, object]:
    """Aggregate top-label confidence without retaining text or identifiers."""
    if len(actual_labels) != len(result.predictions):
        raise ValueError("Actual labels and predictions must be aligned for confidence reporting.")
    edges = np.linspace(0.0, 1.0, bins + 1)
    correct = np.asarray(
        [
            actual == str(predicted)
            for actual, predicted in zip(actual_labels, result.predictions, strict=True)
        ],
        dtype=bool,
    )
    counts, _ = np.histogram(result.confidences, bins=edges)
    correct_counts, _ = np.histogram(result.confidences[correct], bins=edges)
    incorrect_counts, _ = np.histogram(result.confidences[~correct], bins=edges)
    return {
        "bin_edges": edges.tolist(),
        "counts": counts.astype(int).tolist(),
        "correct_counts": correct_counts.astype(int).tolist(),
        "incorrect_counts": incorrect_counts.astype(int).tolist(),
        "mean": float(result.confidences.mean()),
        "minimum": float(result.confidences.min()),
        "maximum": float(result.confidences.max()),
        "record_count": len(result.confidences),
    }


def calibration_curve_points(
    result: EvaluationResult,
    *,
    actual_labels: Sequence[str],
    bins: int,
) -> tuple[list[float], list[float], list[int]]:
    """Calculate multiclass top-label reliability points."""
    if len(actual_labels) != len(result.predictions):
        raise ValueError("Actual labels and predictions must be aligned for calibration.")
    edges = np.linspace(0.0, 1.0, bins + 1)
    correctness = np.asarray(
        [
            actual == str(predicted)
            for actual, predicted in zip(actual_labels, result.predictions, strict=True)
        ],
        dtype=float,
    )
    bin_indices = np.minimum(np.digitize(result.confidences, edges[1:-1]), bins - 1)
    mean_confidences: list[float] = []
    observed_accuracies: list[float] = []
    counts: list[int] = []
    for bin_index in range(bins):
        mask = bin_indices == bin_index
        if not mask.any():
            continue
        mean_confidences.append(float(result.confidences[mask].mean()))
        observed_accuracies.append(float(correctness[mask].mean()))
        counts.append(int(mask.sum()))
    return mean_confidences, observed_accuracies, counts


def write_calibration_plot(
    results: Mapping[str, EvaluationResult],
    *,
    actual_labels: Sequence[str],
    bins: int,
    path: Path,
) -> dict[str, dict[str, list[float] | list[int]]]:
    """Write an aggregate top-label reliability plot for probability candidates."""
    figure, axis = plt.subplots(figsize=(8, 7))
    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="black", label="ideal")
    points: dict[str, dict[str, list[float] | list[int]]] = {}
    for model_name, result in results.items():
        confidence, accuracy, counts = calibration_curve_points(
            result,
            actual_labels=actual_labels,
            bins=bins,
        )
        points[model_name] = {
            "mean_confidence": confidence,
            "observed_accuracy": accuracy,
            "counts": counts,
        }
        axis.plot(confidence, accuracy, marker="o", linewidth=1.5, label=model_name)
    axis.set(
        title="Validation top-label calibration",
        xlabel="Mean prediction confidence",
        ylabel="Observed accuracy",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    axis.legend(fontsize="small")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".tmp-{uuid4().hex[:8]}")
    try:
        figure.savefig(temporary, format="png", dpi=140, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)
    return points


def update_model_leaderboard(path: Path, candidate_rows: Sequence[Mapping[str, object]]) -> None:
    """Preserve prior baselines and add the current candidate comparison."""
    existing: list[dict[str, str]] = []
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as input_file:
            existing = list(csv.DictReader(input_file))
    normalized: list[dict[str, object]] = [
        {column: row.get(column, "") for column in LEADERBOARD_COLUMNS} for row in existing
    ]
    new_keys = {(str(row["run_id"]), str(row["model_name"])) for row in candidate_rows}
    retained = [row for row in normalized if (row["run_id"], row["model_name"]) not in new_keys]
    combined: list[dict[str, object]] = [
        *retained,
        *(dict(row) for row in candidate_rows),
    ]
    combined.sort(
        key=lambda row: (
            -_as_float(row["macro_f1"]),
            str(row["model_name"]),
        )
    )
    for rank, row in enumerate(combined, start=1):
        row["rank"] = rank
        if not row.get("stage"):
            row["stage"] = "stage5_baseline"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".tmp-{uuid4().hex[:8]}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(
                output_file, fieldnames=LEADERBOARD_COLUMNS, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(combined)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _as_float(value: object) -> float:
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError(f"Expected numeric leaderboard value, received {type(value).__name__}.")
