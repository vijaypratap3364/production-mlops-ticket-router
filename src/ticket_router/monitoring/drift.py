"""Evidently-backed drift analysis over an explicit aggregate feature set."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import polars as pl

from ticket_router.monitoring.config import DriftSettings
from ticket_router.monitoring.features import (
    CATEGORICAL_OUTPUT_FEATURES,
    NUMERIC_INPUT_FEATURES,
    NUMERIC_OUTPUT_FEATURES,
)


@dataclass(frozen=True)
class ColumnDrift:
    column: str
    score: float
    threshold: float
    drifted: bool
    method: str


@dataclass(frozen=True)
class DriftResult:
    columns: tuple[ColumnDrift, ...]
    drifted_input_feature_count: int
    input_feature_count: int
    drifted_input_feature_share: float
    new_predicted_classes: tuple[str, ...]
    missing_predicted_classes: tuple[str, ...]
    reference_low_confidence_rate: float
    current_low_confidence_rate: float
    low_confidence_rate_change: float
    reference_mean_combined_length: float
    current_mean_combined_length: float
    combined_length_relative_change: float

    def column(self, name: str) -> ColumnDrift:
        return next(item for item in self.columns if item.column == name)

    def to_dict(self) -> dict[str, object]:
        return {
            "columns": [asdict(item) for item in self.columns],
            "drifted_input_feature_count": self.drifted_input_feature_count,
            "input_feature_count": self.input_feature_count,
            "drifted_input_feature_share": self.drifted_input_feature_share,
            "new_predicted_classes": list(self.new_predicted_classes),
            "missing_predicted_classes": list(self.missing_predicted_classes),
            "reference_low_confidence_rate": self.reference_low_confidence_rate,
            "current_low_confidence_rate": self.current_low_confidence_rate,
            "low_confidence_rate_change": self.low_confidence_rate_change,
            "reference_mean_combined_length": self.reference_mean_combined_length,
            "current_mean_combined_length": self.current_mean_combined_length,
            "combined_length_relative_change": self.combined_length_relative_change,
        }


def generate_drift_report(
    *,
    reference: pl.DataFrame,
    current: pl.DataFrame,
    settings: DriftSettings,
    html_path: Path,
    json_path: Path,
) -> DriftResult:
    """Run Evidently with declared numerical/categorical columns and persist its report."""
    _validate_columns(reference, current)
    Report, DataDefinition, Dataset, DataDriftPreset = _evidently_components()
    numerical = [*NUMERIC_INPUT_FEATURES, *NUMERIC_OUTPUT_FEATURES]
    categorical = list(CATEGORICAL_OUTPUT_FEATURES)
    definition = DataDefinition(
        numerical_columns=numerical,
        categorical_columns=categorical,
        unknown_columns=["model_version"],
    )
    reference_dataset = Dataset.from_pandas(
        reference.select(*numerical, *categorical, "model_version").to_pandas(), definition
    )
    current_dataset = Dataset.from_pandas(
        current.select(*numerical, *categorical, "model_version").to_pandas(), definition
    )
    per_column_threshold = {
        **{name: settings.numeric_feature_threshold for name in NUMERIC_INPUT_FEATURES},
        "prediction_confidence": settings.confidence_distribution_threshold,
        "predicted_queue": settings.prediction_distribution_threshold,
        "low_confidence": settings.low_confidence_distribution_threshold,
    }
    snapshot = Report(
        [
            DataDriftPreset(
                columns=[*numerical, *categorical],
                num_method=settings.numeric_method,
                cat_method=settings.categorical_method,
                per_column_threshold=per_column_threshold,
            )
        ],
        metadata={"feature_policy": "privacy-safe-derived-fields-only"},
    ).run(current_dataset, reference_dataset)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html_path))
    snapshot.save_json(str(json_path))
    columns = _parse_column_drift(snapshot.dict(), settings)
    input_results = [item for item in columns if item.column in NUMERIC_INPUT_FEATURES]
    reference_classes = set(reference["predicted_queue"].cast(pl.String).to_list())
    current_classes = set(current["predicted_queue"].cast(pl.String).to_list())
    reference_low_confidence_rate = float(
        cast(float, reference["low_confidence"].cast(pl.Float64).mean())
    )
    current_low_confidence_rate = float(
        cast(float, current["low_confidence"].cast(pl.Float64).mean())
    )
    reference_length = float(cast(float, reference["combined_length"].mean()))
    current_length = float(cast(float, current["combined_length"].mean()))
    return DriftResult(
        columns=columns,
        drifted_input_feature_count=sum(item.drifted for item in input_results),
        input_feature_count=len(input_results),
        drifted_input_feature_share=(
            sum(item.drifted for item in input_results) / len(input_results)
        ),
        new_predicted_classes=tuple(sorted(current_classes - reference_classes)),
        missing_predicted_classes=tuple(sorted(reference_classes - current_classes)),
        reference_low_confidence_rate=reference_low_confidence_rate,
        current_low_confidence_rate=current_low_confidence_rate,
        low_confidence_rate_change=current_low_confidence_rate - reference_low_confidence_rate,
        reference_mean_combined_length=reference_length,
        current_mean_combined_length=current_length,
        combined_length_relative_change=_relative_change(reference_length, current_length),
    )


def _evidently_components() -> tuple[Any, Any, Any, Any]:
    """Import Evidently, handling NLTK 3.10's project-local-interpreter false positive."""
    disable_was_set = "NLTK_DISABLE_IMPORT_SECURITY" in os.environ
    interpreter_inside_cwd = _is_relative_to(Path(sys.base_prefix), Path.cwd())
    if interpreter_inside_cwd and not disable_was_set:
        os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
    try:
        from evidently import DataDefinition, Dataset, Report  # type: ignore[import-untyped]
        from evidently.presets import DataDriftPreset  # type: ignore[import-untyped]
    finally:
        if interpreter_inside_cwd and not disable_was_set:
            os.environ.pop("NLTK_DISABLE_IMPORT_SECURITY", None)
    return Report, DataDefinition, Dataset, DataDriftPreset


def _parse_column_drift(
    report: dict[str, object], settings: DriftSettings
) -> tuple[ColumnDrift, ...]:
    raw_metrics = report.get("metrics")
    if not isinstance(raw_metrics, list):
        raise RuntimeError("Evidently report does not contain metrics")
    result: list[ColumnDrift] = []
    for raw_metric in raw_metrics:
        if not isinstance(raw_metric, dict):
            continue
        config = raw_metric.get("config")
        if not isinstance(config, dict) or config.get("type") != "evidently:metric_v2:ValueDrift":
            continue
        column = str(config["column"])
        threshold = float(cast(float, config["threshold"]))
        score = float(cast(float, raw_metric["value"]))
        method = (
            settings.numeric_method
            if column in {*NUMERIC_INPUT_FEATURES, *NUMERIC_OUTPUT_FEATURES}
            else settings.categorical_method
        )
        result.append(
            ColumnDrift(
                column=column,
                score=score,
                threshold=threshold,
                drifted=score >= threshold,
                method=method,
            )
        )
    expected = {*NUMERIC_INPUT_FEATURES, *NUMERIC_OUTPUT_FEATURES, *CATEGORICAL_OUTPUT_FEATURES}
    observed = {item.column for item in result}
    if observed != expected:
        raise RuntimeError(f"Evidently omitted configured columns: {sorted(expected - observed)}")
    return tuple(sorted(result, key=lambda item: item.column))


def _validate_columns(reference: pl.DataFrame, current: pl.DataFrame) -> None:
    required = {
        *NUMERIC_INPUT_FEATURES,
        *NUMERIC_OUTPUT_FEATURES,
        *CATEGORICAL_OUTPUT_FEATURES,
        "model_version",
    }
    for name, frame in (("reference", reference), ("current", current)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} monitoring data is missing columns: {', '.join(missing)}")
        if frame.is_empty():
            raise ValueError(f"{name} monitoring data cannot be empty")


def _relative_change(reference: float, current: float) -> float:
    if reference == 0.0:
        return 0.0 if current == 0.0 else 1.0
    return (current - reference) / reference


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True
