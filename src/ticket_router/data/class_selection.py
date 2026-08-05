"""Deterministic, configuration-driven target queue selection."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Self

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from ticket_router.config import SplitRatios
from ticket_router.data.manifests import atomic_write_json


class ClassSelectionError(ValueError):
    """Raised when the configured class-selection policy is infeasible."""


class ExcludedClass(BaseModel):
    """A queue omitted by the deterministic policy and its reason."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    count: int = Field(ge=0)
    reason: str


class ClassSelectionReport(BaseModel):
    """Versioned label-space definition derived only from observed counts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_file_path: str
    input_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_class_count: int = Field(gt=1)
    configured_minimum_class_count: int = Field(gt=0)
    minimum_count_for_stratified_split: int = Field(gt=0)
    effective_minimum_class_count: int = Field(gt=0)
    original_class_counts: dict[str, int]
    selected_classes: tuple[str, ...]
    label_mapping: dict[str, int]
    selected_class_counts: dict[str, int]
    excluded_classes: tuple[ExcludedClass, ...]
    final_row_count: int = Field(ge=0)
    class_proportions: dict[str, float]
    imbalance_ratio: float = Field(ge=1.0)

    @classmethod
    def read(cls, path: Path) -> Self:
        """Load a generated class-selection report."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def write(self, path: Path) -> None:
        """Write the deterministic report atomically."""
        atomic_write_json(path, self.model_dump(mode="json"))


def select_target_classes(
    frame: pl.DataFrame,
    *,
    target_column: str,
    number_of_classes: int,
    minimum_class_count: int,
    split_ratios: SplitRatios,
    input_file_path: str,
    input_file_sha256: str,
    configuration_hash: str,
) -> ClassSelectionReport:
    """Rank eligible labels by count with a stable label tie-break."""
    if target_column not in frame.columns:
        raise ClassSelectionError(f"Target column is missing: {target_column}")

    counts_frame = frame.group_by(target_column).len().rename({"len": "count"})
    if counts_frame[target_column].null_count() > 0:
        raise ClassSelectionError(f"Target column contains null labels: {target_column}")

    counts = {
        str(row[target_column]): int(row["count"]) for row in counts_frame.iter_rows(named=True)
    }
    original_counts = dict(sorted(counts.items(), key=lambda item: item[0].casefold()))
    split_minimum = minimum_count_for_stratified_split(split_ratios)
    effective_minimum = max(minimum_class_count, split_minimum)
    eligible = [(label, count) for label, count in counts.items() if count >= effective_minimum]
    eligible.sort(key=lambda item: (-item[1], item[0].casefold(), item[0]))
    if len(eligible) < number_of_classes:
        raise ClassSelectionError(
            f"Only {len(eligible)} queue classes meet effective minimum count "
            f"{effective_minimum}; {number_of_classes} are required."
        )

    selected_pairs = eligible[:number_of_classes]
    selected_labels = tuple(label for label, _ in selected_pairs)
    selected_counts = {label: count for label, count in selected_pairs}
    final_row_count = sum(selected_counts.values())
    proportions = {label: count / final_row_count for label, count in selected_pairs}
    excluded = tuple(
        ExcludedClass(
            label=label,
            count=count,
            reason=_exclusion_reason(
                label=label,
                count=count,
                selected_labels=frozenset(selected_labels),
                minimum_class_count=minimum_class_count,
                split_minimum=split_minimum,
            ),
        )
        for label, count in sorted(counts.items(), key=lambda item: item[0].casefold())
        if label not in selected_labels
    )
    smallest_count = min(selected_counts.values())
    largest_count = max(selected_counts.values())

    return ClassSelectionReport(
        input_file_path=input_file_path,
        input_file_sha256=input_file_sha256,
        configuration_hash=configuration_hash,
        requested_class_count=number_of_classes,
        configured_minimum_class_count=minimum_class_count,
        minimum_count_for_stratified_split=split_minimum,
        effective_minimum_class_count=effective_minimum,
        original_class_counts=original_counts,
        selected_classes=selected_labels,
        label_mapping={label: index for index, label in enumerate(selected_labels)},
        selected_class_counts=selected_counts,
        excluded_classes=excluded,
        final_row_count=final_row_count,
        class_proportions=proportions,
        imbalance_ratio=largest_count / smallest_count,
    )


def minimum_count_for_stratified_split(split_ratios: SplitRatios) -> int:
    """Require an expected support of at least one record in every split."""
    smallest_ratio = min(split_ratios.train, split_ratios.validation, split_ratios.test)
    return math.ceil(1.0 / smallest_ratio)


def _exclusion_reason(
    *,
    label: str,
    count: int,
    selected_labels: frozenset[str],
    minimum_class_count: int,
    split_minimum: int,
) -> str:
    if label in selected_labels:
        raise ValueError("Selected labels cannot have an exclusion reason.")
    if count < minimum_class_count:
        return f"below configured minimum_class_count={minimum_class_count}"
    if count < split_minimum:
        return f"insufficient support for configured split ratios; requires {split_minimum}"
    return "eligible but outside deterministic top-k by count and label tie-break"
