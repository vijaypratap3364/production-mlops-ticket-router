"""Deterministic stratified group splitting with duplicate leakage prevention."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import polars as pl

from ticket_router.config import SplitRatios, SplittingSettings, TextPreprocessingSettings
from ticket_router.data.duplicates import (
    NORMALIZED_TEXT_HASH_COLUMN,
    add_duplicate_hashes,
)
from ticket_router.features.contracts import validate_model_feature_columns
from ticket_router.features.text import preprocess_model_text

SPLIT_NAMES = ("train", "validation", "test")
MODEL_READY_FEATURE_COLUMNS = ("model_text",)
PREPARED_COLUMNS = (
    "ticket_record_id",
    "source_row_id",
    "source_file",
    "source_row_number",
    "language",
    "subject",
    "body",
    "text",
    "model_text",
    NORMALIZED_TEXT_HASH_COLUMN,
    "queue",
)


class SplitError(ValueError):
    """Raised when grouped stratification or its invariants are infeasible."""


@dataclass(frozen=True)
class PreparedSplits:
    """Prepared split frames plus duplicate-policy evidence."""

    train: pl.DataFrame
    validation: pl.DataFrame
    test: pl.DataFrame
    selected_input_rows: int
    contradictory_group_count: int
    contradictory_rows_excluded: int

    def as_dict(self) -> dict[str, pl.DataFrame]:
        """Return split frames by stable public name."""
        return {"train": self.train, "validation": self.validation, "test": self.test}


@dataclass(frozen=True)
class SplitValidationSummary:
    """Aggregate invariant results for a successfully validated split."""

    total_records: int
    split_counts: dict[str, int]
    split_percentages: dict[str, float]
    per_class_counts: dict[str, dict[str, int]]
    maximum_class_proportion_deviation: dict[str, float]
    most_deviant_class: dict[str, str]


def create_prepared_splits(
    frame: pl.DataFrame,
    *,
    selected_classes: tuple[str, ...],
    preprocessing: TextPreprocessingSettings,
    splitting: SplittingSettings,
    split_ratios: SplitRatios,
    random_seed: int,
) -> PreparedSplits:
    """Clean text, exclude contradictions, and allocate exact groups once."""
    selected = frame.filter(pl.col("queue").is_in(selected_classes))
    enriched = add_duplicate_hashes(selected)
    contradictory_hashes = (
        enriched.group_by(NORMALIZED_TEXT_HASH_COLUMN)
        .agg(pl.col("queue").n_unique().alias("queue_count"))
        .filter(pl.col("queue_count") > 1)[NORMALIZED_TEXT_HASH_COLUMN]
        .to_list()
    )
    contradictory_rows = enriched.filter(
        pl.col(NORMALIZED_TEXT_HASH_COLUMN).is_in(contradictory_hashes)
    ).height
    if splitting.duplicate_policy != "exclude_contradictory_group_exact_duplicates":
        raise SplitError(f"Unsupported duplicate policy: {splitting.duplicate_policy}")
    usable = enriched.filter(
        ~pl.col(NORMALIZED_TEXT_HASH_COLUMN).is_in(contradictory_hashes)
    ).with_columns(
        pl.col("text")
        .map_elements(
            lambda value: preprocess_model_text(value, preprocessing),
            return_dtype=pl.String,
        )
        .alias("model_text")
    )
    if (
        usable["model_text"].null_count()
        or usable.filter(pl.col("model_text").str.strip_chars().str.len_chars() == 0).height
    ):
        raise SplitError("Preprocessing produced null or empty model_text records.")

    assignments = _group_assignments(
        usable,
        selected_classes=selected_classes,
        split_ratios=split_ratios,
        random_seed=random_seed,
    )
    assigned = usable.join(assignments, on=NORMALIZED_TEXT_HASH_COLUMN, how="inner")
    if assigned.height != usable.height:
        raise SplitError(
            f"Group assignment lost records: expected {usable.height}, received {assigned.height}."
        )

    frames = {
        split_name: (
            assigned.filter(pl.col("split") == split_name)
            .select(PREPARED_COLUMNS)
            .sort("ticket_record_id")
        )
        for split_name in SPLIT_NAMES
    }
    return PreparedSplits(
        train=frames["train"],
        validation=frames["validation"],
        test=frames["test"],
        selected_input_rows=selected.height,
        contradictory_group_count=len(contradictory_hashes),
        contradictory_rows_excluded=contradictory_rows,
    )


def validate_prepared_splits(
    prepared: PreparedSplits,
    *,
    selected_classes: tuple[str, ...],
    split_ratios: SplitRatios,
    splitting: SplittingSettings,
) -> SplitValidationSummary:
    """Verify disjointness, representation, ratios, and feature exposure."""
    frames = prepared.as_dict()
    _validate_pairwise_disjoint(frames, "ticket_record_id")
    _validate_pairwise_disjoint(frames, NORMALIZED_TEXT_HASH_COLUMN)
    validate_model_feature_columns(MODEL_READY_FEATURE_COLUMNS)

    expected_labels = set(selected_classes)
    total_records = sum(frame.height for frame in frames.values())
    if total_records == 0:
        raise SplitError("Prepared splits contain no records.")
    overall = pl.concat(list(frames.values()))
    overall_counts = _class_counts(overall, selected_classes)
    overall_proportions = {label: count / total_records for label, count in overall_counts.items()}
    ratio_by_split = _ratio_dict(split_ratios)
    split_counts: dict[str, int] = {}
    split_percentages: dict[str, float] = {}
    per_class_counts: dict[str, dict[str, int]] = {}
    maximum_deviations: dict[str, float] = {}
    most_deviant_classes: dict[str, str] = {}

    for split_name, split_frame in frames.items():
        labels = set(split_frame["queue"].unique().to_list())
        missing_labels = sorted(expected_labels - labels)
        if missing_labels:
            raise SplitError(
                f"Split '{split_name}' is missing selected classes: {', '.join(missing_labels)}"
            )
        count = split_frame.height
        percentage = count / total_records
        expected_percentage = ratio_by_split[split_name]
        if abs(percentage - expected_percentage) > splitting.split_size_tolerance:
            raise SplitError(
                f"Split '{split_name}' percentage {percentage:.6f} differs from target "
                f"{expected_percentage:.6f} by more than tolerance "
                f"{splitting.split_size_tolerance:.6f}."
            )
        class_counts = _class_counts(split_frame, selected_classes)
        deviations = {
            label: abs(class_counts[label] / count - overall_proportions[label])
            for label in selected_classes
        }
        maximum_deviation = max(deviations.values())
        difficult_label = max(deviations, key=deviations.__getitem__)
        if maximum_deviation > splitting.class_proportion_tolerance:
            raise SplitError(
                f"Split '{split_name}' class proportion for '{difficult_label}' deviates by "
                f"{maximum_deviation:.6f}, exceeding tolerance "
                f"{splitting.class_proportion_tolerance:.6f}."
            )
        split_counts[split_name] = count
        split_percentages[split_name] = percentage
        per_class_counts[split_name] = class_counts
        maximum_deviations[split_name] = maximum_deviation
        most_deviant_classes[split_name] = difficult_label

    return SplitValidationSummary(
        total_records=total_records,
        split_counts=split_counts,
        split_percentages=split_percentages,
        per_class_counts=per_class_counts,
        maximum_class_proportion_deviation=maximum_deviations,
        most_deviant_class=most_deviant_classes,
    )


def _group_assignments(
    frame: pl.DataFrame,
    *,
    selected_classes: tuple[str, ...],
    split_ratios: SplitRatios,
    random_seed: int,
) -> pl.DataFrame:
    assignments: list[dict[str, str]] = []
    ratios = _ratio_dict(split_ratios)
    for label in selected_classes:
        class_groups = (
            frame.filter(pl.col("queue") == label)
            .group_by(NORMALIZED_TEXT_HASH_COLUMN)
            .len()
            .rename({"len": "group_size"})
        )
        if class_groups.height < len(SPLIT_NAMES):
            raise SplitError(
                f"Class '{label}' has only {class_groups.height} exact-text groups; "
                f"at least {len(SPLIT_NAMES)} are required."
            )
        groups = [
            (str(row[NORMALIZED_TEXT_HASH_COLUMN]), int(row["group_size"]))
            for row in class_groups.iter_rows(named=True)
        ]
        groups.sort(
            key=lambda item: (
                -item[1],
                _seeded_order_key(random_seed, label, item[0]),
                item[0],
            )
        )
        class_total = sum(size for _, size in groups)
        targets = {name: class_total * ratios[name] for name in SPLIT_NAMES}
        allocated = dict.fromkeys(SPLIT_NAMES, 0)
        for group_hash, group_size in groups:
            split_name = max(
                SPLIT_NAMES,
                key=lambda name: (
                    (targets[name] - allocated[name]) / targets[name],
                    -SPLIT_NAMES.index(name),
                ),
            )
            assignments.append({NORMALIZED_TEXT_HASH_COLUMN: group_hash, "split": split_name})
            allocated[split_name] += group_size

    return pl.DataFrame(
        assignments,
        schema={NORMALIZED_TEXT_HASH_COLUMN: pl.String, "split": pl.String},
    ).sort(NORMALIZED_TEXT_HASH_COLUMN)


def _validate_pairwise_disjoint(frames: dict[str, pl.DataFrame], column: str) -> None:
    values = {name: set(frame[column].to_list()) for name, frame in frames.items()}
    for index, left_name in enumerate(SPLIT_NAMES):
        for right_name in SPLIT_NAMES[index + 1 :]:
            overlap = values[left_name] & values[right_name]
            if overlap:
                raise SplitError(
                    f"Splits '{left_name}' and '{right_name}' overlap on {column}: "
                    f"{len(overlap)} value(s)."
                )


def _class_counts(frame: pl.DataFrame, labels: tuple[str, ...]) -> dict[str, int]:
    observed = {
        str(row["queue"]): int(row["count"])
        for row in frame.group_by("queue").len().rename({"len": "count"}).iter_rows(named=True)
    }
    return {label: observed.get(label, 0) for label in labels}


def _ratio_dict(split_ratios: SplitRatios) -> dict[str, float]:
    return {
        "train": split_ratios.train,
        "validation": split_ratios.validation,
        "test": split_ratios.test,
    }


def _seeded_order_key(random_seed: int, label: str, group_hash: str) -> str:
    value = f"{random_seed}\x1f{label}\x1f{group_hash}"
    return sha256(value.encode("utf-8")).hexdigest()
