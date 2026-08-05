"""Tests for deterministic, minimum-aware target class selection."""

from __future__ import annotations

import polars as pl
import pytest

from ticket_router.config import SplitRatios
from ticket_router.data.class_selection import (
    ClassSelectionError,
    ClassSelectionReport,
    select_target_classes,
)

HASH = "a" * 64
RATIOS = SplitRatios(train=0.70, validation=0.15, test=0.15)


def _frame(counts: dict[str, int]) -> pl.DataFrame:
    return pl.DataFrame({"queue": [label for label, count in counts.items() for _ in range(count)]})


def _select(
    frame: pl.DataFrame,
    *,
    top_k: int = 2,
    minimum: int = 7,
) -> ClassSelectionReport:
    return select_target_classes(
        frame,
        target_column="queue",
        number_of_classes=top_k,
        minimum_class_count=minimum,
        split_ratios=RATIOS,
        input_file_path="data/interim/normalized.parquet",
        input_file_sha256=HASH,
        configuration_hash=HASH,
    )


def test_class_selection_is_deterministic_with_label_tie_break() -> None:
    frame = _frame({"Zulu": 8, "Alpha": 8, "Beta": 9})
    first = _select(frame)
    second = _select(frame.reverse())

    assert first == second
    assert first.selected_classes == ("Beta", "Alpha")
    assert first.label_mapping == {"Beta": 0, "Alpha": 1}
    assert first.excluded_classes[0].reason.startswith("eligible but outside")


def test_minimum_class_size_is_enforced() -> None:
    frame = _frame({"Eligible": 8, "Too Small": 6})

    with pytest.raises(ClassSelectionError, match="Only 1 queue classes"):
        _select(frame, top_k=2, minimum=7)
