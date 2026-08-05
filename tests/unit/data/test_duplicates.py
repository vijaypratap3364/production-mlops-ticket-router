"""Tests for exact, contradictory, near-empty, and template duplicate detection."""

from __future__ import annotations

import polars as pl

from ticket_router.data.duplicates import (
    analyze_duplicates,
    contradictory_duplicate_groups,
    normalized_text_hash,
)


def _duplicate_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticket_record_id": [f"{index:064x}" for index in range(1, 7)],
            "queue": ["A", "A", "A", "B", "A", "A"],
            "text": [
                "Cannot open order 100",
                " cannot   OPEN order 100 ",
                "Cannot open order 200",
                "Cannot open order 200",
                "Help",
                "Cannot open order 300",
            ],
        }
    )


def test_normalized_text_hash_is_deterministic() -> None:
    assert normalized_text_hash(" A\nB ") == normalized_text_hash("a b")
    assert len(normalized_text_hash("a b")) == 64


def test_duplicate_analysis_detects_exact_and_template_groups() -> None:
    result = analyze_duplicates(
        _duplicate_frame(),
        near_empty_word_threshold=1,
        template_min_group_size=3,
    )

    assert result.exact_duplicate_groups == 2
    assert result.exact_duplicate_rows == 4
    assert result.exact_duplicate_redundant_records == 2
    assert result.near_empty_records == 1
    assert result.possible_template_duplicate_groups == 1
    assert result.possible_template_duplicate_rows == 5


def test_contradictory_labels_are_reported_by_hash_only() -> None:
    contradictions = contradictory_duplicate_groups(_duplicate_frame())

    assert contradictions.height == 1
    assert contradictions["record_count"].to_list() == [2]
    assert contradictions["queues"].to_list() == [["A", "B"]]
    assert "text" not in contradictions.columns
