"""Tests for deterministic stratified duplicate-group splitting."""

from __future__ import annotations

import polars as pl

from ticket_router.config import Settings
from ticket_router.data.normalize import combine_ticket_text
from ticket_router.data.splitting import (
    PreparedSplits,
    create_prepared_splits,
    validate_prepared_splits,
)


def synthetic_split_frame(*, include_contradiction: bool = False) -> pl.DataFrame:
    records: list[dict[str, object]] = []
    record_number = 0
    for class_index, label in enumerate(("Queue A", "Queue B", "Queue C")):
        for group_index in range(30):
            repeats = 2 if group_index % 5 == 0 else 1
            body = f"Synthetic body for class {class_index} group {group_index}."
            for _ in range(repeats):
                record_number += 1
                subject = f"Request {class_index}-{group_index}"
                records.append(
                    {
                        "ticket_record_id": f"{record_number:064x}",
                        "source_row_id": f"fixture.csv:{record_number}",
                        "source_file": "fixture.csv",
                        "source_row_number": record_number,
                        "language": "en",
                        "subject": subject,
                        "body": body,
                        "text": combine_ticket_text(subject, body),
                        "queue": label,
                    }
                )
    if include_contradiction:
        contradictory_text = combine_ticket_text("Conflict", "Same exact content")
        for label in ("Queue A", "Queue B"):
            record_number += 1
            records.append(
                {
                    "ticket_record_id": f"{record_number:064x}",
                    "source_row_id": f"fixture.csv:{record_number}",
                    "source_file": "fixture.csv",
                    "source_row_number": record_number,
                    "language": "en",
                    "subject": "Conflict",
                    "body": "Same exact content",
                    "text": contradictory_text,
                    "queue": label,
                }
            )
    return pl.DataFrame(records)


def _create(frame: pl.DataFrame) -> PreparedSplits:
    settings = Settings.load(env_file=None).project_config
    relaxed_splitting = settings.splitting.model_copy(
        update={
            "class_proportion_tolerance": 0.10,
            "split_size_tolerance": 0.10,
        }
    )
    return create_prepared_splits(
        frame,
        selected_classes=("Queue A", "Queue B", "Queue C"),
        preprocessing=settings.preprocessing,
        splitting=relaxed_splitting,
        split_ratios=settings.split_ratios,
        random_seed=42,
    )


def test_split_is_deterministic_and_duplicate_groups_do_not_leak() -> None:
    settings = Settings.load(env_file=None).project_config
    splitting = settings.splitting.model_copy(
        update={
            "class_proportion_tolerance": 0.10,
            "split_size_tolerance": 0.10,
        }
    )
    first = create_prepared_splits(
        synthetic_split_frame(),
        selected_classes=("Queue A", "Queue B", "Queue C"),
        preprocessing=settings.preprocessing,
        splitting=splitting,
        split_ratios=settings.split_ratios,
        random_seed=42,
    )
    second = create_prepared_splits(
        synthetic_split_frame().reverse(),
        selected_classes=("Queue A", "Queue B", "Queue C"),
        preprocessing=settings.preprocessing,
        splitting=splitting,
        split_ratios=settings.split_ratios,
        random_seed=42,
    )

    assert {
        name: frame["ticket_record_id"].to_list() for name, frame in first.as_dict().items()
    } == {name: frame["ticket_record_id"].to_list() for name, frame in second.as_dict().items()}
    summary = validate_prepared_splits(
        first,
        selected_classes=("Queue A", "Queue B", "Queue C"),
        split_ratios=settings.split_ratios,
        splitting=splitting,
    )
    assert summary.total_records == synthetic_split_frame().height
    for split_frame in first.as_dict().values():
        assert split_frame["queue"].n_unique() == 3
    split_frames = first.as_dict()
    for left_name, right_name in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        left = split_frames[left_name]
        right = split_frames[right_name]
        assert set(left["ticket_record_id"]).isdisjoint(right["ticket_record_id"])
        assert set(left["normalized_text_hash"]).isdisjoint(right["normalized_text_hash"])


def test_contradictory_exact_group_is_excluded() -> None:
    prepared = _create(synthetic_split_frame(include_contradiction=True))

    assert prepared.contradictory_group_count == 1
    assert prepared.contradictory_rows_excluded == 2
    assert sum(frame.height for frame in prepared.as_dict().values()) == 108
