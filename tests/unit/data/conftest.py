"""Shared synthetic records for Stage 3 data tests."""

from __future__ import annotations

import polars as pl
import pytest

from ticket_router.config import AnalysisSettings
from ticket_router.data.normalize import combine_ticket_text


@pytest.fixture
def analysis_settings() -> AnalysisSettings:
    return AnalysisSettings(
        max_subject_characters=100,
        max_body_characters=500,
        max_text_characters=650,
        near_empty_word_threshold=2,
        template_min_group_size=2,
        token_min_document_frequency=2,
        common_tokens_per_class=5,
    )


@pytest.fixture
def valid_normalized_frame() -> pl.DataFrame:
    records: list[dict[str, object]] = []
    values = [
        ("Login failure", "The account login fails.", "Technical Support"),
        (None, "Please provide the current invoice.", "Billing"),
        ("Return", "The unopened item should be returned.", "Returns"),
    ]
    for index, (subject, body, queue) in enumerate(values, start=1):
        records.append(
            {
                "ticket_record_id": f"{index:064x}",
                "source_row_id": f"fixture.csv:{index}",
                "source_file": "fixture.csv",
                "source_row_number": index,
                "language": "en",
                "subject": subject,
                "body": body,
                "text": combine_ticket_text(subject, body),
                "queue": queue,
            }
        )
    return pl.DataFrame(records)
