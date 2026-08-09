"""CSV schema, batch bounds, and privacy-safe export tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ticket_router.api.schemas import (
    BatchPredictionResponse,
    ClassProbability,
    PredictionResponse,
)
from ticket_router.dashboard.validation import (
    DashboardValidationError,
    batch_results_frame,
    parse_ticket_csv,
)


def test_valid_csv_selects_only_approved_columns() -> None:
    contents = b"subject,body,priority\nInvoice,Please review,high\n,Network unavailable,urgent\n"

    result = parse_ticket_csv(contents, maximum_batch_size=10)

    assert len(result.items) == 2
    assert result.items[1].body == "Network unavailable"
    assert result.ignored_columns == ("priority",)
    assert list(result.frame.columns) == ["subject", "body"]


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (b"subject\nInvoice\n", "missing required columns"),
        (b"subject,body\n,\n", "no usable subject or body"),
        (b"subject,body\na,b\nc,d\n", "API limit is 1"),
    ],
)
def test_invalid_csv_is_rejected(contents: bytes, message: str) -> None:
    with pytest.raises(DashboardValidationError, match=message):
        parse_ticket_csv(contents, maximum_batch_size=1)


def test_csv_field_length_is_enforced() -> None:
    with pytest.raises(DashboardValidationError, match="subject length"):
        parse_ticket_csv(
            b"subject,body\ntoo-long,ok\n",
            maximum_batch_size=10,
            maximum_subject_characters=4,
        )


def test_downloadable_results_do_not_copy_ticket_text() -> None:
    response = BatchPredictionResponse(
        predictions=[
            PredictionResponse(
                request_id="request-1",
                predicted_queue="Billing",
                confidence=0.8,
                top_k=[ClassProbability(queue="Billing", confidence=0.8)],
                model_name="fixture",
                model_version="7",
                prediction_timestamp=datetime(2026, 8, 8, tzinfo=UTC),
            )
        ]
    )

    frame = batch_results_frame(response)

    assert "subject" not in frame.columns
    assert "body" not in frame.columns
    assert frame.loc[0, "predicted_queue"] == "Billing"
