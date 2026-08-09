"""Pure dashboard input validation and downloadable batch-result shaping."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import pandas as pd  # type: ignore[import-untyped]

from ticket_router.api.schemas import BatchPredictionResponse, TicketRequest

REQUIRED_CSV_COLUMNS = ("subject", "body")
MAXIMUM_UPLOAD_BYTES = 5 * 1024 * 1024


class DashboardValidationError(ValueError):
    """Actionable upload validation failure safe to display in Streamlit."""


@dataclass(frozen=True)
class ValidatedBatch:
    frame: pd.DataFrame
    items: tuple[TicketRequest, ...]
    ignored_columns: tuple[str, ...]


def parse_ticket_csv(
    contents: bytes,
    *,
    maximum_batch_size: int,
    maximum_subject_characters: int = 2000,
    maximum_body_characters: int = 20000,
) -> ValidatedBatch:
    """Parse a bounded CSV and expose only subject/body to the API client."""
    if not contents:
        raise DashboardValidationError("The uploaded CSV is empty.")
    if len(contents) > MAXIMUM_UPLOAD_BYTES:
        raise DashboardValidationError("The uploaded CSV exceeds the 5 MiB dashboard limit.")
    try:
        frame = pd.read_csv(BytesIO(contents), dtype="string")
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DashboardValidationError("The upload is not a valid UTF-8 CSV file.") from exc
    missing = [column for column in REQUIRED_CSV_COLUMNS if column not in frame.columns]
    if missing:
        raise DashboardValidationError(f"CSV is missing required columns: {', '.join(missing)}.")
    if frame.empty:
        raise DashboardValidationError("The CSV must contain at least one ticket row.")
    if len(frame.index) > maximum_batch_size:
        raise DashboardValidationError(
            f"The CSV contains {len(frame.index)} rows; the API limit is {maximum_batch_size}."
        )
    selected = frame.loc[:, list(REQUIRED_CSV_COLUMNS)].fillna("").astype(str)
    selected = selected.apply(lambda column: column.str.strip())
    subject_lengths = selected["subject"].str.len()
    body_lengths = selected["body"].str.len()
    oversized_subject = subject_lengths > maximum_subject_characters
    oversized_body = body_lengths > maximum_body_characters
    empty_rows = (subject_lengths == 0) & (body_lengths == 0)
    if oversized_subject.any():
        row = int(oversized_subject[oversized_subject].index[0]) + 2
        raise DashboardValidationError(f"CSV row {row} exceeds the subject length limit.")
    if oversized_body.any():
        row = int(oversized_body[oversized_body].index[0]) + 2
        raise DashboardValidationError(f"CSV row {row} exceeds the body length limit.")
    if empty_rows.any():
        row = int(empty_rows[empty_rows].index[0]) + 2
        raise DashboardValidationError(f"CSV row {row} has no usable subject or body text.")
    items = tuple(
        TicketRequest(subject=row.subject, body=row.body)
        for row in selected.itertuples(index=False)
    )
    ignored = tuple(
        sorted(str(column) for column in frame.columns if column not in REQUIRED_CSV_COLUMNS)
    )
    return ValidatedBatch(frame=selected, items=items, ignored_columns=ignored)


def batch_results_frame(response: BatchPredictionResponse) -> pd.DataFrame:
    """Return privacy-safe result rows without copying uploaded ticket text."""
    rows: list[dict[str, object]] = []
    for row_number, prediction in enumerate(response.predictions, start=1):
        row: dict[str, object] = {
            "row_number": row_number,
            "request_id": prediction.request_id,
            "predicted_queue": prediction.predicted_queue,
            "confidence": prediction.confidence,
            "low_confidence": prediction.warning is not None,
            "model_name": prediction.model_name,
            "model_version": prediction.model_version,
            "prediction_timestamp": prediction.prediction_timestamp.isoformat(),
        }
        for rank, candidate in enumerate(prediction.top_k[:3], start=1):
            row[f"top_{rank}_queue"] = candidate.queue
            row[f"top_{rank}_confidence"] = candidate.confidence
        rows.append(row)
    return pd.DataFrame(rows)
