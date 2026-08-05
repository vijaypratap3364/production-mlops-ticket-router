"""Typed validation contract for normalized ticket records."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd  # type: ignore[import-untyped]
import pandera.pandas as pa
import polars as pl
from pandera.errors import SchemaErrors

from ticket_router.config import AnalysisSettings
from ticket_router.features.contracts import validate_model_feature_frame

EXPECTED_NORMALIZED_COLUMNS = (
    "ticket_record_id",
    "source_row_id",
    "source_file",
    "source_row_number",
    "language",
    "subject",
    "body",
    "text",
    "queue",
)
MODEL_FEATURE_COLUMNS = ("subject", "body", "text")
SHA256_PATTERN = r"^[0-9a-f]{64}$"


class DataContractError(ValueError):
    """Raised when normalized data violates an actionable contract rule."""


@dataclass(frozen=True)
class ValidationSummary:
    """Privacy-safe result of successful contract validation."""

    total_records: int
    valid_records: int
    records_removed: int
    validated_columns: tuple[str, ...]


def validate_normalized_frame(
    frame: pl.DataFrame,
    *,
    settings: AnalysisSettings,
    language_filter: str,
) -> ValidationSummary:
    """Validate normalized records and their physically separated feature frame."""
    _validate_columns(frame.columns)
    validate_model_feature_frame(frame.select(MODEL_FEATURE_COLUMNS))

    schema = _normalized_schema(settings=settings, language_filter=language_filter)
    pandas_frame = frame.to_pandas()
    try:
        schema.validate(pandas_frame, lazy=True)
    except SchemaErrors as exc:
        raise DataContractError(_actionable_schema_error(exc)) from exc

    return ValidationSummary(
        total_records=frame.height,
        valid_records=frame.height,
        records_removed=0,
        validated_columns=EXPECTED_NORMALIZED_COLUMNS,
    )


def _validate_columns(columns: list[str]) -> None:
    expected = set(EXPECTED_NORMALIZED_COLUMNS)
    actual = set(columns)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing columns: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected/leakage-prone columns: " + ", ".join(unexpected))
        raise DataContractError("Normalized dataset column contract failed; " + "; ".join(details))
    if tuple(columns) != EXPECTED_NORMALIZED_COLUMNS:
        raise DataContractError(
            "Normalized dataset columns are out of canonical order; expected: "
            + ", ".join(EXPECTED_NORMALIZED_COLUMNS)
        )


def _normalized_schema(
    *,
    settings: AnalysisSettings,
    language_filter: str,
) -> pa.DataFrameSchema:
    nonempty = pa.Check(
        lambda values: values.str.strip().str.len().gt(0),
        error="must be non-empty after trimming",
    )
    return pa.DataFrameSchema(
        {
            "ticket_record_id": pa.Column(
                str,
                checks=pa.Check.str_matches(SHA256_PATTERN),
                nullable=False,
                unique=True,
            ),
            "source_row_id": pa.Column(str, checks=nonempty, nullable=False),
            "source_file": pa.Column(str, checks=nonempty, nullable=False),
            "source_row_number": pa.Column(int, checks=pa.Check.ge(1), nullable=False),
            "language": pa.Column(
                str,
                checks=pa.Check.equal_to(language_filter.casefold()),
                nullable=False,
            ),
            "subject": pa.Column(
                str,
                checks=pa.Check(
                    lambda values: values.str.len().le(settings.max_subject_characters),
                    error=(
                        "subject exceeds configured max_subject_characters="
                        f"{settings.max_subject_characters}"
                    ),
                ),
                nullable=True,
            ),
            "body": pa.Column(
                str,
                checks=pa.Check(
                    lambda values: values.str.len().le(settings.max_body_characters),
                    error=(
                        "body exceeds configured max_body_characters="
                        f"{settings.max_body_characters}"
                    ),
                ),
                nullable=True,
            ),
            "text": pa.Column(
                str,
                checks=[
                    nonempty,
                    pa.Check(
                        lambda values: values.str.len().le(settings.max_text_characters),
                        error=(
                            "text exceeds configured max_text_characters="
                            f"{settings.max_text_characters}"
                        ),
                    ),
                ],
                nullable=False,
            ),
            "queue": pa.Column(str, checks=nonempty, nullable=False),
        },
        checks=[
            pa.Check(_has_usable_source_text, error="subject and body cannot both be empty"),
            pa.Check(
                _combined_text_matches_sources,
                error="text must be derived exactly from subject and body",
            ),
        ],
        strict=True,
        ordered=True,
        coerce=False,
    )


def _has_usable_source_text(frame: pd.DataFrame) -> pd.Series:
    subject = frame["subject"].fillna("").astype(str).str.strip()
    body = frame["body"].fillna("").astype(str).str.strip()
    return subject.str.len().gt(0) | body.str.len().gt(0)


def _combined_text_matches_sources(frame: pd.DataFrame) -> pd.Series:
    subject = frame["subject"].fillna("").astype(str)
    body = frame["body"].fillna("").astype(str)
    expected = ("[SUBJECT] " + subject + "\n[BODY] " + body).str.strip()
    return frame["text"].eq(expected)


def _actionable_schema_error(exc: SchemaErrors) -> str:
    failure_cases = exc.failure_cases
    summaries: list[str] = []
    group_columns = ["schema_context", "column", "check"]
    if all(column in failure_cases.columns for column in group_columns):
        grouped = (
            failure_cases.groupby(group_columns, dropna=False).size().sort_values(ascending=False)
        )
        for keys, count in grouped.head(10).items():
            context, column, check = keys
            summaries.append(f"{context}.{column}: {check} ({count} failure(s))")
    if not summaries:
        summaries.append("schema validation failed; inspect the contract and source manifest")
    return "Normalized data contract failed: " + "; ".join(summaries)
