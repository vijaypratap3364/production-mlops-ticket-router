"""Tests for the normalized Pandera data contract."""

from __future__ import annotations

import polars as pl
import pytest

from ticket_router.config import AnalysisSettings
from ticket_router.data.validation import DataContractError, validate_normalized_frame


def test_valid_normalized_frame_passes_contract(
    valid_normalized_frame: pl.DataFrame,
    analysis_settings: AnalysisSettings,
) -> None:
    summary = validate_normalized_frame(
        valid_normalized_frame,
        settings=analysis_settings,
        language_filter="en",
    )

    assert summary.valid_records == 3
    assert summary.records_removed == 0


def test_contract_reports_invalid_record_id_without_ticket_content(
    valid_normalized_frame: pl.DataFrame,
    analysis_settings: AnalysisSettings,
) -> None:
    invalid = valid_normalized_frame.with_columns(
        pl.when(pl.col("source_row_number") == 1)
        .then(pl.lit("not-a-sha"))
        .otherwise(pl.col("ticket_record_id"))
        .alias("ticket_record_id")
    )

    with pytest.raises(DataContractError, match="ticket_record_id") as error:
        validate_normalized_frame(invalid, settings=analysis_settings, language_filter="en")

    assert "Login failure" not in str(error.value)


def test_contract_rejects_text_not_derived_from_subject_and_body(
    valid_normalized_frame: pl.DataFrame,
    analysis_settings: AnalysisSettings,
) -> None:
    invalid = valid_normalized_frame.with_columns(pl.lit("unrelated").alias("text"))

    with pytest.raises(DataContractError, match="derived exactly"):
        validate_normalized_frame(invalid, settings=analysis_settings, language_filter="en")


def test_contract_rejects_forbidden_extra_column(
    valid_normalized_frame: pl.DataFrame,
    analysis_settings: AnalysisSettings,
) -> None:
    invalid = valid_normalized_frame.with_columns(pl.lit("agent text").alias("answer"))

    with pytest.raises(DataContractError, match="unexpected/leakage-prone columns: answer"):
        validate_normalized_frame(invalid, settings=analysis_settings, language_filter="en")


def test_contract_rejects_duplicate_record_ids_and_non_english_rows(
    valid_normalized_frame: pl.DataFrame,
    analysis_settings: AnalysisSettings,
) -> None:
    duplicate_id = valid_normalized_frame["ticket_record_id"][0]
    invalid = valid_normalized_frame.with_columns(
        pl.when(pl.col("source_row_number") == 2)
        .then(pl.lit(duplicate_id))
        .otherwise(pl.col("ticket_record_id"))
        .alias("ticket_record_id"),
        pl.when(pl.col("source_row_number") == 3)
        .then(pl.lit("de"))
        .otherwise(pl.col("language"))
        .alias("language"),
    )

    with pytest.raises(DataContractError) as error:
        validate_normalized_frame(invalid, settings=analysis_settings, language_filter="en")

    message = str(error.value)
    assert "ticket_record_id" in message
    assert "language" in message
