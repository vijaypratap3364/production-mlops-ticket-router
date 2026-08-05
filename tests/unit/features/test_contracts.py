"""Tests for the modeling feature allowlist and leakage guard."""

from __future__ import annotations

import pytest

from ticket_router.features.contracts import (
    ALLOWED_MODEL_INPUTS,
    FeatureLeakageError,
    find_forbidden_feature_columns,
    validate_model_feature_columns,
)


def test_allowed_model_inputs_are_explicit() -> None:
    assert {"subject", "body", "text", "model_text"} == ALLOWED_MODEL_INPUTS
    assert validate_model_feature_columns(["subject", "body", "text"]) == (
        "subject",
        "body",
        "text",
    )
    assert validate_model_feature_columns(["model_text"]) == ("model_text",)


@pytest.mark.parametrize(
    "forbidden_column",
    [
        "answer",
        "response",
        "queue",
        "priority",
        "type",
        "tags",
        "tag_1",
        "agent_notes",
        "resolution_text",
    ],
)
def test_leakage_prone_fields_are_rejected(forbidden_column: str) -> None:
    with pytest.raises(FeatureLeakageError, match="forbidden"):
        validate_model_feature_columns(["subject", forbidden_column])


def test_non_predictive_metadata_is_rejected() -> None:
    with pytest.raises(FeatureLeakageError, match="Only subject"):
        validate_model_feature_columns(["subject", "ticket_record_id"])


def test_raw_schema_leakage_audit_uses_the_same_contract() -> None:
    assert find_forbidden_feature_columns(
        ["subject", "body", "Answer", "queue", "tag_1", "agent_notes"]
    ) == ("agent_notes", "answer", "queue", "tag_1")
