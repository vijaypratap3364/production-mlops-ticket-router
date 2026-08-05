"""Feature allowlist and leakage guards for every modeling entry point."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Protocol

ALLOWED_MODEL_INPUTS = frozenset({"subject", "body", "text"})
FORBIDDEN_MODEL_INPUTS = frozenset(
    {
        "answer",
        "response",
        "queue",
        "priority",
        "type",
        "tag",
        "tags",
        "agent_generated",
        "resolution",
        "resolution_code",
        "resolution_text",
    }
)
FORBIDDEN_MODEL_INPUT_PREFIXES = (
    "answer_",
    "response_",
    "tag_",
    "agent_",
    "assigned_",
    "resolution_",
)


class FeatureLeakageError(ValueError):
    """Raised when non-allowlisted or leakage-prone fields reach modeling."""


class FeatureFrame(Protocol):
    """Minimal dataframe interface required by the feature guard."""

    @property
    def columns(self) -> Iterable[object]:
        """Return dataframe column labels."""


def validate_model_feature_columns(columns: Iterable[object]) -> tuple[str, ...]:
    """Validate and return normalized modeling feature names."""
    normalized_columns = tuple(_normalize_feature_name(str(column)) for column in columns)
    if not normalized_columns:
        raise FeatureLeakageError("Model feature dataframe must contain at least one column.")
    if len(set(normalized_columns)) != len(normalized_columns):
        raise FeatureLeakageError("Model feature columns collide after normalization.")

    forbidden = sorted(column for column in normalized_columns if _is_forbidden(column))
    if forbidden:
        raise FeatureLeakageError(
            "Leakage-prone columns are forbidden from modeling: " + ", ".join(forbidden)
        )

    unexpected = sorted(set(normalized_columns) - ALLOWED_MODEL_INPUTS)
    if unexpected:
        raise FeatureLeakageError(
            "Only subject, body, and text derived from those fields are allowed; received: "
            + ", ".join(unexpected)
        )
    return normalized_columns


def validate_model_feature_frame(frame: FeatureFrame) -> None:
    """Reject a dataframe containing identifiers, target, metadata, or leakage fields."""
    validate_model_feature_columns(frame.columns)


def find_forbidden_feature_columns(columns: Iterable[object]) -> tuple[str, ...]:
    """Return normalized leakage-prone names for audits of source schemas."""
    normalized = {_normalize_feature_name(str(column)) for column in columns}
    return tuple(sorted(column for column in normalized if _is_forbidden(column)))


def _normalize_feature_name(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", value.strip()).strip("_").casefold()


def _is_forbidden(column: str) -> bool:
    return column in FORBIDDEN_MODEL_INPUTS or column.startswith(FORBIDDEN_MODEL_INPUT_PREFIXES)
