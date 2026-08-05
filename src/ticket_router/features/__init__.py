"""Leakage-safe feature engineering boundary."""

from ticket_router.features.contracts import (
    ALLOWED_MODEL_INPUTS,
    FeatureLeakageError,
    find_forbidden_feature_columns,
    validate_model_feature_columns,
    validate_model_feature_frame,
)
from ticket_router.features.text import preprocess_model_text

__all__ = [
    "ALLOWED_MODEL_INPUTS",
    "FeatureLeakageError",
    "find_forbidden_feature_columns",
    "preprocess_model_text",
    "validate_model_feature_columns",
    "validate_model_feature_frame",
]
