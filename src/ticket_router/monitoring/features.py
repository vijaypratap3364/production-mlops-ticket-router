"""Privacy-safe, stateless feature definitions for text-input monitoring."""

from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass

from ticket_router.features.text import EMAIL_PATTERN, URL_PATTERN


@dataclass(frozen=True)
class TextMonitoringFeatures:
    subject_length: int
    body_length: int
    combined_length: int
    word_count: int
    uppercase_ratio: float
    digit_ratio: float
    punctuation_ratio: float
    url_count: int
    email_marker_count: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


FEATURE_DEFINITIONS = {
    "subject_length": "Unicode code-point count of the submitted subject.",
    "body_length": "Unicode code-point count of the submitted body.",
    "combined_length": "Sum of subject_length and body_length; field markers are excluded.",
    "word_count": "Whitespace-delimited token count after configured model-text masking.",
    "uppercase_ratio": "Uppercase letters divided by all cased letters in subject and body.",
    "digit_ratio": "Digit characters divided by non-whitespace characters in subject and body.",
    "punctuation_ratio": (
        "Unicode punctuation characters divided by non-whitespace characters in subject and body."
    ),
    "url_count": "Count of URL-pattern matches before model-text URL masking.",
    "email_marker_count": "Count of email-pattern matches before model-text email masking.",
    "predicted_queue": "Champion model top-1 queue; no source target is used as an input.",
    "prediction_confidence": "Champion calibrated probability for the top-1 prediction.",
    "low_confidence": "Whether confidence is below the configured API review threshold.",
    "model_version": "Immutable MLflow model version that produced the prediction.",
}

NUMERIC_INPUT_FEATURES = (
    "subject_length",
    "body_length",
    "combined_length",
    "word_count",
    "uppercase_ratio",
    "digit_ratio",
    "punctuation_ratio",
    "url_count",
    "email_marker_count",
)
NUMERIC_OUTPUT_FEATURES = ("prediction_confidence",)
CATEGORICAL_OUTPUT_FEATURES = ("predicted_queue", "low_confidence")
MONITORING_COLUMNS = (
    *NUMERIC_INPUT_FEATURES,
    *NUMERIC_OUTPUT_FEATURES,
    *CATEGORICAL_OUTPUT_FEATURES,
    "model_version",
)


def derive_text_monitoring_features(
    *,
    subject: str,
    body: str,
    model_text: str,
) -> TextMonitoringFeatures:
    """Derive aggregate features without retaining or returning the input text."""
    combined = f"{subject}\n{body}"
    cased_count = sum(
        character.isalpha() and character.lower() != character.upper() for character in combined
    )
    visible_count = sum(not character.isspace() for character in combined)
    uppercase_count = sum(character.isupper() for character in combined)
    digit_count = sum(character.isdigit() for character in combined)
    punctuation_count = sum(
        unicodedata.category(character).startswith("P") for character in combined
    )
    return TextMonitoringFeatures(
        subject_length=len(subject),
        body_length=len(body),
        combined_length=len(subject) + len(body),
        word_count=len(model_text.split()),
        uppercase_ratio=_safe_ratio(uppercase_count, cased_count),
        digit_ratio=_safe_ratio(digit_count, visible_count),
        punctuation_ratio=_safe_ratio(punctuation_count, visible_count),
        url_count=len(URL_PATTERN.findall(combined)),
        email_marker_count=len(EMAIL_PATTERN.findall(combined)),
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0
