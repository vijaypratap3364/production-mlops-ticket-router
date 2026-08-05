"""Tests for conservative stateless ticket-text preprocessing."""

from __future__ import annotations

from ticket_router.config import TextPreprocessingSettings
from ticket_router.features.text import preprocess_model_text


def _settings(*, mask_phone_numbers: bool = True) -> TextPreprocessingSettings:
    return TextPreprocessingSettings(
        unicode_normalization="NFKC",
        mask_email_addresses=True,
        mask_urls=True,
        mask_phone_numbers=mask_phone_numbers,
        email_mask="<EMAIL>",
        url_mask="<URL>",
        phone_mask="<PHONE>",
    )


def test_preprocessing_normalizes_unicode_whitespace_and_masks_pii() -> None:
    value = (
        "\uff23\uff41\uff46\uff45\u0301\tcontact user@example.com, "
        "visit https://example.com/path. "
        "Call +1 (312) 555-0188!"
    )

    cleaned = preprocess_model_text(value, _settings())

    assert cleaned == "Café contact <EMAIL>, visit <URL>. Call <PHONE>!"


def test_preprocessing_preserves_useful_punctuation_and_words() -> None:
    value = "Don't remove: useful punctuation, or stop words!"

    assert preprocess_model_text(value, _settings()) == value


def test_phone_masking_is_optional() -> None:
    value = "Call 312-555-0188."

    assert preprocess_model_text(value, _settings(mask_phone_numbers=False)) == value
