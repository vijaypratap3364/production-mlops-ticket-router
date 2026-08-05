"""Stateless, conservative preprocessing for sparse ticket-text models."""

from __future__ import annotations

import re
import unicodedata

from ticket_router.config import TextPreprocessingSettings

EMAIL_PATTERN = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
URL_PATTERN = re.compile(r"\b(?:https?://|www\.)[^\s<>{}\[\]]+", flags=re.IGNORECASE)
PHONE_CANDIDATE_PATTERN = re.compile(r"(?<!\w)\+?\d[\d().\s-]{5,}\d(?!\w)")
WHITESPACE_PATTERN = re.compile(r"\s+")
TRAILING_URL_PUNCTUATION = ".,!?;:"


def preprocess_model_text(value: str, settings: TextPreprocessingSettings) -> str:
    """Normalize and mask text without learning vocabulary or deleting punctuation."""
    cleaned = unicodedata.normalize(settings.unicode_normalization, value)
    if settings.mask_email_addresses:
        cleaned = EMAIL_PATTERN.sub(settings.email_mask, cleaned)
    if settings.mask_urls:
        cleaned = URL_PATTERN.sub(
            lambda match: _masked_url(match.group(0), settings.url_mask),
            cleaned,
        )
    if settings.mask_phone_numbers:
        cleaned = PHONE_CANDIDATE_PATTERN.sub(
            lambda match: _masked_phone(match.group(0), settings.phone_mask),
            cleaned,
        )
    return WHITESPACE_PATTERN.sub(" ", cleaned).strip()


def _masked_url(value: str, mask: str) -> str:
    stripped = value.rstrip(TRAILING_URL_PUNCTUATION)
    trailing = value[len(stripped) :]
    return mask + trailing


def _masked_phone(value: str, mask: str) -> str:
    digit_count = sum(character.isdigit() for character in value)
    return mask if digit_count >= 7 else value
