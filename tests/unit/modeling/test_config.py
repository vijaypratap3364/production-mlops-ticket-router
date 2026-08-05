"""Tests for typed baseline experiment configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ticket_router.modeling.config import BaselineSettings, baseline_configuration_hash


def test_baseline_configuration_hash_is_deterministic(
    small_baseline_config: BaselineSettings,
) -> None:
    first = baseline_configuration_hash(small_baseline_config, random_seed=42)
    second = baseline_configuration_hash(small_baseline_config, random_seed=42)

    assert first == second
    assert len(first) == 64
    assert first != baseline_configuration_hash(small_baseline_config, random_seed=43)


def test_baseline_configuration_rejects_out_of_scope_ngrams(
    small_baseline_config: BaselineSettings,
) -> None:
    raw = small_baseline_config.model_dump(mode="json")
    raw["tfidf"]["ngram_range"] = [2, 3]

    with pytest.raises(ValidationError, match="ngram_range"):
        BaselineSettings.model_validate(raw)
