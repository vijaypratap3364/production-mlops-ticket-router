"""Tests for serialized candidate-search and guardrail configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ticket_router.modeling.experiment_config import (
    CandidateExperimentSettings,
    experiment_configuration_hash,
)


def test_experiment_configuration_serialization_and_hash_are_deterministic(
    small_experiment_config: CandidateExperimentSettings,
) -> None:
    serialized = small_experiment_config.model_dump_json()
    restored = CandidateExperimentSettings.model_validate_json(serialized)

    assert restored == small_experiment_config
    assert experiment_configuration_hash(restored, random_seed=42) == (
        experiment_configuration_hash(small_experiment_config, random_seed=42)
    )


def test_experiment_configuration_rejects_unbounded_search(
    small_experiment_config: CandidateExperimentSettings,
) -> None:
    raw = small_experiment_config.model_dump(mode="json")
    raw["search"]["iterations_per_candidate"] = 11

    with pytest.raises(ValidationError, match="iterations_per_candidate"):
        CandidateExperimentSettings.model_validate(raw)
