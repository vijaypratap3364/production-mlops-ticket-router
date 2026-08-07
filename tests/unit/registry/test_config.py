"""Tests for the frozen final-model configuration."""

from ticket_router.registry.config import FinalModelConfig, final_model_configuration_hash


def test_final_model_configuration_is_frozen_and_hashable() -> None:
    config = FinalModelConfig.load()

    assert config.selected_candidate.name == "word_tfidf_calibrated_linear_svc"
    assert config.selected_candidate.word_ngram_range == (1, 2)
    assert config.selected_candidate.regularization_c == 1.5
    assert config.registry.model_name == "ticket-router"
    assert config.registry.candidate_alias == "candidate"
    assert config.registry.champion_alias == "champion"
    assert final_model_configuration_hash(config, random_seed=42) == (
        final_model_configuration_hash(config, random_seed=42)
    )
