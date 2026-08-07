"""Tests for typed project and environment settings."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ticket_router.config import ProjectSettings, Settings


def test_settings_load_versioned_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    settings = Settings.load(env_file=None)

    assert settings.random_seed == 42
    assert settings.project_config.dataset.repository == "Tobi-Bueck/customer-support-tickets"
    assert len(settings.project_config.dataset.revision) == 40
    assert settings.project_config.dataset.language_filter == "en"
    assert settings.project_config.dataset.target_column == "queue"
    assert settings.project_config.dataset.text_columns == ("subject", "body")
    assert settings.project_config.dataset.number_of_target_queues == 10
    assert settings.project_config.analysis.near_empty_word_threshold == 3
    assert settings.project_config.preprocessing.mask_email_addresses is True
    assert settings.project_config.preprocessing.mask_phone_numbers is True
    assert settings.project_config.splitting.class_proportion_tolerance == 0.01
    assert settings.api_settings.maximum_batch_size == 100
    assert settings.api_settings.confidence_warning_threshold == 0.50
    assert settings.effective_mlflow_tracking_uri == "http://127.0.0.1:5000"
    assert settings.effective_registered_model_name == "ticket-router"


def test_environment_overrides_runtime_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("GLOBAL_RANDOM_SEED", "7")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setenv("INPUT_HMAC_SECRET", "unit-test-secret")

    settings = Settings.load(env_file=None)

    assert settings.log_level == "DEBUG"
    assert settings.random_seed == 7
    assert settings.effective_mlflow_tracking_uri == "http://mlflow:5000"
    assert "unit-test-secret" not in repr(settings)


def test_invalid_split_ratios_are_rejected() -> None:
    invalid_config = {
        "global_random_seed": 42,
        "dataset": {
            "repository": "owner/dataset",
            "revision": "a" * 40,
            "language_filter": "en",
            "target_column": "queue",
            "text_columns": ["subject", "body"],
            "number_of_target_queues": 10,
            "minimum_class_count": 100,
        },
        "split_ratios": {"train": 0.8, "validation": 0.15, "test": 0.15},
        "preprocessing": {
            "unicode_normalization": "NFKC",
            "mask_email_addresses": True,
            "mask_urls": True,
            "mask_phone_numbers": True,
            "email_mask": "<EMAIL>",
            "url_mask": "<URL>",
            "phone_mask": "<PHONE>",
        },
        "splitting": {
            "duplicate_policy": "exclude_contradictory_group_exact_duplicates",
            "class_proportion_tolerance": 0.01,
            "split_size_tolerance": 0.01,
        },
        "analysis": {
            "max_subject_characters": 2000,
            "max_body_characters": 20000,
            "max_text_characters": 22050,
            "near_empty_word_threshold": 3,
            "template_min_group_size": 5,
            "token_min_document_frequency": 10,
            "common_tokens_per_class": 10,
        },
        "mlflow": {"tracking_uri": "http://127.0.0.1:5000", "model_name": "router"},
    }

    with pytest.raises(ValidationError, match=r"split ratios must sum to 1\.0"):
        ProjectSettings.model_validate(invalid_config)


def test_missing_config_file_is_reported(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError):
        Settings.load(missing_path, env_file=None)
