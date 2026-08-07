"""Synthetic model and settings for API persistence integration tests."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import cast

import numpy as np
import pytest

from ticket_router.api.service import LoadedChampion, ProbabilisticPredictor
from ticket_router.config import Settings


class IntegrationProbabilityModel:
    classes_ = np.asarray(["Billing", "Returns", "Technical"], dtype=object)

    def predict_proba(self, values: Sequence[str]) -> object:
        return np.asarray([[0.8, 0.1, 0.1]] * len(values), dtype=np.float64)

    def predict(self, values: Sequence[str]) -> object:
        return np.asarray(["Billing"] * len(values), dtype=object)


@pytest.fixture
def api_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    return Settings.load(env_file=None).model_copy(
        update={"database_url": None, "database_required": False}
    )


@pytest.fixture
def loaded_champion() -> LoadedChampion:
    return LoadedChampion(
        model=cast(ProbabilisticPredictor, IntegrationProbabilityModel()),
        model_name="fixture-ticket-router",
        model_version="7",
        alias="champion",
        loaded_at=datetime(2026, 8, 7, tzinfo=UTC),
        labels=("Billing", "Returns", "Technical"),
        input_contract={"predictive_fields": ["subject", "body"]},
    )
