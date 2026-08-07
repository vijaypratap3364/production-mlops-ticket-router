"""Dependency-injected FastAPI fixtures with no MLflow server or database."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from typing import cast

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ticket_router.api.app import create_app
from ticket_router.api.service import LoadedChampion, ProbabilisticPredictor
from ticket_router.config import Settings


class FakeProbabilityModel:
    classes_ = np.asarray(["Billing", "Returns", "Technical"], dtype=object)

    def __init__(self) -> None:
        self.last_inputs: list[str] = []

    def predict_proba(self, values: Sequence[str]) -> object:
        self.last_inputs = list(values)
        rows: list[list[float]] = []
        for value in values:
            lowered = value.casefold()
            if "uncertain" in lowered:
                rows.append([0.40, 0.35, 0.25])
            elif "return" in lowered:
                rows.append([0.05, 0.90, 0.05])
            elif "network" in lowered:
                rows.append([0.05, 0.10, 0.85])
            else:
                rows.append([0.80, 0.10, 0.10])
        return np.asarray(rows, dtype=np.float64)

    def predict(self, values: Sequence[str]) -> object:
        probabilities = np.asarray(self.predict_proba(values), dtype=np.float64)
        return self.classes_[probabilities.argmax(axis=1)]


@pytest.fixture
def api_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_REQUIRED", raising=False)
    return Settings.load(env_file=None).model_copy(
        update={"database_url": None, "database_required": False}
    )


@pytest.fixture
def fake_model() -> FakeProbabilityModel:
    return FakeProbabilityModel()


@pytest.fixture
def loaded_champion(fake_model: FakeProbabilityModel) -> LoadedChampion:
    return LoadedChampion(
        model=cast(ProbabilisticPredictor, fake_model),
        model_name="fixture-ticket-router",
        model_version="7",
        alias="champion",
        loaded_at=datetime(2026, 8, 7, tzinfo=UTC),
        labels=("Billing", "Returns", "Technical"),
        input_contract={
            "predictive_fields": ["subject", "body"],
            "derived_model_field": "model_text",
        },
    )


@pytest.fixture
def api_client(
    api_settings: Settings,
    loaded_champion: LoadedChampion,
) -> Iterator[TestClient]:
    app = create_app(
        settings=api_settings,
        champion_loader=lambda _: loaded_champion,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
