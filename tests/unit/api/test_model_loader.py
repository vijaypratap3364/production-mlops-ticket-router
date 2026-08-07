"""Champion alias resolution and load-time probability contract tests."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from ticket_router.api.errors import ModelUnavailableError
from ticket_router.api.model_loader import load_champion
from ticket_router.config import Settings
from ticket_router.registry.service import ModelRegistryService, RegisteredVersion


class LoaderFixtureModel:
    classes_ = np.asarray(["Billing", "Technical"], dtype=object)

    def predict(self, values: Sequence[str]) -> object:
        return np.asarray(["Billing"] * len(values), dtype=object)

    def predict_proba(self, values: Sequence[str]) -> object:
        return np.asarray([[0.8, 0.2]] * len(values), dtype=np.float64)


def test_loader_resolves_alias_then_loads_numeric_version(
    api_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str]] = []

    def resolve_alias(
        _: ModelRegistryService,
        *,
        name: str,
        alias: str,
    ) -> RegisteredVersion:
        calls.append(("resolve", name, alias))
        return RegisteredVersion(name=name, version="12", run_id="run", source="source")

    def load_version(
        _: ModelRegistryService,
        *,
        name: str,
        version: str,
    ) -> LoaderFixtureModel:
        calls.append(("load", name, version))
        return LoaderFixtureModel()

    monkeypatch.setattr(ModelRegistryService, "resolve_alias", resolve_alias)
    monkeypatch.setattr(ModelRegistryService, "load_version", load_version)

    champion = load_champion(api_settings)

    assert calls == [
        ("resolve", "ticket-router", "champion"),
        ("load", "ticket-router", "12"),
    ]
    assert champion.model_version == "12"
    assert champion.labels == ("Billing", "Technical")


def test_loader_rejects_missing_champion_alias(
    api_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def resolve_alias(
        _: ModelRegistryService,
        *,
        name: str,
        alias: str,
    ) -> None:
        return None

    monkeypatch.setattr(ModelRegistryService, "resolve_alias", resolve_alias)

    with pytest.raises(ModelUnavailableError, match="champion alias"):
        load_champion(api_settings)


def test_loader_rejects_invalid_probability_contract(
    api_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidProbabilityModel(LoaderFixtureModel):
        def predict_proba(self, values: Sequence[str]) -> object:
            return np.asarray([[0.8, 0.8]] * len(values), dtype=np.float64)

    monkeypatch.setattr(
        ModelRegistryService,
        "resolve_alias",
        lambda _self, *, name, alias: RegisteredVersion(
            name=name,
            version="12",
            run_id="run",
            source="source",
        ),
    )
    monkeypatch.setattr(
        ModelRegistryService,
        "load_version",
        lambda _self, *, name, version: InvalidProbabilityModel(),
    )

    with pytest.raises(ModelUnavailableError, match="prediction contract"):
        load_champion(api_settings)
