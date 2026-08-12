"""API lifecycle reliability after champion load and across process restarts."""

from __future__ import annotations

from fastapi.testclient import TestClient

from ticket_router.api.app import create_app
from ticket_router.api.errors import ModelUnavailableError
from ticket_router.api.service import LoadedChampion
from ticket_router.config import Settings


def test_loaded_model_does_not_depend_on_mlflow_for_later_predictions(
    api_settings: Settings,
    loaded_champion: LoadedChampion,
) -> None:
    mlflow_available = True
    load_calls = 0

    def loader(_: Settings) -> LoadedChampion:
        nonlocal load_calls
        load_calls += 1
        if not mlflow_available:
            raise ModelUnavailableError("synthetic MLflow outage")
        return loaded_champion

    app = create_app(settings=api_settings, champion_loader=loader)
    with TestClient(app, raise_server_exceptions=False) as client:
        first = client.post("/predict", json={"body": "Synthetic billing request"})
        mlflow_available = False
        second = client.post("/predict", json={"body": "Synthetic network request"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert load_calls == 1


def test_api_restart_reloads_champion_and_returns_ready(
    api_settings: Settings,
    loaded_champion: LoadedChampion,
) -> None:
    load_calls = 0

    def loader(_: Settings) -> LoadedChampion:
        nonlocal load_calls
        load_calls += 1
        return loaded_champion

    app = create_app(settings=api_settings, champion_loader=loader)
    for _ in range(2):
        with TestClient(app, raise_server_exceptions=False) as client:
            assert client.get("/ready").status_code == 200
            assert (
                client.post("/predict", json={"body": "Synthetic restart check"}).status_code == 200
            )

    assert load_calls == 2
