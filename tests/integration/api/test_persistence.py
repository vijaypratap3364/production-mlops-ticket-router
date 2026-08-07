"""API-to-SQLAlchemy persistence, privacy, feedback, and degradation tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session

from ticket_router.api.app import create_app
from ticket_router.api.service import LoadedChampion
from ticket_router.config import Settings
from ticket_router.db.contracts import PredictionEvent
from ticket_router.db.exceptions import PersistenceUnavailableError
from ticket_router.db.models import FeedbackEventModel, PredictionEventModel
from ticket_router.db.repositories import InMemoryPredictionFeedbackRepository


def _migrated_database_url(tmp_path: Path) -> str:
    database_url = f"sqlite:///{(tmp_path / 'api-integration.db').as_posix()}"
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            config = Config("alembic.ini")
            config.attributes["connection"] = connection
            command.upgrade(config, "head")
    finally:
        engine.dispose()
    return database_url


@pytest.fixture
def persistent_api_client(
    tmp_path: Path,
    api_settings: Settings,
    loaded_champion: LoadedChampion,
) -> Iterator[tuple[TestClient, str]]:
    database_url = _migrated_database_url(tmp_path)
    settings = api_settings.model_copy(
        update={
            "database_url": SecretStr(database_url),
            "input_hmac_secret": SecretStr("integration-hmac-secret"),
        }
    )
    app = create_app(settings=settings, champion_loader=lambda _: loaded_champion)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, database_url


def test_prediction_and_feedback_are_persisted_without_raw_text(
    persistent_api_client: tuple[TestClient, str],
) -> None:
    client, database_url = persistent_api_client
    raw_marker = "private-body-marker-91b6@example.com"
    prediction_response = client.post(
        "/predict",
        json={
            "subject": "Invoice help",
            "body": f"Contact {raw_marker} about this charge",
            "metadata": {"client_name": "integration-test", "correlation_id": "corr-7"},
        },
    )
    prediction = prediction_response.json()
    feedback_response = client.post(
        "/feedback",
        json={
            "request_id": prediction["request_id"],
            "corrected_queue": "Technical",
            "accepted": False,
            "source": "reviewer",
            "comment": "Synthetic integration review",
        },
    )
    duplicate_response = client.post(
        "/feedback",
        json={
            "request_id": prediction["request_id"],
            "corrected_queue": "Billing",
            "source": "user",
        },
    )

    assert prediction_response.status_code == 200
    assert feedback_response.status_code == 201
    assert duplicate_response.status_code == 409
    assert duplicate_response.json()["error"]["code"] == "duplicate_feedback"

    engine = create_engine(database_url)
    try:
        columns = {column["name"] for column in inspect(engine).get_columns("prediction_events")}
        with Session(engine) as session:
            persisted_prediction = session.scalar(select(PredictionEventModel))
            persisted_feedback = session.scalar(select(FeedbackEventModel))
            assert persisted_prediction is not None
            assert persisted_feedback is not None
            assert persisted_prediction.redacted_text is None
            assert persisted_prediction.request_metadata == {
                "client_name": "integration-test",
                "correlation_id": "corr-7",
            }
            assert persisted_prediction.text_hash_algorithm == "hmac-sha256"
            assert raw_marker not in persisted_prediction.text_hash
            assert persisted_feedback.model_version == prediction["model_version"]
            assert persisted_feedback.source == "reviewer"
        assert "subject" not in columns
        assert "body" not in columns
    finally:
        engine.dispose()


def test_redacted_text_requires_explicit_flag(
    tmp_path: Path,
    api_settings: Settings,
    loaded_champion: LoadedChampion,
) -> None:
    database_url = _migrated_database_url(tmp_path)
    settings = api_settings.model_copy(
        update={
            "database_url": SecretStr(database_url),
            "store_redacted_ticket_text": True,
        }
    )
    app = create_app(settings=settings, champion_loader=lambda _: loaded_champion)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/predict",
            json={"body": "Email explicit@example.com about billing"},
        )

    assert response.status_code == 200
    engine = create_engine(database_url)
    try:
        with Session(engine) as session:
            persisted = session.scalar(select(PredictionEventModel))
            assert persisted is not None
            assert persisted.redacted_text is not None
            assert "explicit@example.com" not in persisted.redacted_text
            assert "<EMAIL>" in persisted.redacted_text
    finally:
        engine.dispose()


def test_prediction_response_degrades_gracefully_when_analytics_write_fails(
    api_settings: Settings,
    loaded_champion: LoadedChampion,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FailingAnalyticsRepository(InMemoryPredictionFeedbackRepository):
        def save_predictions(self, events: tuple[PredictionEvent, ...]) -> None:
            raise PersistenceUnavailableError("must-not-leak raw database diagnostics")

    raw_marker = "never-log-persistence-ticket-650f"
    app = create_app(
        settings=api_settings,
        champion_loader=lambda _: loaded_champion,
        store_factory=FailingAnalyticsRepository,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/predict", json={"body": raw_marker})
        metrics = client.get("/metrics")
        captured = capsys.readouterr().out

    assert response.status_code == 200
    assert 'ticket_router_persistence_failures_total{operation="prediction"} 1.0' in (metrics.text)
    assert raw_marker not in captured
    assert "must-not-leak" not in captured
