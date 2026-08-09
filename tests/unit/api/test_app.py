"""Production inference API behavior and privacy regression tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ticket_router.api.app import create_app
from ticket_router.api.errors import ModelUnavailableError
from ticket_router.api.service import LoadedChampion, ProbabilisticPredictor
from ticket_router.config import Settings
from ticket_router.db.connectivity import DatabaseProbe


def test_health_and_readiness(api_client: TestClient) -> None:
    assert api_client.get("/health").json() == {"status": "ok"}
    readiness = api_client.get("/ready")
    assert readiness.status_code == 200
    assert readiness.json() == {
        "ready": True,
        "model_ready": True,
        "database_ready": True,
    }


def test_valid_prediction_and_top_k_ordering(api_client: TestClient) -> None:
    response = api_client.post(
        "/predict",
        json={
            "subject": "Invoice question",
            "body": "Please explain this billing charge.",
            "metadata": {"client_name": "unit-test"},
        },
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["predicted_queue"] == "Billing"
    assert payload["confidence"] == pytest.approx(0.8)
    assert payload["model_name"] == "fixture-ticket-router"
    assert payload["model_version"] == "7"
    assert payload["warning"] is None
    confidences = [item["confidence"] for item in payload["top_k"]]
    assert confidences == sorted(confidences, reverse=True)
    assert response.headers["X-Request-ID"] == payload["request_id"]


def test_shared_preprocessing_masks_pii(
    api_client: TestClient,
    fake_model: Any,
) -> None:
    response = api_client.post(
        "/predict",
        json={"subject": "Contact me", "body": "Email private@example.com for billing"},
    )

    assert response.status_code == 200
    observed = cast(list[str], fake_model.last_inputs)[0]
    assert "private@example.com" not in observed
    assert "<EMAIL>" in observed


@pytest.mark.parametrize(
    "payload",
    [
        {"subject": "  ", "body": "\n"},
        {"subject": "ok", "body": "x" * 20001},
    ],
)
def test_invalid_empty_or_oversized_ticket(
    api_client: TestClient,
    payload: dict[str, str],
) -> None:
    response = api_client.post("/predict", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_batch_prediction(api_client: TestClient) -> None:
    response = api_client.post(
        "/predict/batch",
        json={
            "items": [
                {"subject": "Return", "body": "I need an exchange"},
                {"subject": "Outage", "body": "Network is unavailable"},
            ]
        },
    )

    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert [item["predicted_queue"] for item in predictions] == ["Returns", "Technical"]
    assert len({item["request_id"] for item in predictions}) == 2


def test_oversized_batch_is_rejected(
    api_settings: Settings,
    loaded_champion: LoadedChampion,
) -> None:
    settings = api_settings.model_copy(update={"api_maximum_batch_size": 1})
    app = create_app(settings=settings, champion_loader=lambda _: loaded_champion)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/predict/batch",
            json={"items": [{"body": "billing"}, {"body": "network"}]},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_low_confidence_warning(api_client: TestClient) -> None:
    response = api_client.post(
        "/predict",
        json={"subject": "Uncertain request", "body": "uncertain category"},
    )

    assert response.status_code == 200
    assert response.json()["confidence"] == pytest.approx(0.4)
    assert response.json()["warning"] is not None


def test_model_metadata(api_client: TestClient) -> None:
    response = api_client.get("/model")

    assert response.status_code == 200
    assert response.json()["model_name"] == "fixture-ticket-router"
    assert response.json()["model_version"] == "7"
    assert response.json()["alias"] == "champion"
    assert response.json()["input_contract"]["predictive_fields"] == ["subject", "body"]
    assert response.json()["model_card_summary"]
    assert response.json()["limitations"]


def test_dashboard_read_endpoints_are_safe_without_postgresql(api_client: TestClient) -> None:
    history = api_client.get("/monitoring/history")
    status = api_client.get("/system/status")

    assert history.status_code == 200
    assert history.json() == {"runs": []}
    assert status.status_code == 200
    assert status.json()["database_mode"] == "memory"
    assert status.json()["database_status"] == "not_configured"
    assert status.json()["mlflow_model_available"] is True
    assert status.json()["latest_monitoring_run"] is None
    assert status.json()["latest_retraining_run"] is None


def test_missing_champion_keeps_process_live_but_not_ready(api_settings: Settings) -> None:
    def missing_loader(_: Settings) -> LoadedChampion:
        raise ModelUnavailableError("fixture has no champion")

    app = create_app(settings=api_settings, champion_loader=missing_loader)
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/health").status_code == 200
        readiness = client.get("/ready")
        prediction = client.post("/predict", json={"body": "billing"})

    assert readiness.status_code == 503
    assert readiness.json()["model_ready"] is False
    assert prediction.status_code == 503
    assert prediction.json()["error"]["code"] == "model_unavailable"


def test_database_failure_is_not_ready_and_returns_structured_error(
    api_settings: Settings,
    loaded_champion: LoadedChampion,
) -> None:
    class FailingProbe:
        @property
        def ready(self) -> bool:
            return False

        def connect(self) -> None:
            raise ConnectionError("secret database diagnostics")

        def close(self) -> None:
            return None

    settings = api_settings.model_copy(
        update={"database_url": SecretStr("postgresql://secret@localhost/example")}
    )
    app = create_app(
        settings=settings,
        champion_loader=lambda _: loaded_champion,
        database_probe_factory=lambda _: cast(DatabaseProbe, FailingProbe()),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        readiness = client.get("/ready")
        prediction = client.post("/predict", json={"body": "billing"})

    assert readiness.status_code == 503
    assert readiness.json()["database_ready"] is False
    assert prediction.status_code == 503
    assert prediction.json()["error"]["code"] == "database_unavailable"
    assert "secret" not in prediction.text


def test_mocked_model_failure_is_sanitized(
    api_settings: Settings,
    loaded_champion: LoadedChampion,
) -> None:
    class FailingModel:
        classes_ = np.asarray(["Billing", "Returns", "Technical"], dtype=object)

        def predict(self, values: Sequence[str]) -> object:
            raise RuntimeError(f"sensitive model input: {values[0]}")

        def predict_proba(self, values: Sequence[str]) -> object:
            raise RuntimeError(f"sensitive model input: {values[0]}")

    failed_champion = LoadedChampion(
        model=cast(ProbabilisticPredictor, FailingModel()),
        model_name=loaded_champion.model_name,
        model_version=loaded_champion.model_version,
        alias=loaded_champion.alias,
        loaded_at=loaded_champion.loaded_at,
        labels=loaded_champion.labels,
        input_contract=loaded_champion.input_contract,
    )
    app = create_app(settings=api_settings, champion_loader=lambda _: failed_champion)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/predict", json={"body": "private ticket content"})

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "prediction_failure"
    assert "private ticket content" not in response.text


def test_feedback_requires_known_request_and_valid_label(api_client: TestClient) -> None:
    prediction = api_client.post("/predict", json={"body": "billing invoice"}).json()
    accepted = api_client.post(
        "/feedback",
        json={
            "request_id": prediction["request_id"],
            "corrected_queue": "Technical",
            "accepted": False,
            "comment": "Reviewed synthetic example",
        },
    )
    invalid_label = api_client.post(
        "/feedback",
        json={"request_id": prediction["request_id"], "corrected_queue": "Unknown"},
    )
    unknown_request = api_client.post(
        "/feedback",
        json={"request_id": "missing", "corrected_queue": "Billing"},
    )

    assert accepted.status_code == 201
    assert accepted.json()["corrected_queue"] == "Technical"
    assert invalid_label.status_code == 422
    assert invalid_label.json()["error"]["code"] == "invalid_feedback_label"
    assert unknown_request.status_code == 404
    assert unknown_request.json()["error"]["code"] == "unknown_feedback_request"


def test_metrics_are_prometheus_compatible(api_client: TestClient) -> None:
    api_client.post("/predict", json={"body": "network outage"})
    response = api_client.get("/metrics")

    assert response.status_code == 200
    assert "ticket_router_predictions_total 1.0" in response.text
    assert 'ticket_router_predicted_labels_total{queue="Technical"} 1.0' in response.text


def test_normal_logs_do_not_contain_raw_ticket_text(
    api_settings: Settings,
    loaded_champion: LoadedChampion,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sensitive_text = "never-log-this-ticket-7f963e"
    app = create_app(settings=api_settings, champion_loader=lambda _: loaded_champion)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/predict", json={"body": sensitive_text})
        captured = capsys.readouterr().out

    assert response.status_code == 200
    assert sensitive_text not in captured
