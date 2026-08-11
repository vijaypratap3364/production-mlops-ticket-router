"""End-to-end smoke test for an already-started local Compose stack."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


class SmokeTestError(RuntimeError):
    """Sanitized smoke-test failure safe for local logs."""


class SmokeTransport(Protocol):
    def get_json(self, path: str) -> dict[str, Any]: ...

    def post_json(self, path: str, payload: Mapping[str, object]) -> dict[str, Any]: ...

    def dashboard_health(self) -> None: ...


@dataclass(frozen=True)
class SmokeResult:
    request_id: str
    predicted_queue: str
    model_version: str
    feedback_id: str


class URLTransport:
    def __init__(self, *, api_base_url: str, dashboard_base_url: str, timeout: float) -> None:
        self._api_base_url = api_base_url.rstrip("/")
        self._dashboard_base_url = dashboard_base_url.rstrip("/")
        self._timeout = timeout

    def get_json(self, path: str) -> dict[str, Any]:
        return self._request_json("GET", self._api_base_url + path, None)

    def post_json(self, path: str, payload: Mapping[str, object]) -> dict[str, Any]:
        return self._request_json("POST", self._api_base_url + path, payload)

    def dashboard_health(self) -> None:
        request = urllib.request.Request(
            self._dashboard_base_url + "/_stcore/health",
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                if int(response.status) != 200:
                    raise SmokeTestError("Dashboard health check did not return HTTP 200.")
        except (OSError, urllib.error.URLError) as exc:
            raise SmokeTestError("Dashboard health check is unavailable.") from exc

    def _request_json(
        self,
        method: str,
        url: str,
        payload: Mapping[str, object] | None,
    ) -> dict[str, Any]:
        encoded = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url,
            data=encoded,
            headers={"Content-Type": "application/json"} if encoded is not None else {},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise SmokeTestError(f"API endpoint returned HTTP {exc.code}.") from exc
        except (OSError, urllib.error.URLError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SmokeTestError("API endpoint is unavailable or returned invalid JSON.") from exc
        if not isinstance(result, dict):
            raise SmokeTestError("API endpoint returned an unexpected response contract.")
        return cast(dict[str, Any], result)


def execute_smoke_test(
    *,
    transport: SmokeTransport,
    persistence_check: Callable[[str], None],
    ready_timeout_seconds: float,
    poll_interval_seconds: float = 2.0,
) -> SmokeResult:
    health = transport.get_json("/health")
    if health.get("status") != "ok":
        raise SmokeTestError("API process health check failed.")
    readiness = _wait_until_ready(
        transport,
        timeout_seconds=ready_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    if not readiness.get("model_ready") or not readiness.get("database_ready"):
        raise SmokeTestError("API dependencies are not ready.")
    prediction = transport.post_json(
        "/predict",
        {
            "subject": "Synthetic Compose smoke-test invoice question",
            "body": "Please route this local non-sensitive demonstration ticket.",
            "metadata": {"client_name": "compose-smoke"},
        },
    )
    request_id = _required_string(prediction, "request_id")
    predicted_queue = _required_string(prediction, "predicted_queue")
    model_version = _required_string(prediction, "model_version")
    feedback = transport.post_json(
        "/feedback",
        {
            "request_id": request_id,
            "corrected_queue": predicted_queue,
            "accepted": True,
            "comment": "Synthetic Compose smoke test",
            "source": "demo",
        },
    )
    feedback_id = _required_string(feedback, "feedback_id")
    persistence_check(request_id)
    transport.dashboard_health()
    return SmokeResult(
        request_id=request_id,
        predicted_queue=predicted_queue,
        model_version=model_version,
        feedback_id=feedback_id,
    )


def verify_postgres_persistence(database_url: str, request_id: str) -> None:
    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            prediction_count = connection.execute(
                text("SELECT COUNT(*) FROM prediction_events WHERE request_id = :request_id"),
                {"request_id": request_id},
            ).scalar_one()
            feedback_count = connection.execute(
                text("SELECT COUNT(*) FROM feedback_events WHERE request_id = :request_id"),
                {"request_id": request_id},
            ).scalar_one()
    except SQLAlchemyError as exc:
        raise SmokeTestError("PostgreSQL persistence verification failed.") from exc
    finally:
        engine.dispose()
    if prediction_count != 1 or feedback_count != 1:
        raise SmokeTestError("Prediction or feedback persistence record is missing.")


def _wait_until_ready(
    transport: SmokeTransport,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            readiness = transport.get_json("/ready")
        except SmokeTestError:
            readiness = {"ready": False}
        if readiness.get("ready") is True:
            return readiness
        if time.monotonic() >= deadline:
            raise SmokeTestError("API did not become ready before the configured timeout.")
        time.sleep(poll_interval_seconds)


def _required_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise SmokeTestError(f"API response is missing required field {key!r}.")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=os.getenv("API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument(
        "--dashboard-url",
        default=os.getenv("DASHBOARD_BASE_URL", "http://127.0.0.1:8501"),
    )
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument(
        "--ready-timeout-seconds",
        type=float,
        default=float(os.getenv("SMOKE_READY_TIMEOUT_SECONDS", "120")),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.database_url:
        print("Smoke test failed: DATABASE_URL is required for persistence verification.")
        return 2
    transport = URLTransport(
        api_base_url=args.api_url,
        dashboard_base_url=args.dashboard_url,
        timeout=5.0,
    )
    try:
        result = execute_smoke_test(
            transport=transport,
            persistence_check=lambda request_id: verify_postgres_persistence(
                args.database_url,
                request_id,
            ),
            ready_timeout_seconds=args.ready_timeout_seconds,
        )
    except SmokeTestError as exc:
        print(f"Smoke test failed: {exc}")
        return 1
    print("Compose smoke test passed.")
    print(f"Request ID: {result.request_id}")
    print(f"Predicted queue: {result.predicted_queue}")
    print(f"Model version: {result.model_version}")
    print(f"Feedback ID: {result.feedback_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
