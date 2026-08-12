"""Bounded local API reliability checks with optional recoverable Compose disruptions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient

from ticket_router.api.app import create_app
from ticket_router.api.errors import ModelUnavailableError
from ticket_router.api.service import LoadedChampion
from ticket_router.benchmarking.contracts import ReliabilityScenario
from ticket_router.config import Settings
from ticket_router.data.manifests import atomic_write_json
from ticket_router.monitoring.config import MonitoringConfig
from ticket_router.monitoring.policy import evaluate_alert_policy
from ticket_router.monitoring.quality import calculate_delayed_quality


class OperationalValidationError(RuntimeError):
    """Sanitized failure from a local validation prerequisite."""


class ComposeController:
    """Recoverable, project-scoped Docker Compose service controller."""

    def __init__(self, *, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds

    def stop(self, service: str) -> None:
        self._run("stop", service)

    def start(self, service: str) -> None:
        self._run("start", service)

    def restart(self, service: str) -> None:
        self._run("restart", service)

    def wait_postgres(self, *, user: str, database: str) -> None:
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            result = subprocess.run(
                [
                    "docker",
                    "compose",
                    "exec",
                    "-T",
                    "postgres",
                    "pg_isready",
                    "-U",
                    user,
                    "-d",
                    database,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                return
            time.sleep(1.0)
        raise OperationalValidationError("PostgreSQL did not recover before the timeout")

    def _run(self, operation: str, service: str) -> None:
        result = subprocess.run(
            ["docker", "compose", operation, service],
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        if result.returncode != 0:
            raise OperationalValidationError(
                f"Docker Compose {operation} failed for service {service!r}"
            )


def run_validation(
    *,
    api_url: str,
    mlflow_url: str,
    maximum_batch_size: int,
    run_compose_disruptions: bool,
    recovery_timeout_seconds: float,
    postgres_user: str,
    postgres_database: str,
) -> tuple[ReliabilityScenario, ...]:
    """Execute bounded HTTP checks and always recover intentionally stopped services."""
    scenarios: list[ReliabilityScenario] = []
    with httpx.Client(base_url=api_url, timeout=35.0) as client:
        scenarios.extend(_baseline_http_scenarios(client, maximum_batch_size=maximum_batch_size))
        if run_compose_disruptions:
            controller = ComposeController(timeout_seconds=recovery_timeout_seconds)
            scenarios.append(
                _mlflow_outage_scenario(
                    client,
                    controller=controller,
                    mlflow_url=mlflow_url,
                    recovery_timeout_seconds=recovery_timeout_seconds,
                )
            )
            scenarios.append(
                _database_outage_scenario(
                    client,
                    controller=controller,
                    mlflow_url=mlflow_url,
                    recovery_timeout_seconds=recovery_timeout_seconds,
                    postgres_user=postgres_user,
                    postgres_database=postgres_database,
                )
            )
            scenarios.append(
                _api_restart_scenario(
                    client,
                    controller=controller,
                    recovery_timeout_seconds=recovery_timeout_seconds,
                )
            )
    scenarios.append(_missing_champion_scenario())
    scenarios.append(_insufficient_monitoring_scenario())
    return tuple(scenarios)


def _baseline_http_scenarios(
    client: httpx.Client,
    *,
    maximum_batch_size: int,
) -> tuple[ReliabilityScenario, ...]:
    readiness = client.get("/ready")
    ready = readiness.status_code == 200 and readiness.json().get("ready") is True
    malformed = client.post(
        "/predict",
        content=b"{not-valid-json",
        headers={"Content-Type": "application/json"},
    )
    oversized = client.post(
        "/predict/batch",
        json={"items": [{"body": "Synthetic bounded item"}] * (maximum_batch_size + 1)},
    )
    prediction = client.post(
        "/predict",
        json={"subject": "Synthetic reliability invoice", "body": "Local bounded check."},
    )
    feedback_status = -1
    duplicate_status = -1
    if prediction.status_code == 200:
        payload = prediction.json()
        feedback_payload = {
            "request_id": payload["request_id"],
            "corrected_queue": payload["predicted_queue"],
            "accepted": True,
            "comment": "Synthetic reliability validation",
            "source": "demo",
        }
        feedback_status = client.post("/feedback", json=feedback_payload).status_code
        duplicate_status = client.post("/feedback", json=feedback_payload).status_code
    return (
        ReliabilityScenario(
            name="api_ready_before_validation",
            passed=ready,
            expected="HTTP 200 and ready=true",
            observed=f"HTTP {readiness.status_code} and ready={readiness.json().get('ready')}",
        ),
        _status_scenario("malformed_request", malformed, expected_status=422),
        _status_scenario("oversized_batch", oversized, expected_status=422),
        ReliabilityScenario(
            name="duplicate_feedback",
            passed=feedback_status == 201 and duplicate_status == 409,
            expected="first feedback HTTP 201; duplicate HTTP 409",
            observed=f"first HTTP {feedback_status}; duplicate HTTP {duplicate_status}",
        ),
    )


def _mlflow_outage_scenario(
    client: httpx.Client,
    *,
    controller: ComposeController,
    mlflow_url: str,
    recovery_timeout_seconds: float,
) -> ReliabilityScenario:
    try:
        controller.stop("mlflow")
        scenario = _probe_mlflow_outage(client)
    finally:
        controller.start("mlflow")
        _wait_url(f"{mlflow_url.rstrip('/')}/health", timeout_seconds=recovery_timeout_seconds)
    return scenario


def _database_outage_scenario(
    client: httpx.Client,
    *,
    controller: ComposeController,
    mlflow_url: str,
    recovery_timeout_seconds: float,
    postgres_user: str,
    postgres_database: str,
) -> ReliabilityScenario:
    try:
        controller.stop("postgres")
        scenario = _probe_database_outage(client)
    finally:
        controller.start("postgres")
        controller.wait_postgres(user=postgres_user, database=postgres_database)
        _wait_url(f"{mlflow_url.rstrip('/')}/health", timeout_seconds=recovery_timeout_seconds)
    return scenario


def _api_restart_scenario(
    client: httpx.Client,
    *,
    controller: ComposeController,
    recovery_timeout_seconds: float,
) -> ReliabilityScenario:
    controller.restart("api")
    return _probe_api_restart(client, timeout_seconds=recovery_timeout_seconds)


def _probe_mlflow_outage(client: httpx.Client) -> ReliabilityScenario:
    prediction = client.post(
        "/predict",
        json={"body": "Synthetic prediction while MLflow is temporarily stopped."},
    )
    return ReliabilityScenario(
        name="mlflow_unavailable_after_model_load",
        passed=prediction.status_code == 200,
        expected="prediction HTTP 200 from already-loaded champion",
        observed=f"prediction HTTP {prediction.status_code}",
    )


def _probe_database_outage(client: httpx.Client) -> ReliabilityScenario:
    prediction = client.post(
        "/predict",
        json={"body": "Synthetic prediction while PostgreSQL is temporarily stopped."},
    )
    feedback_status = -1
    if prediction.status_code == 200:
        payload = prediction.json()
        feedback_status = client.post(
            "/feedback",
            json={
                "request_id": payload["request_id"],
                "corrected_queue": payload["predicted_queue"],
                "source": "demo",
            },
        ).status_code
    return ReliabilityScenario(
        name="database_temporarily_unavailable",
        passed=prediction.status_code == 200 and feedback_status == 503,
        expected="prediction degrades to HTTP 200; required feedback returns HTTP 503",
        observed=f"prediction HTTP {prediction.status_code}; feedback HTTP {feedback_status}",
    )


def _probe_api_restart(
    client: httpx.Client,
    *,
    timeout_seconds: float,
) -> ReliabilityScenario:
    readiness = _wait_ready(client, timeout_seconds=timeout_seconds)
    prediction = client.post(
        "/predict",
        json={"body": "Synthetic prediction after the local API restart."},
    )
    return ReliabilityScenario(
        name="api_restart",
        passed=readiness and prediction.status_code == 200,
        expected="API returns ready=true and prediction HTTP 200 after restart",
        observed=f"ready={readiness}; prediction HTTP {prediction.status_code}",
    )


def _missing_champion_scenario() -> ReliabilityScenario:
    settings = Settings.load(env_file=None).model_copy(
        update={"database_url": None, "database_required": False}
    )

    def missing_loader(_: Settings) -> LoadedChampion:
        raise ModelUnavailableError("synthetic missing champion")

    app = create_app(settings=settings, champion_loader=missing_loader)
    with TestClient(app, raise_server_exceptions=False) as client:
        health_status = client.get("/health").status_code
        ready_status = client.get("/ready").status_code
        prediction_status = client.post("/predict", json={"body": "Synthetic request"}).status_code
    return ReliabilityScenario(
        name="champion_model_missing",
        passed=(health_status, ready_status, prediction_status) == (200, 503, 503),
        expected="health HTTP 200; readiness and prediction HTTP 503",
        observed=(
            f"health HTTP {health_status}; readiness HTTP {ready_status}; "
            f"prediction HTTP {prediction_status}"
        ),
    )


def _insufficient_monitoring_scenario() -> ReliabilityScenario:
    config = MonitoringConfig.load()
    quality = calculate_delayed_quality(
        (),
        minimum_sample_count=config.quality.minimum_feedback_count,
    )
    decision = evaluate_alert_policy(
        event_count=0,
        minimum_event_count=config.current_window.minimum_event_count,
        drift=None,
        quality=quality,
        reference_macro_f1=1.0,
        settings=config.alerts,
    )
    return ReliabilityScenario(
        name="monitoring_insufficient_data",
        passed=decision.status == "insufficient_data",
        expected="status insufficient_data without a fabricated drift alert",
        observed=f"status {decision.status}",
    )


def _status_scenario(
    name: str,
    response: httpx.Response,
    *,
    expected_status: int,
) -> ReliabilityScenario:
    return ReliabilityScenario(
        name=name,
        passed=response.status_code == expected_status,
        expected=f"HTTP {expected_status}",
        observed=f"HTTP {response.status_code}",
    )


def _wait_url(url: str, *, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            if httpx.get(url, timeout=5.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise OperationalValidationError("local service did not recover before the timeout")


def _wait_ready(client: httpx.Client, *, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            response = client.get("/ready")
            if response.status_code == 200 and response.json().get("ready") is True:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--mlflow-url", default="http://127.0.0.1:5000")
    parser.add_argument("--maximum-batch-size", type=int, default=100)
    parser.add_argument("--recovery-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--postgres-user", default="ticket_router")
    parser.add_argument("--postgres-database", default="ticket_router")
    parser.add_argument("--run-compose-disruptions", action="store_true")
    parser.add_argument(
        "--probe",
        choices=("mlflow-outage", "database-outage", "api-restart"),
        help="Measure one externally orchestrated state without controlling Docker.",
    )
    parser.add_argument(
        "--additional-result",
        type=Path,
        action="append",
        default=[],
        help="Merge scenarios from another generated operational result file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/benchmarks/reliability_results.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    scenarios: tuple[ReliabilityScenario, ...]
    try:
        if args.probe:
            with httpx.Client(base_url=args.api_url, timeout=35.0) as client:
                probes: dict[str, Callable[[], ReliabilityScenario]] = {
                    "mlflow-outage": lambda: _probe_mlflow_outage(client),
                    "database-outage": lambda: _probe_database_outage(client),
                    "api-restart": lambda: _probe_api_restart(
                        client,
                        timeout_seconds=args.recovery_timeout_seconds,
                    ),
                }
                scenarios = (probes[args.probe](),)
        else:
            scenarios = run_validation(
                api_url=args.api_url,
                mlflow_url=args.mlflow_url,
                maximum_batch_size=args.maximum_batch_size,
                run_compose_disruptions=args.run_compose_disruptions,
                recovery_timeout_seconds=args.recovery_timeout_seconds,
                postgres_user=args.postgres_user,
                postgres_database=args.postgres_database,
            )
        scenarios = (*scenarios, *_read_additional_results(args.additional_result))
    except (OperationalValidationError, httpx.HTTPError, OSError, ValueError) as exc:
        print(f"Operational validation failed: {exc}", file=sys.stderr)
        return 1
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "api_url": args.api_url,
        "compose_disruptions_enabled": args.run_compose_disruptions,
        "externally_orchestrated_result_files": [
            path.as_posix() for path in args.additional_result
        ],
        "scenarios": [scenario.model_dump(mode="json") for scenario in scenarios],
        "all_passed": all(scenario.passed for scenario in scenarios),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0 if payload["all_passed"] else 1


def _read_additional_results(paths: Sequence[Path]) -> tuple[ReliabilityScenario, ...]:
    results: list[ReliabilityScenario] = []
    for path in paths:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or not isinstance(loaded.get("scenarios"), list):
            raise ValueError(f"operational result contract is invalid: {path}")
        results.extend(ReliabilityScenario.model_validate(item) for item in loaded["scenarios"])
    return tuple(results)


if __name__ == "__main__":
    raise SystemExit(main())
