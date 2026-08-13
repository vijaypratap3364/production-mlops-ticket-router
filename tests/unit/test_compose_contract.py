"""Static Docker/Compose safety contract that does not require a local daemon."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml


def _compose() -> dict[str, Any]:
    payload = yaml.safe_load(Path("compose.yaml").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(dict[str, Any], payload)


def test_compose_declares_required_services_health_and_persistence() -> None:
    compose = _compose()
    services = cast(dict[str, dict[str, Any]], compose["services"])
    required = {
        "postgres",
        "mlflow",
        "migrate",
        "api",
        "dashboard",
        "prefect-server",
        "prefect-worker",
        "bootstrap",
        "smoke-test",
    }

    assert required.issubset(services)
    for service in ("postgres", "mlflow", "api", "dashboard", "prefect-server"):
        assert "healthcheck" in services[service]
    assert services["api"]["depends_on"]["migrate"]["condition"] == (
        "service_completed_successfully"
    )
    assert services["api"]["depends_on"]["mlflow"]["condition"] == "service_healthy"
    assert "postgres_data" in compose["volumes"]
    assert "mlflow_artifacts" in compose["volumes"]
    assert services["bootstrap"]["profiles"] == ["bootstrap"]
    mlflow_command = " ".join(str(value) for value in services["mlflow"]["command"])
    assert "postgresql+psycopg2://" in mlflow_command
    assert "--workers ${MLFLOW_WORKERS:-1}" in mlflow_command


def test_normal_api_startup_cannot_train_or_promote() -> None:
    services = cast(dict[str, dict[str, Any]], _compose()["services"])
    api_command = " ".join(str(value) for value in services["api"].get("command", []))
    forbidden = ("train", "evaluate_final", "promote", "bootstrap")

    assert not any(token in api_command for token in forbidden)
    assert services["bootstrap"]["command"][-1] == "--promote-champion"


def test_container_ports_bind_only_to_loopback() -> None:
    services = cast(dict[str, dict[str, Any]], _compose()["services"])
    for service_name in ("postgres", "mlflow", "api", "dashboard", "prefect-server"):
        for port in services[service_name].get("ports", []):
            assert str(port).startswith("127.0.0.1:")


def test_project_runtime_images_use_non_root_users() -> None:
    for filename in (
        "docker/api.Dockerfile",
        "docker/dashboard.Dockerfile",
        "docker/worker.Dockerfile",
        "docker/mlflow.Dockerfile",
    ):
        dockerfile = Path(filename).read_text(encoding="utf-8")
        assert "FROM python:3.12-slim-bookworm" in dockerfile
        assert "USER ticket-router" in dockerfile
        assert "COPY ." not in dockerfile


def test_lineage_capable_images_receive_explicit_source_identity() -> None:
    compose = _compose()
    for extension_name in ("x-api-build", "x-worker-build"):
        build = cast(dict[str, Any], compose[extension_name])
        assert build["args"] == {
            "SOURCE_GIT_COMMIT": "${SOURCE_GIT_COMMIT:-}",
            "SOURCE_GIT_DIRTY": "${SOURCE_GIT_DIRTY:-false}",
        }
    for filename in ("docker/api.Dockerfile", "docker/worker.Dockerfile"):
        dockerfile = Path(filename).read_text(encoding="utf-8")
        assert 'ARG SOURCE_GIT_COMMIT=""' in dockerfile
        assert "SOURCE_GIT_COMMIT=$SOURCE_GIT_COMMIT" in dockerfile


def test_dashboard_image_disables_usage_telemetry() -> None:
    dockerfile = Path("docker/dashboard.Dockerfile").read_text(encoding="utf-8")

    assert '"--browser.gatherUsageStats", "false"' in dockerfile
