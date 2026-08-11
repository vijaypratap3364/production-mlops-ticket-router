from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from ticket_router.orchestration import deploy
from ticket_router.orchestration.config import (
    OrchestrationConfig,
    orchestration_configuration_hash,
)
from ticket_router.orchestration.deploy import deployment_specs


def test_orchestration_config_is_stable_and_schedules_are_disabled() -> None:
    config = OrchestrationConfig.load(Path("configs/orchestration.yaml"))

    assert config.runtime.temporary_failure_retries == 2
    assert len(orchestration_configuration_hash(config)) == 64
    specs = deployment_specs(config)
    assert {spec.flow_name for spec in specs} == {
        "ingest-data-flow",
        "train-candidate-flow",
        "monitoring-flow",
        "conditional-retraining-flow",
    }
    assert all(spec.paused for spec in specs)
    monitoring = next(spec for spec in specs if spec.name == "daily-monitoring")
    assert monitoring.cron == "0 2 * * *"
    assert monitoring.timezone == "UTC"


def test_local_process_deployments_are_applied_without_an_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applied: list[dict[str, Any]] = []

    class FakeDeployment:
        def __init__(self, identifier: UUID) -> None:
            self.identifier = identifier

        def apply(self) -> UUID:
            return self.identifier

    class FakeFlow:
        def __init__(self, identifier: UUID) -> None:
            self.identifier = identifier

        def to_deployment(self, name: str, **kwargs: Any) -> FakeDeployment:
            applied.append({"name": name, **kwargs})
            return FakeDeployment(self.identifier)

    identifiers = tuple(UUID(int=index) for index in range(1, 5))
    for attribute, identifier in zip(
        (
            "ingest_data_flow",
            "train_candidate_flow",
            "monitoring_flow",
            "conditional_retraining_flow",
        ),
        identifiers,
        strict=True,
    ):
        monkeypatch.setattr(deploy, attribute, FakeFlow(identifier))

    result = deploy.register_deployments(
        OrchestrationConfig.load(Path("configs/orchestration.yaml"))
    )

    assert result == tuple(str(identifier) for identifier in identifiers)
    assert len(applied) == 4
    assert all(item["work_pool_name"] == "ticket-router-local" for item in applied)
    assert all(item["paused"] is True for item in applied)
    assert all("image" not in item for item in applied)
