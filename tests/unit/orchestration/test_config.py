from pathlib import Path

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
