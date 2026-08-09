"""Register local Prefect deployments and disabled-by-default schedules."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prefect.client.schemas.schedules import CronSchedule

from ticket_router.orchestration.config import OrchestrationConfig
from ticket_router.orchestration.flows import (
    conditional_retraining_flow,
    ingest_data_flow,
    monitoring_flow,
    train_candidate_flow,
)


@dataclass(frozen=True)
class DeploymentSpec:
    name: str
    flow_name: str
    cron: str | None
    timezone: str | None
    paused: bool


def deployment_specs(config: OrchestrationConfig) -> tuple[DeploymentSpec, ...]:
    """Build deterministic local deployment metadata from versioned configuration."""
    paused = not config.schedules.enabled
    return (
        DeploymentSpec("ingest-data", "ingest-data-flow", None, None, paused),
        DeploymentSpec("train-candidate", "train-candidate-flow", None, None, paused),
        DeploymentSpec(
            "daily-monitoring",
            "monitoring-flow",
            config.schedules.monitoring_cron,
            config.schedules.timezone,
            paused,
        ),
        DeploymentSpec(
            "weekly-retraining-evaluation",
            "conditional-retraining-flow",
            config.schedules.retraining_evaluation_cron,
            config.schedules.timezone,
            paused,
        ),
    )


def register_deployments(config: OrchestrationConfig) -> tuple[str, ...]:
    """Write four local-process deployments to a running Prefect server."""
    flows: dict[str, Any] = {
        "ingest-data-flow": ingest_data_flow,
        "train-candidate-flow": train_candidate_flow,
        "monitoring-flow": monitoring_flow,
        "conditional-retraining-flow": conditional_retraining_flow,
    }
    identifiers: list[str] = []
    for spec in deployment_specs(config):
        deployment_id = flows[spec.flow_name].deploy(
            spec.name,
            work_pool_name=config.runtime.work_pool_name,
            work_queue_name=config.runtime.work_queue_name,
            build=False,
            push=False,
            schedule=(
                CronSchedule(cron=spec.cron, timezone=spec.timezone)
                if spec.cron is not None
                else None
            ),
            paused=spec.paused,
            tags=["ticket-router", "local", spec.flow_name],
            print_next_steps=False,
        )
        identifiers.append(str(deployment_id))
    return tuple(identifiers)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/orchestration.yaml"))
    args = parser.parse_args()
    identifiers = register_deployments(OrchestrationConfig.load(args.config))
    print("Registered deployment IDs:")
    for identifier in identifiers:
        print(identifier)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
