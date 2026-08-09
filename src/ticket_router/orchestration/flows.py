"""Prefect 3 flows for the local ticket-routing MLOps lifecycle."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from prefect import flow
from prefect.logging import get_run_logger

from ticket_router.config import Settings
from ticket_router.data.manifests import atomic_write_json
from ticket_router.db.contracts import RetrainingRun
from ticket_router.modeling.experiment_config import CandidateExperimentSettings
from ticket_router.monitoring.config import MonitoringConfig
from ticket_router.monitoring.service import MonitoringExecution
from ticket_router.orchestration.candidate import read_candidate_workflow_result
from ticket_router.orchestration.config import (
    DEFAULT_ORCHESTRATION_CONFIG_PATH,
    OrchestrationConfig,
    orchestration_configuration_hash,
)
from ticket_router.orchestration.contracts import (
    CandidateWorkflowResult,
    ConditionalRetrainingResult,
)
from ticket_router.orchestration.tasks import (
    analyze_data_task,
    download_data_task,
    execute_monitoring_task,
    normalize_data_task,
    prepare_retraining_dataset_task,
    prepare_splits_task,
    recent_monitoring_statuses_task,
    register_candidate_task,
    retraining_decision_task,
    store_retraining_run_task,
    train_candidates_task,
    validate_data_task,
    verify_data_manifests_task,
)
from ticket_router.registry.config import FinalModelConfig


@flow(name="ingest-data-flow", log_prints=False)
def ingest_data_flow(
    *,
    orchestration_config_path: str = "configs/orchestration.yaml",
    project_root: str = ".",
    force: bool = False,
) -> dict[str, object]:
    """Download, normalize, validate, analyze, and split one pinned snapshot."""
    root = Path(project_root).resolve()
    config = OrchestrationConfig.load(root / orchestration_config_path)
    settings = Settings.load(root / config.paths.base_config)
    raw_manifest = download_data_task.with_options(
        retries=config.runtime.temporary_failure_retries,
        retry_delay_seconds=config.runtime.retry_delay_seconds,
    )(settings, force=force, project_root=root)
    normalization_manifest = normalize_data_task.with_options(
        retries=config.runtime.temporary_failure_retries,
        retry_delay_seconds=config.runtime.retry_delay_seconds,
    )(settings, force=force, project_root=root, wait_for=[raw_manifest])
    validation = validate_data_task(settings, project_root=root, wait_for=[normalization_manifest])
    analysis = analyze_data_task(settings, project_root=root, wait_for=[validation])
    split_manifest = prepare_splits_task(
        settings, force=force, project_root=root, wait_for=[analysis]
    )
    return {
        "raw_manifest": raw_manifest,
        "normalization_manifest": normalization_manifest,
        "validation": validation,
        "analysis": analysis,
        "split_manifest": split_manifest,
    }


@flow(name="train-candidate-flow", log_prints=False)
def train_candidate_flow(
    *,
    orchestration_config_path: str = "configs/orchestration.yaml",
    project_root: str = ".",
    processed_directory: str | None = None,
    split_manifest_path: str | None = None,
    workflow_run_id: str | None = None,
) -> CandidateWorkflowResult:
    """Train, validate, register candidate, and record gates without promotion."""
    root = Path(project_root).resolve()
    config = OrchestrationConfig.load(root / orchestration_config_path)
    settings = Settings.load(root / config.paths.base_config)
    experiment_config = CandidateExperimentSettings.load(root / config.paths.experiment_config)
    final_config = FinalModelConfig.load(root / config.paths.final_model_config)
    run_id = workflow_run_id or uuid4().hex
    summary_path = (
        root / config.paths.orchestration_output_directory / "candidates" / f"{run_id}.json"
    )
    processed = (
        Path(processed_directory).resolve()
        if processed_directory
        else root / config.paths.processed_directory
    )
    manifest = (
        Path(split_manifest_path).resolve()
        if split_manifest_path
        else processed / "split_manifest.json"
    )
    manifest_hash = verify_data_manifests_task(processed, manifest)
    orchestration_hash = orchestration_configuration_hash(config)
    if summary_path.exists():
        get_run_logger().info("reusing completed candidate workflow %s", run_id)
        return read_candidate_workflow_result(
            summary_path,
            expected_dataset_manifest_sha256=manifest_hash,
            expected_orchestration_configuration_sha256=orchestration_hash,
        )
    experiment = train_candidates_task(
        settings=settings,
        experiment_config=experiment_config,
        processed_dir=processed,
        manifest_path=manifest,
        model_artifacts_dir=root / config.paths.model_artifacts_directory / "orchestrated",
        reports_dir=root / config.paths.reports_directory / "candidate-experiments",
        leaderboard_path=root / config.paths.leaderboard,
        project_root=root,
        experiment_run_id=run_id,
        wait_for=[manifest_hash],
    )
    return register_candidate_task.with_options(
        retries=config.runtime.temporary_failure_retries,
        retry_delay_seconds=config.runtime.retry_delay_seconds,
    )(
        experiment=experiment,
        final_config=final_config,
        workflow_run_id=run_id,
        dataset_manifest_sha256=manifest_hash,
        orchestration_configuration_sha256=orchestration_hash,
        summary_path=summary_path,
    )


@flow(name="monitoring-flow", log_prints=False)
def monitoring_flow(
    *,
    orchestration_config_path: str = "configs/orchestration.yaml",
    project_root: str = ".",
    start: datetime | None = None,
    end: datetime | None = None,
    run_id: str | None = None,
) -> MonitoringExecution:
    """Run drift and delayed-label quality monitoring and return its health state."""
    root = Path(project_root).resolve()
    config = OrchestrationConfig.load(root / orchestration_config_path)
    settings = Settings.load(root / config.paths.base_config)
    monitoring_config = MonitoringConfig.load(root / config.paths.monitoring_config)
    window_end = end or datetime.now(UTC)
    window_start = start or window_end - timedelta(
        days=monitoring_config.current_window.default_lookback_days
    )
    return execute_monitoring_task.with_options(
        retries=config.runtime.temporary_failure_retries,
        retry_delay_seconds=config.runtime.retry_delay_seconds,
    )(
        settings=settings,
        config=monitoring_config,
        start=window_start,
        end=window_end,
        output_root=root / config.paths.monitoring_output_directory,
        run_id=run_id,
    )


@flow(name="conditional-retraining-flow", log_prints=False)
def conditional_retraining_flow(
    *,
    orchestration_config_path: str = "configs/orchestration.yaml",
    project_root: str = ".",
    start: datetime | None = None,
    end: datetime | None = None,
    manual_trigger: bool = False,
    run_id: str | None = None,
) -> ConditionalRetrainingResult:
    """Propose retraining, register a passing candidate, and stop before promotion."""
    root = Path(project_root).resolve()
    config = OrchestrationConfig.load(root / orchestration_config_path)
    settings = Settings.load(root / config.paths.base_config)
    identifier = run_id or str(uuid4())
    window_end = end or datetime.now(UTC)
    monitoring_config = MonitoringConfig.load(root / config.paths.monitoring_config)
    window_start = start or window_end - timedelta(
        days=monitoring_config.current_window.default_lookback_days
    )
    monitoring = monitoring_flow(
        orchestration_config_path=orchestration_config_path,
        project_root=project_root,
        start=window_start,
        end=window_end,
        run_id=identifier,
    )
    recent = recent_monitoring_statuses_task.with_options(
        retries=config.runtime.temporary_failure_retries,
        retry_delay_seconds=config.runtime.retry_delay_seconds,
    )(
        settings,
        limit=config.retraining.required_consecutive_critical_windows,
    )
    prior_statuses = recent[1:] if recent and recent[0] == monitoring.decision.status else recent
    decision = retraining_decision_task(
        monitoring_summary_path=monitoring.summary_path,
        recent_statuses=prior_statuses,
        config=config,
        manual_trigger=manual_trigger,
    )
    if not decision.should_retrain:
        result = ConditionalRetrainingResult(
            run_id=identifier,
            status=(
                "insufficient_feedback"
                if decision.feedback_count < config.retraining.minimum_new_feedback_labels
                else "not_triggered"
            ),
            decision=decision,
            dataset_manifest_path=None,
            candidate=None,
        )
        _write_conditional_summary(root, config, result)
        return result

    started_at = datetime.now(UTC)
    store_task = store_retraining_run_task.with_options(
        retries=config.runtime.temporary_failure_retries,
        retry_delay_seconds=config.runtime.retry_delay_seconds,
    )
    store_task(
        settings,
        RetrainingRun(
            run_id=identifier,
            trigger_reason="; ".join(decision.reasons),
            source_data_period_start=window_start,
            source_data_period_end=window_end,
            status="preparing_data",
            mlflow_run_id=None,
            candidate_model_version=None,
            gate_results={},
            started_at=started_at,
            completed_at=None,
        ),
    )
    dataset_id = f"retrain-{identifier}"
    try:
        candidate_manifest_path = prepare_retraining_dataset_task(
            config=config,
            settings=settings,
            dataset_id=dataset_id,
            start=window_start,
            end=window_end,
            feedback_label_count=decision.feedback_count,
            orchestration_configuration_sha256=orchestration_configuration_hash(config),
            project_root=root,
        )
        candidate_manifest = Path(candidate_manifest_path)
        candidate = train_candidate_flow(
            orchestration_config_path=orchestration_config_path,
            project_root=project_root,
            processed_directory=str(candidate_manifest.parent / "processed"),
            split_manifest_path=str(candidate_manifest),
            workflow_run_id=dataset_id,
        )
    except Exception as exc:
        store_task(
            settings,
            RetrainingRun(
                run_id=identifier,
                trigger_reason="; ".join(decision.reasons),
                source_data_period_start=window_start,
                source_data_period_end=window_end,
                status="failed",
                mlflow_run_id=None,
                candidate_model_version=None,
                gate_results={"error_type": type(exc).__name__},
                started_at=started_at,
                completed_at=datetime.now(UTC),
            ),
        )
        failed = ConditionalRetrainingResult(
            run_id=identifier,
            status="failed",
            decision=decision,
            dataset_manifest_path=None,
            candidate=None,
        )
        _write_conditional_summary(root, config, failed)
        raise
    store_task(
        settings,
        RetrainingRun(
            run_id=identifier,
            trigger_reason="; ".join(decision.reasons),
            source_data_period_start=window_start,
            source_data_period_end=window_end,
            status="candidate_registered",
            mlflow_run_id=candidate.mlflow_run_id,
            candidate_model_version=candidate.candidate_model_version,
            gate_results={"promotion_gates_passed": candidate.promotion_gates_passed},
            started_at=started_at,
            completed_at=datetime.now(UTC),
        ),
    )
    result = ConditionalRetrainingResult(
        run_id=identifier,
        status="candidate_registered",
        decision=decision,
        dataset_manifest_path=candidate_manifest_path,
        candidate=candidate,
    )
    _write_conditional_summary(root, config, result)
    return result


def _write_conditional_summary(
    root: Path, config: OrchestrationConfig, result: ConditionalRetrainingResult
) -> None:
    path = (
        root / config.paths.orchestration_output_directory / "retraining" / f"{result.run_id}.json"
    )
    atomic_write_json(path, result.to_dict())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("flow", choices=("ingest", "train-candidate", "monitor", "retraining"))
    parser.add_argument("--orchestration-config", default=str(DEFAULT_ORCHESTRATION_CONFIG_PATH))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--manual-trigger", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {"orchestration_config_path": args.orchestration_config}
    if args.flow == "ingest":
        result: object = ingest_data_flow(**common, force=args.force)
    elif args.flow == "train-candidate":
        result = train_candidate_flow(**common)
    elif args.flow == "monitor":
        result = monitoring_flow(**common)
    else:
        result = conditional_retraining_flow(**common, manual_trigger=args.manual_trigger)
    if hasattr(result, "to_dict"):
        result = result.to_dict()
    elif isinstance(result, MonitoringExecution):
        result = {
            "run_id": result.run_id,
            "status": result.decision.status,
            "summary_path": result.summary_path.as_posix(),
        }
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
