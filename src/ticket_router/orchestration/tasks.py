"""Discrete Prefect tasks used by the Stage 11 flows."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl
from prefect import task
from prefect.logging import get_run_logger

from ticket_router.config import Settings
from ticket_router.data.analyze import run_analysis
from ticket_router.data.download import configuration_hash, download_dataset
from ticket_router.data.manifests import NormalizationManifest
from ticket_router.data.normalize import normalize_dataset
from ticket_router.data.prepare import prepare_dataset
from ticket_router.data.split_manifest import SplitManifest
from ticket_router.data.validation import validate_normalized_frame
from ticket_router.db.contracts import RetrainingRun
from ticket_router.db.repositories import (
    SQLAlchemyMonitoringDataRepository,
    SQLAlchemyMonitoringRunRepository,
    SQLAlchemyRetrainingRunRepository,
)
from ticket_router.db.session import DatabaseSessionManager
from ticket_router.hashing import sha256_file
from ticket_router.modeling.experiment_config import CandidateExperimentSettings
from ticket_router.modeling.train_candidates import (
    CandidateExperimentResult,
    run_candidate_experiments,
)
from ticket_router.monitoring.config import MonitoringConfig
from ticket_router.monitoring.service import MonitoringExecution, execute_monitoring_run
from ticket_router.orchestration.candidate import register_and_gate_candidate
from ticket_router.orchestration.config import OrchestrationConfig
from ticket_router.orchestration.contracts import CandidateWorkflowResult, RetrainingDecision
from ticket_router.orchestration.retraining import (
    evaluate_retraining_conditions,
    prepare_retraining_dataset,
)
from ticket_router.registry.config import FinalModelConfig


@task(
    name="download pinned support-ticket data", tags=["ingestion", "network"], persist_result=False
)
def download_data_task(settings: Settings, *, force: bool, project_root: Path) -> str:
    dataset = settings.project_config.dataset
    manifest = download_dataset(
        repository=dataset.repository,
        revision=dataset.revision,
        raw_dir=project_root / "data/raw",
        project_root=project_root,
        configuration_digest=configuration_hash(settings),
        force=force,
    )
    get_run_logger().info("raw data ready with %s rows", manifest.row_count)
    return (project_root / "data/raw/data_manifest.json").as_posix()


@task(
    name="normalize English ticket data", tags=["ingestion", "normalization"], persist_result=False
)
def normalize_data_task(settings: Settings, *, force: bool, project_root: Path) -> str:
    manifest = normalize_dataset(
        raw_manifest_path=project_root / "data/raw/data_manifest.json",
        interim_dir=project_root / "data/interim",
        project_root=project_root,
        language_filter=settings.project_config.dataset.language_filter,
        configuration_digest=configuration_hash(settings),
        force=force,
    )
    get_run_logger().info("normalized data ready with %s rows", manifest.output_row_count)
    return (project_root / "data/interim/normalization_manifest.json").as_posix()


@task(
    name="validate normalized data contract", tags=["ingestion", "validation"], persist_result=False
)
def validate_data_task(settings: Settings, *, project_root: Path) -> dict[str, int]:
    manifest = NormalizationManifest.read(project_root / "data/interim/normalization_manifest.json")
    frame = pl.read_parquet(project_root / "data/interim/normalized_tickets.parquet")
    result = validate_normalized_frame(
        frame,
        settings=settings.project_config.analysis,
        language_filter=settings.project_config.dataset.language_filter,
    )
    if frame.height != manifest.output_row_count:
        raise ValueError("normalized row count does not match its manifest")
    return {"valid_rows": result.valid_records, "removed_rows": result.records_removed}


@task(name="analyze data and select classes", tags=["ingestion", "analysis"], persist_result=False)
def analyze_data_task(settings: Settings, *, project_root: Path) -> dict[str, object]:
    return run_analysis(
        settings=settings,
        normalized_path=project_root / "data/interim/normalized_tickets.parquet",
        normalization_manifest_path=project_root / "data/interim/normalization_manifest.json",
        raw_manifest_path=project_root / "data/raw/data_manifest.json",
        reports_dir=project_root / "artifacts/reports",
        project_root=project_root,
    )


@task(name="prepare duplicate-safe splits", tags=["ingestion", "splitting"], persist_result=False)
def prepare_splits_task(settings: Settings, *, force: bool, project_root: Path) -> str:
    prepare_dataset(
        settings=settings,
        normalized_path=project_root / "data/interim/normalized_tickets.parquet",
        normalization_manifest_path=project_root / "data/interim/normalization_manifest.json",
        selected_classes_path=project_root / "artifacts/reports/selected_classes.json",
        processed_dir=project_root / "data/processed",
        reference_dir=project_root / "data/reference",
        reports_dir=project_root / "artifacts/reports",
        project_root=project_root,
        force=force,
    )
    return (project_root / "data/processed/split_manifest.json").as_posix()


@task(name="verify candidate data lineage", tags=["training", "lineage"], persist_result=False)
def verify_data_manifests_task(processed_dir: Path, manifest_path: Path) -> str:
    manifest = SplitManifest.read(manifest_path)
    if manifest.model_feature_columns != ("model_text",):
        raise ValueError("candidate manifest must expose only model_text")
    for split in ("train", "validation"):
        path = processed_dir / f"{split}.parquet"
        if not path.exists() or sha256_file(path) != manifest.output_files[split].sha256:
            raise ValueError(f"{split} data does not match its manifest")
    return sha256_file(manifest_path)


@task(
    name="train and evaluate validation candidates",
    tags=["training", "mlflow"],
    persist_result=False,
)
def train_candidates_task(
    *,
    settings: Settings,
    experiment_config: CandidateExperimentSettings,
    processed_dir: Path,
    manifest_path: Path,
    model_artifacts_dir: Path,
    reports_dir: Path,
    leaderboard_path: Path,
    project_root: Path,
    experiment_run_id: str,
) -> CandidateExperimentResult:
    return run_candidate_experiments(
        settings=settings,
        experiment_config=experiment_config,
        processed_dir=processed_dir,
        split_manifest_path=manifest_path,
        model_artifacts_dir=model_artifacts_dir,
        reports_dir=reports_dir,
        leaderboard_path=leaderboard_path,
        project_root=project_root,
        experiment_run_id=experiment_run_id,
    )


@task(
    name="register candidate and execute gates",
    tags=["registry", "human-approval"],
    persist_result=False,
)
def register_candidate_task(
    *,
    experiment: CandidateExperimentResult,
    final_config: FinalModelConfig,
    workflow_run_id: str,
    dataset_manifest_sha256: str,
    orchestration_configuration_sha256: str,
    summary_path: Path,
) -> CandidateWorkflowResult:
    return register_and_gate_candidate(
        experiment=experiment,
        final_config=final_config,
        workflow_run_id=workflow_run_id,
        dataset_manifest_sha256=dataset_manifest_sha256,
        orchestration_configuration_sha256=orchestration_configuration_sha256,
        summary_path=summary_path,
    )


@task(name="run batch monitoring", tags=["monitoring", "postgresql"], persist_result=False)
def execute_monitoring_task(
    *,
    settings: Settings,
    config: MonitoringConfig,
    start: datetime,
    end: datetime,
    output_root: Path,
    run_id: str | None,
) -> MonitoringExecution:
    if settings.database_url is None:
        raise ValueError("DATABASE_URL is required for the monitoring flow")
    manager = DatabaseSessionManager(settings.database_url.get_secret_value())
    try:
        manager.connect()
        return execute_monitoring_run(
            reference_path=config.reference.output_path,
            reference_manifest_path=config.reference.manifest_path,
            data_repository=SQLAlchemyMonitoringDataRepository(manager.session_factory),
            run_repository=SQLAlchemyMonitoringRunRepository(manager.session_factory),
            config=config,
            start=start,
            end=end,
            output_root=output_root,
            run_id=run_id,
        )
    finally:
        manager.close()


@task(name="load recent monitoring states", tags=["monitoring", "postgresql"], persist_result=False)
def recent_monitoring_statuses_task(settings: Settings, *, limit: int) -> tuple[str, ...]:
    if settings.database_url is None:
        return ()
    manager = DatabaseSessionManager(settings.database_url.get_secret_value())
    try:
        manager.connect()
        repository = SQLAlchemyMonitoringRunRepository(manager.session_factory)
        return tuple(run.drift_status for run in repository.list_recent(limit=limit))
    finally:
        manager.close()


@task(
    name="evaluate controlled retraining conditions",
    tags=["retraining", "policy"],
    persist_result=False,
)
def retraining_decision_task(
    *,
    monitoring_summary_path: Path,
    recent_statuses: tuple[str, ...],
    config: OrchestrationConfig,
    manual_trigger: bool,
) -> RetrainingDecision:
    summary = json.loads(monitoring_summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("monitoring summary must be a JSON object")
    return evaluate_retraining_conditions(
        monitoring_summary=summary,
        recent_statuses=recent_statuses,
        settings=config.retraining,
        manual_trigger=manual_trigger,
    )


@task(name="version approved retraining data", tags=["retraining", "lineage"], persist_result=False)
def prepare_retraining_dataset_task(
    *,
    config: OrchestrationConfig,
    settings: Settings,
    dataset_id: str,
    start: datetime,
    end: datetime,
    feedback_label_count: int,
    orchestration_configuration_sha256: str,
    project_root: Path,
) -> str:
    path = prepare_retraining_dataset(
        approved_input_path=project_root / config.paths.approved_feedback_input,
        parent_processed_dir=project_root / config.paths.processed_directory,
        parent_manifest_path=project_root
        / config.paths.processed_directory
        / "split_manifest.json",
        output_root=project_root / config.paths.retraining_output_directory,
        dataset_id=dataset_id,
        source_period_start=start,
        source_period_end=end,
        feedback_label_count=feedback_label_count,
        settings=settings,
        orchestration_configuration_sha256=orchestration_configuration_sha256,
        project_root=project_root,
    )
    return path.as_posix()


@task(name="store retraining run state", tags=["retraining", "postgresql"], persist_result=False)
def store_retraining_run_task(settings: Settings, run: RetrainingRun) -> None:
    """Persist state when PostgreSQL is configured; local fixture runs may omit it."""
    if settings.database_url is None:
        return
    manager = DatabaseSessionManager(settings.database_url.get_secret_value())
    try:
        manager.connect()
        SQLAlchemyRetrainingRunRepository(manager.session_factory).save(run)
    finally:
        manager.close()
