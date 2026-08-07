"""Run restrained train-only CV searches and validation-only candidate selection."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import mlflow
import numpy as np
from sklearn.model_selection import (  # type: ignore[import-untyped]
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]

from ticket_router.config import Settings
from ticket_router.data.load import ModelingDataset, load_training_split
from ticket_router.data.manifests import atomic_write_json, get_git_version
from ticket_router.data.split_manifest import SplitManifest
from ticket_router.hashing import sha256_file
from ticket_router.logging_config import configure_logging, get_logger
from ticket_router.modeling.artifacts import (
    ModelArtifactSummary,
    build_error_analysis,
    write_model_artifacts,
)
from ticket_router.modeling.baselines import pipeline_components
from ticket_router.modeling.candidates import (
    CandidateSpec,
    build_candidate_specs,
    calibrated_final_estimator,
)
from ticket_router.modeling.evaluation import (
    EvaluationResult,
    benchmark_batch_inference,
    evaluate_classifier,
)
from ticket_router.modeling.experiment_artifacts import (
    confidence_distribution,
    update_model_leaderboard,
    write_calibration_plot,
)
from ticket_router.modeling.experiment_config import (
    DEFAULT_EXPERIMENT_CONFIG_PATH,
    CandidateExperimentSettings,
    experiment_configuration_hash,
)
from ticket_router.modeling.mlflow_tracking import (
    LoggedCandidate,
    TrackingResolution,
    configure_experiment_tracking,
    log_candidate_to_mlflow,
)
from ticket_router.modeling.selection import CandidateEvidence, CandidateRanking, rank_candidates


class CandidateExperimentError(RuntimeError):
    """Raised when the candidate experiment cannot safely complete."""


@dataclass(frozen=True)
class CandidateRunRecord:
    """All fitted-model evidence and tracking identity for one family."""

    spec: CandidateSpec
    fitted_pipeline: Pipeline
    evaluation: EvaluationResult
    artifacts: ModelArtifactSummary
    best_parameters: dict[str, object]
    cv_summary: dict[str, object]
    mlflow_run: LoggedCandidate


@dataclass(frozen=True)
class CandidateExperimentResult:
    """CLI-facing summary of a completed candidate experiment."""

    experiment_run_id: str
    selected_candidate: str
    tracking: TrackingResolution
    report_directory: Path
    candidates: tuple[CandidateRunRecord, ...]
    ranking: CandidateRanking


def run_candidate_experiments(
    *,
    settings: Settings,
    experiment_config: CandidateExperimentSettings,
    processed_dir: Path,
    split_manifest_path: Path,
    model_artifacts_dir: Path,
    reports_dir: Path,
    leaderboard_path: Path,
    project_root: Path,
    experiment_run_id: str | None = None,
    clock: Callable[[], datetime] | None = None,
    availability_check: Callable[[str], bool] | None = None,
) -> CandidateExperimentResult:
    """Search on training folds, compare on validation, and log candidates to MLflow."""
    resolved_root = project_root.resolve()
    processed_dir = processed_dir.resolve()
    split_manifest_path = split_manifest_path.resolve()
    split_manifest = SplitManifest.read(split_manifest_path)
    _validate_data_lineage(split_manifest, processed_dir)
    training, validation = load_candidate_datasets(processed_dir)
    label_order = tuple(
        label for label, _ in sorted(split_manifest.label_mapping.items(), key=lambda item: item[1])
    )
    _validate_datasets(training, validation, label_order)
    configuration_hash = experiment_configuration_hash(
        experiment_config,
        random_seed=settings.random_seed,
    )
    run_id = experiment_run_id or _new_experiment_run_id(clock or _utc_now, configuration_hash)
    run_directory = model_artifacts_dir.resolve() / run_id
    report_directory = reports_dir.resolve() / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    report_directory.mkdir(parents=True, exist_ok=False)
    tracking = configure_experiment_tracking(
        settings=settings,
        experiment_config=experiment_config,
        project_root=resolved_root,
        availability_check=availability_check,
    )
    data_manifest_hash = sha256_file(split_manifest_path)
    git_version = get_git_version(resolved_root)
    lineage: dict[str, object] = {
        "experiment_run_id": run_id,
        "git_commit": git_version.commit if git_version else None,
        "git_dirty": git_version.dirty if git_version else None,
        "dataset_manifest_sha256": data_manifest_hash,
        "training_data_sha256": split_manifest.output_files["train"].sha256,
        "validation_data_sha256": split_manifest.output_files["validation"].sha256,
        "configuration_sha256": configuration_hash,
        "random_seed": settings.random_seed,
        "training_row_count": training.target.len(),
        "validation_row_count": validation.target.len(),
        "test_evaluated": False,
        "requested_tracking_uri": tracking.requested_uri,
        "resolved_tracking_uri": tracking.resolved_uri,
        "local_tracking_fallback_used": tracking.local_fallback_used,
    }
    training_texts = training.features["model_text"].to_list()
    training_labels = training.target.to_list()
    validation_texts = validation.features["model_text"].to_list()
    validation_labels = validation.target.to_list()
    records: list[CandidateRunRecord] = []
    for spec in build_candidate_specs(experiment_config.search, random_seed=settings.random_seed):
        records.append(
            _run_candidate(
                spec=spec,
                experiment_config=experiment_config,
                random_seed=settings.random_seed,
                run_id=run_id,
                run_directory=run_directory,
                training_texts=training_texts,
                training_labels=training_labels,
                validation_texts=validation_texts,
                validation_labels=validation_labels,
                label_order=label_order,
                lineage=lineage,
                tracking=tracking,
            )
        )
    evidence = tuple(_candidate_evidence(record) for record in records)
    ranking = rank_candidates(evidence, settings=experiment_config.selection)
    decision_by_name = {decision.candidate.name: decision for decision in ranking.decisions}
    client = mlflow.MlflowClient()
    for record in records:
        decision = decision_by_name[record.spec.name]
        client.set_tag(
            record.mlflow_run.run_id,
            "selection_status",
            "selected"
            if record.spec.name == ranking.selected_candidate
            else ("eligible_not_selected" if decision.eligible else "rejected"),
        )
        client.set_tag(
            record.mlflow_run.run_id,
            "selection_reasons",
            "none" if not decision.rejection_reasons else " | ".join(decision.rejection_reasons),
        )
    _write_experiment_reports(
        report_directory=report_directory,
        run_id=run_id,
        records=tuple(records),
        ranking=ranking,
        validation_texts=validation_texts,
        validation_labels=validation_labels,
        label_order=label_order,
        experiment_config=experiment_config,
        lineage=lineage,
        tracking=tracking,
    )
    _update_leaderboard(
        leaderboard_path.resolve(),
        run_id=run_id,
        records=tuple(records),
        ranking=ranking,
        data_manifest_hash=data_manifest_hash,
        configuration_hash=configuration_hash,
        project_root=resolved_root,
    )
    return CandidateExperimentResult(
        experiment_run_id=run_id,
        selected_candidate=ranking.selected_candidate,
        tracking=tracking,
        report_directory=report_directory,
        candidates=tuple(records),
        ranking=ranking,
    )


def load_candidate_datasets(processed_dir: Path) -> tuple[ModelingDataset, ModelingDataset]:
    """Central testable boundary that can never request the sealed test split."""
    return (
        load_training_split(processed_dir, "train"),
        load_training_split(processed_dir, "validation"),
    )


def _run_candidate(
    *,
    spec: CandidateSpec,
    experiment_config: CandidateExperimentSettings,
    random_seed: int,
    run_id: str,
    run_directory: Path,
    training_texts: list[str],
    training_labels: list[str],
    validation_texts: list[str],
    validation_labels: list[str],
    label_order: tuple[str, ...],
    lineage: dict[str, object],
    tracking: TrackingResolution,
) -> CandidateRunRecord:
    cross_validation = StratifiedKFold(
        n_splits=experiment_config.search.cv_folds,
        shuffle=True,
        random_state=random_seed,
    )
    search = RandomizedSearchCV(
        estimator=spec.estimator,
        param_distributions=spec.parameter_distributions,
        n_iter=experiment_config.search.iterations_per_candidate,
        scoring="f1_macro",
        n_jobs=experiment_config.search.n_jobs,
        cv=cross_validation,
        refit=True,
        random_state=random_seed,
        return_train_score=False,
        error_score="raise",
    )
    started = perf_counter()
    search.fit(training_texts, training_labels)
    fitted_pipeline = calibrated_final_estimator(
        spec,
        cast(Pipeline, search.best_estimator_),
        calibration_cv_folds=experiment_config.search.calibration_cv_folds,
        n_jobs=experiment_config.search.n_jobs,
    )
    if spec.requires_calibration:
        fitted_pipeline.fit(training_texts, training_labels)
    training_duration = perf_counter() - started
    evaluation = evaluate_classifier(
        fitted_pipeline,
        validation_texts=validation_texts,
        validation_labels=validation_labels,
        label_order=label_order,
    )
    benchmark = benchmark_batch_inference(
        fitted_pipeline,
        validation_texts,
        batch_size=experiment_config.evaluation.inference_batch_size,
        repeats=experiment_config.evaluation.inference_repeats,
    )
    best_parameters = {str(key): _json_value(value) for key, value in search.best_params_.items()}
    cv_summary = _cross_validation_summary(search, spec.name)
    artifacts = write_model_artifacts(
        run_directory=run_directory,
        model_name=spec.artifact_slug,
        pipeline=fitted_pipeline,
        result=evaluation,
        validation_texts=validation_texts,
        validation_labels=validation_labels,
        label_order=label_order,
        model_configuration={
            "candidate_name": spec.name,
            "model_family": spec.family,
            "pipeline_class": type(fitted_pipeline).__name__,
            "pipeline_components": list(pipeline_components(fitted_pipeline)),
            "component_types": {
                name: type(component).__name__ for name, component in fitted_pipeline.steps
            },
            "best_parameters": best_parameters,
            "search_iterations": experiment_config.search.iterations_per_candidate,
            "cross_validation_folds": experiment_config.search.cv_folds,
            "probability_calibrated": spec.requires_calibration,
        },
        training_duration_seconds=training_duration,
        inference_benchmark=benchmark,
        lineage={**lineage, "candidate_name": spec.name},
        error_sample_size=experiment_config.evaluation.error_sample_size,
        confused_pair_count=experiment_config.evaluation.confused_pair_count,
    )
    atomic_write_json(artifacts.artifact_directory / "cross_validation_summary.json", cv_summary)
    mlflow_metrics = {
        **{f"validation_{key}": value for key, value in artifacts.metrics.items()},
        "cv_macro_f1_mean": _as_float(cv_summary["best_mean_macro_f1"]),
        "cv_macro_f1_standard_deviation": _as_float(cv_summary["best_std_macro_f1"]),
    }
    logged = log_candidate_to_mlflow(
        experiment_name=experiment_config.experiment_name,
        run_name=f"{run_id}-{spec.name}",
        pipeline=fitted_pipeline,
        parameters={
            "candidate_name": spec.name,
            "model_family": spec.family,
            "cv_folds": experiment_config.search.cv_folds,
            "search_iterations": experiment_config.search.iterations_per_candidate,
            **best_parameters,
        },
        metrics=mlflow_metrics,
        per_class_metrics=evaluation.per_class_metrics,
        tags={
            "stage": "stage6_candidate_experimentation",
            "model_family": spec.family,
            "candidate_name": spec.name,
            "primary_metric": experiment_config.primary_metric,
            "test_evaluated": "false",
            "git_commit": str(lineage["git_commit"] or "unavailable"),
            "dataset_manifest_sha256": str(lineage["dataset_manifest_sha256"]),
            "training_data_sha256": str(lineage["training_data_sha256"]),
            "configuration_sha256": str(lineage["configuration_sha256"]),
            "local_tracking_fallback_used": str(tracking.local_fallback_used).lower(),
        },
        lineage={**lineage, "candidate_name": spec.name},
        cross_validation_summary=cv_summary,
        local_artifact_directory=artifacts.artifact_directory,
    )
    return CandidateRunRecord(
        spec=spec,
        fitted_pipeline=fitted_pipeline,
        evaluation=evaluation,
        artifacts=artifacts,
        best_parameters=best_parameters,
        cv_summary=cv_summary,
        mlflow_run=logged,
    )


def _cross_validation_summary(
    search: RandomizedSearchCV,
    candidate_name: str,
) -> dict[str, object]:
    results = cast(Mapping[str, Any], search.cv_results_)
    trials = [
        {
            "rank": int(results["rank_test_score"][index]),
            "mean_macro_f1": float(results["mean_test_score"][index]),
            "std_macro_f1": float(results["std_test_score"][index]),
            "parameters": {
                str(key): _json_value(value)
                for key, value in cast(dict[str, object], results["params"][index]).items()
            },
        }
        for index in range(len(results["params"]))
    ]
    trials.sort(key=lambda row: (_as_int(row["rank"]), -_as_float(row["mean_macro_f1"])))
    best_index = int(search.best_index_)
    return {
        "candidate_name": candidate_name,
        "scoring": "f1_macro",
        "best_mean_macro_f1": float(results["mean_test_score"][best_index]),
        "best_std_macro_f1": float(results["std_test_score"][best_index]),
        "best_parameters": {
            str(key): _json_value(value) for key, value in search.best_params_.items()
        },
        "trials": trials,
    }


def _candidate_evidence(record: CandidateRunRecord) -> CandidateEvidence:
    return CandidateEvidence(
        name=record.spec.name,
        metrics=record.evaluation.metrics,
        per_class_metrics=record.evaluation.per_class_metrics,
        cv_macro_f1_mean=_as_float(record.cv_summary["best_mean_macro_f1"]),
        cv_macro_f1_standard_deviation=_as_float(record.cv_summary["best_std_macro_f1"]),
        inference_milliseconds_per_record=float(
            record.artifacts.inference_benchmark["median_milliseconds_per_record"]
        ),
        serialized_model_size_bytes=record.artifacts.serialized_model_size_bytes,
    )


def _write_experiment_reports(
    *,
    report_directory: Path,
    run_id: str,
    records: tuple[CandidateRunRecord, ...],
    ranking: CandidateRanking,
    validation_texts: list[str],
    validation_labels: list[str],
    label_order: tuple[str, ...],
    experiment_config: CandidateExperimentSettings,
    lineage: dict[str, object],
    tracking: TrackingResolution,
) -> None:
    decision_by_name = {decision.candidate.name: decision for decision in ranking.decisions}
    comparison = {
        "experiment_run_id": run_id,
        "primary_metric": experiment_config.primary_metric,
        "selected_candidate": ranking.selected_candidate,
        "test_evaluated": False,
        "selection_guardrails": experiment_config.selection.model_dump(mode="json"),
        "candidates": [
            {
                "candidate_name": record.spec.name,
                "model_family": record.spec.family,
                "mlflow_run_id": record.mlflow_run.run_id,
                "model_uri": record.mlflow_run.model_uri,
                "metrics": record.artifacts.metrics,
                "best_parameters": record.best_parameters,
                "cv_macro_f1_mean": record.cv_summary["best_mean_macro_f1"],
                "cv_macro_f1_standard_deviation": record.cv_summary["best_std_macro_f1"],
                "eligible": decision_by_name[record.spec.name].eligible,
                "rejection_reasons": list(decision_by_name[record.spec.name].rejection_reasons),
            }
            for record in records
        ],
    }
    atomic_write_json(report_directory / "candidate_comparison.json", comparison)
    atomic_write_json(
        report_directory / "cross_validation_summary.json",
        {record.spec.name: record.cv_summary for record in records},
    )
    atomic_write_json(
        report_directory / "validation_error_analysis.json",
        {
            record.spec.name: build_error_analysis(
                record.evaluation,
                validation_texts=validation_texts,
                validation_labels=validation_labels,
                label_order=label_order,
                sample_size=experiment_config.evaluation.error_sample_size,
                confused_pair_count=experiment_config.evaluation.confused_pair_count,
            )
            for record in records
        },
    )
    confidence = {
        record.spec.name: confidence_distribution(
            record.evaluation,
            actual_labels=validation_labels,
            bins=experiment_config.evaluation.confidence_bins,
        )
        for record in records
    }
    atomic_write_json(report_directory / "confidence_distribution.json", confidence)
    calibration_points = write_calibration_plot(
        {record.spec.name: record.evaluation for record in records},
        actual_labels=validation_labels,
        bins=experiment_config.evaluation.confidence_bins,
        path=report_directory / "calibration_plot.png",
    )
    atomic_write_json(report_directory / "calibration_points.json", calibration_points)
    atomic_write_json(
        report_directory / "mlflow_run_comparison.json",
        {
            "experiment_name": experiment_config.experiment_name,
            "requested_tracking_uri": tracking.requested_uri,
            "resolved_tracking_uri": tracking.resolved_uri,
            "local_tracking_fallback_used": tracking.local_fallback_used,
            "runs": {
                record.spec.name: {
                    "run_id": record.mlflow_run.run_id,
                    "model_uri": record.mlflow_run.model_uri,
                }
                for record in records
            },
        },
    )
    atomic_write_json(report_directory / "experiment_lineage.json", lineage)


def _update_leaderboard(
    path: Path,
    *,
    run_id: str,
    records: tuple[CandidateRunRecord, ...],
    ranking: CandidateRanking,
    data_manifest_hash: str,
    configuration_hash: str,
    project_root: Path,
) -> None:
    decisions = {decision.candidate.name: decision for decision in ranking.decisions}
    rows = [
        {
            "stage": "stage6_candidate",
            "run_id": run_id,
            "model_name": record.spec.name,
            "primary_metric": "macro_f1",
            "macro_f1": record.evaluation.metrics["macro_f1"],
            "weighted_f1": record.evaluation.metrics["weighted_f1"],
            "accuracy": record.evaluation.metrics["accuracy"],
            "macro_precision": record.evaluation.metrics["macro_precision"],
            "macro_recall": record.evaluation.metrics["macro_recall"],
            "log_loss": record.evaluation.metrics["log_loss"],
            "cv_macro_f1_mean": record.cv_summary["best_mean_macro_f1"],
            "cv_macro_f1_standard_deviation": record.cv_summary["best_std_macro_f1"],
            "selection_eligible": decisions[record.spec.name].eligible,
            "rejection_reasons": " | ".join(decisions[record.spec.name].rejection_reasons),
            "mlflow_run_id": record.mlflow_run.run_id,
            "inference_milliseconds_per_record": record.artifacts.inference_benchmark[
                "median_milliseconds_per_record"
            ],
            "training_duration_seconds": record.artifacts.training_duration_seconds,
            "serialized_model_size_bytes": record.artifacts.serialized_model_size_bytes,
            "data_manifest_sha256": data_manifest_hash,
            "configuration_sha256": configuration_hash,
            "artifact_directory": _portable_path(
                record.artifacts.artifact_directory,
                project_root,
            ),
        }
        for record in records
    ]
    update_model_leaderboard(path, rows)


def _validate_data_lineage(manifest: SplitManifest, processed_dir: Path) -> None:
    if manifest.model_feature_columns != ("model_text",):
        raise CandidateExperimentError("Split manifest does not expose exactly model_text.")
    for split_name in ("train", "validation"):
        output = manifest.output_files.get(split_name)
        path = processed_dir / f"{split_name}.parquet"
        if output is None or not path.is_file() or sha256_file(path) != output.sha256:
            raise CandidateExperimentError(
                f"{split_name} data is missing or fails its split-manifest hash."
            )


def _validate_datasets(
    training: ModelingDataset,
    validation: ModelingDataset,
    label_order: tuple[str, ...],
) -> None:
    expected = set(label_order)
    for split_name, dataset in (("train", training), ("validation", validation)):
        observed = set(dataset.target.to_list())
        if dataset.target.is_empty() or observed != expected:
            raise CandidateExperimentError(
                f"{split_name} must be non-empty and contain every manifest label."
            )


def _json_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Expected numeric experiment value, received {type(value).__name__}.")


def _as_int(value: object) -> int:
    if isinstance(value, int):
        return value
    raise TypeError(f"Expected integer experiment value, received {type(value).__name__}.")


def _portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _new_experiment_run_id(clock: Callable[[], datetime], configuration_hash: str) -> str:
    timestamp = clock()
    if timestamp.tzinfo is None:
        raise ValueError("experiment timestamp must be timezone-aware")
    rendered = timestamp.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"cand-{rendered}-{configuration_hash[:8]}"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=DEFAULT_EXPERIMENT_CONFIG_PATH,
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/processed/split_manifest.json"),
    )
    parser.add_argument(
        "--model-artifacts-dir",
        type=Path,
        default=Path("artifacts/models/candidates"),
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("artifacts/reports/candidates"),
    )
    parser.add_argument(
        "--leaderboard",
        type=Path,
        default=Path("artifacts/reports/model_leaderboard.csv"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute candidate searches and report measured validation-only outcomes."""
    args = _parser().parse_args(argv)
    settings = Settings.load(args.config)
    experiment_config = CandidateExperimentSettings.load(args.experiment_config)
    configure_logging(settings.log_level)
    logger = get_logger(__name__)
    try:
        result = run_candidate_experiments(
            settings=settings,
            experiment_config=experiment_config,
            processed_dir=args.processed_dir,
            split_manifest_path=args.split_manifest,
            model_artifacts_dir=args.model_artifacts_dir,
            reports_dir=args.reports_dir,
            leaderboard_path=args.leaderboard,
            project_root=Path.cwd(),
        )
    except (
        CandidateExperimentError,
        ConnectionError,
        FileExistsError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        logger.error("candidate_experiment_failed", error=str(exc))
        return 1
    decisions = {decision.candidate.name: decision for decision in result.ranking.decisions}
    print(
        json.dumps(
            {
                "experiment_run_id": result.experiment_run_id,
                "tracking_uri": result.tracking.resolved_uri,
                "local_tracking_fallback_used": result.tracking.local_fallback_used,
                "selected_candidate": result.selected_candidate,
                "test_evaluated": False,
                "candidates": {
                    record.spec.name: {
                        "mlflow_run_id": record.mlflow_run.run_id,
                        "validation_metrics": record.evaluation.metrics,
                        "cv_macro_f1_mean": record.cv_summary["best_mean_macro_f1"],
                        "cv_macro_f1_standard_deviation": record.cv_summary["best_std_macro_f1"],
                        "eligible": decisions[record.spec.name].eligible,
                        "rejection_reasons": list(decisions[record.spec.name].rejection_reasons),
                    }
                    for record in result.candidates
                },
                "report_directory": str(result.report_directory),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
