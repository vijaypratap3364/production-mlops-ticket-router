"""Fit the frozen candidate, evaluate the sealed test once, and register it."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import cast

import mlflow
import mlflow.sklearn
import numpy as np
import polars as pl
from mlflow.exceptions import MlflowException
from mlflow.models import get_model_info
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]

from ticket_router.config import Settings
from ticket_router.data.load import (
    ModelingDataset,
    load_final_evaluation_split,
    load_training_split,
)
from ticket_router.data.manifests import atomic_write_json, get_git_version
from ticket_router.data.split_manifest import SplitManifest
from ticket_router.hashing import sha256_file
from ticket_router.logging_config import configure_logging, get_logger
from ticket_router.modeling.artifacts import ModelArtifactSummary, write_model_artifacts
from ticket_router.modeling.evaluation import (
    EvaluationResult,
    evaluate_classifier,
    predict_in_batches,
)
from ticket_router.modeling.final_model import build_final_pipeline, final_pipeline_parameters
from ticket_router.modeling.mlflow_tracking import (
    LoggedCandidate,
    TrackingResolution,
    configure_experiment_tracking,
    log_final_model_to_mlflow,
)
from ticket_router.registry.config import (
    DEFAULT_FINAL_MODEL_CONFIG_PATH,
    FinalModelConfig,
    final_model_configuration_hash,
)
from ticket_router.registry.contracts import (
    prediction_contract_passes,
    signature_matches_text_api,
)
from ticket_router.registry.gates import (
    PromotionDecision,
    PromotionEvidence,
    evaluate_promotion_gates,
)
from ticket_router.registry.service import ModelRegistryService, RegisteredVersion


class FinalEvaluationError(RuntimeError):
    """Raised when the one-time final-evaluation protocol cannot safely proceed."""


@dataclass(frozen=True)
class FinalEvaluationResult:
    """Complete result of the one authorized final evaluation and registration."""

    evaluation_run_id: str
    tracking: TrackingResolution
    mlflow_run: LoggedCandidate
    registered_version: RegisteredVersion
    candidate_alias: str
    champion_version: str | None
    evaluation: EvaluationResult
    artifacts: ModelArtifactSummary
    promotion_decision: PromotionDecision
    report_directory: Path


def run_final_evaluation(
    *,
    settings: Settings,
    final_config: FinalModelConfig,
    processed_dir: Path,
    split_manifest_path: Path,
    model_artifacts_dir: Path,
    reports_dir: Path,
    project_root: Path,
    clock: Callable[[], datetime] | None = None,
    availability_check: Callable[[str], bool] | None = None,
) -> FinalEvaluationResult:
    """Open the sealed test boundary once after all model and gate choices are frozen."""
    resolved_root = project_root.resolve()
    processed_dir = processed_dir.resolve()
    split_manifest_path = split_manifest_path.resolve()
    reports_dir = reports_dir.resolve()
    access_audit_path = reports_dir / "test_access_audit.json"
    if access_audit_path.exists():
        raise FinalEvaluationError(
            "Final test access has already been recorded. Refusing repeated evaluation: "
            f"{access_audit_path}"
        )
    split_manifest = SplitManifest.read(split_manifest_path)
    _validate_all_split_hashes(split_manifest, processed_dir)
    configuration_hash = final_model_configuration_hash(
        final_config,
        random_seed=settings.random_seed,
    )
    timestamp = (clock or _utc_now)()
    if timestamp.tzinfo is None:
        raise ValueError("final-evaluation timestamp must be timezone-aware")
    run_id = (
        f"final-{timestamp.astimezone(UTC).strftime('%Y%m%dT%H%M%SZ')}-{configuration_hash[:8]}"
    )
    report_directory = reports_dir / run_id
    model_directory = model_artifacts_dir.resolve() / run_id
    report_directory.mkdir(parents=True, exist_ok=False)
    model_directory.mkdir(parents=True, exist_ok=False)
    tracking = configure_experiment_tracking(
        settings=settings,
        experiment_config=final_config,
        project_root=resolved_root,
        availability_check=availability_check,
    )
    training = load_training_split(processed_dir, "train")
    validation = load_training_split(processed_dir, "validation")
    combined = _combine_training_and_validation(training, validation)
    label_order = tuple(
        label for label, _ in sorted(split_manifest.label_mapping.items(), key=lambda item: item[1])
    )
    combined_texts = combined.features["model_text"].to_list()
    combined_labels = combined.target.to_list()
    if set(combined_labels) != set(label_order):
        raise FinalEvaluationError(
            "Combined training data does not contain the frozen label space."
        )
    pipeline = build_final_pipeline(
        final_config.selected_candidate,
        random_seed=settings.random_seed,
    )
    fit_started = perf_counter()
    pipeline.fit(combined_texts, combined_labels)
    training_duration = perf_counter() - fit_started
    split_manifest_hash = sha256_file(split_manifest_path)
    git_version = get_git_version(resolved_root)
    combined_training_hash = _combined_training_hash(split_manifest)
    access_record: dict[str, object] = {
        "status": "authorized_and_opened",
        "test_access_timestamp_utc": timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "evaluation_run_id": run_id,
        "selected_candidate": final_config.selected_candidate.name,
        "stage6_run_id": final_config.selected_candidate.stage6_run_id,
        "stage6_configuration_sha256": (
            final_config.selected_candidate.stage6_configuration_sha256
        ),
        "final_configuration_sha256": configuration_hash,
        "split_manifest_sha256": split_manifest_hash,
        "test_data_sha256": split_manifest.output_files["test"].sha256,
        "git_commit": git_version.commit if git_version else None,
        "git_dirty": git_version.dirty if git_version else None,
        "repeated_test_evaluation_allowed": False,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(access_audit_path, access_record)
    test = load_final_evaluation_split(processed_dir, final_evaluation_authorized=True)
    test_texts = test.features["model_text"].to_list()
    test_labels = test.target.to_list()
    evaluation = evaluate_classifier(
        pipeline,
        validation_texts=test_texts,
        validation_labels=test_labels,
        label_order=label_order,
    )
    latency_distribution = _benchmark_latency_distribution(
        pipeline,
        test_texts,
        batch_size=final_config.evaluation.inference_batch_size,
        repeats=final_config.evaluation.inference_repeats,
    )
    benchmark = {
        "records": len(test_texts),
        "batch_size": final_config.evaluation.inference_batch_size,
        "repeats": final_config.evaluation.inference_repeats,
        "median_total_seconds": cast(float, latency_distribution["median_total_seconds"]),
        "median_milliseconds_per_record": cast(
            float, latency_distribution["median_milliseconds_per_record"]
        ),
        "median_records_per_second": cast(float, latency_distribution["median_records_per_second"]),
    }
    lineage: dict[str, object] = {
        **access_record,
        "combined_training_data_sha256": combined_training_hash,
        "training_data_sha256": split_manifest.output_files["train"].sha256,
        "validation_data_sha256": split_manifest.output_files["validation"].sha256,
        "combined_training_row_count": combined.target.len(),
        "test_row_count": test.target.len(),
        "random_seed": settings.random_seed,
        "requested_tracking_uri": tracking.requested_uri,
        "resolved_tracking_uri": tracking.resolved_uri,
        "local_tracking_fallback_used": tracking.local_fallback_used,
        "test_evaluated": True,
    }
    artifacts = write_model_artifacts(
        run_directory=model_directory,
        model_name="final_model",
        pipeline=pipeline,
        result=evaluation,
        validation_texts=test_texts,
        validation_labels=test_labels,
        label_order=label_order,
        model_configuration={
            "selected_candidate": final_config.selected_candidate.model_dump(mode="json"),
            "pipeline_parameters": final_pipeline_parameters(final_config.selected_candidate),
            "fit_data": "train_plus_validation",
        },
        training_duration_seconds=training_duration,
        inference_benchmark=benchmark,
        lineage=lineage,
        error_sample_size=final_config.evaluation.error_sample_size,
        confused_pair_count=final_config.evaluation.confused_pair_count,
        evaluation_split="test",
    )
    atomic_write_json(
        artifacts.artifact_directory / "latency_distribution.json", latency_distribution
    )
    mlflow_metrics = {f"test_{key}": value for key, value in artifacts.metrics.items()}
    logged = log_final_model_to_mlflow(
        experiment_name=final_config.experiment_name,
        run_name=run_id,
        pipeline=pipeline,
        parameters=final_pipeline_parameters(final_config.selected_candidate),
        metrics=mlflow_metrics,
        per_class_metrics=evaluation.per_class_metrics,
        tags={
            "stage": "stage7_final_evaluation",
            "selected_candidate": final_config.selected_candidate.name,
            "test_evaluated": "true",
            "git_commit": str(lineage["git_commit"] or "unavailable"),
            "split_manifest_sha256": split_manifest_hash,
            "combined_training_data_sha256": combined_training_hash,
            "test_data_sha256": split_manifest.output_files["test"].sha256,
            "configuration_sha256": configuration_hash,
        },
        lineage=lineage,
        local_artifact_directory=artifacts.artifact_directory,
    )
    load_succeeded, contract_passed, signature_compatible = _verify_logged_model(
        logged.model_uri,
        labels=label_order,
        final_config=final_config,
    )
    registry = ModelRegistryService()
    champion_before = registry.resolve_alias(
        name=final_config.registry.model_name,
        alias=final_config.registry.champion_alias,
    )
    champion_macro_f1 = _champion_macro_f1(registry, champion_before)
    minimum_recall = min(float(row["recall"]) for row in evaluation.per_class_metrics)
    evidence = PromotionEvidence(
        macro_f1=evaluation.metrics["macro_f1"],
        minimum_per_class_recall=minimum_recall,
        inference_milliseconds_per_record=float(
            cast(float, latency_distribution["median_milliseconds_per_record"])
        ),
        model_load_succeeded=load_succeeded,
        prediction_contract_passed=contract_passed,
        signature_compatible=signature_compatible,
    )
    version_tags = _registry_tags(
        lineage=lineage,
        evidence=evidence,
        configuration_hash=configuration_hash,
        artifact_directory=artifacts.artifact_directory,
    )
    registered = registry.register_candidate(
        name=final_config.registry.model_name,
        model_uri=logged.model_uri,
        run_id=logged.run_id,
        candidate_alias=final_config.registry.candidate_alias,
        tags=version_tags,
    )
    decision = evaluate_promotion_gates(
        evidence,
        settings=final_config.promotion,
        champion_macro_f1=champion_macro_f1,
    )
    gate_report = {
        "registered_model_name": registered.name,
        "candidate_version": registered.version,
        "candidate_alias": final_config.registry.candidate_alias,
        "champion_version_before_evaluation": (
            champion_before.version if champion_before is not None else None
        ),
        "promotion": decision.to_dict(),
        "human_promotion_required": True,
        "promotion_command": "uv run python -m ticket_router.registry.promote --approve",
    }
    atomic_write_json(report_directory / "promotion_gate_results.json", gate_report)
    atomic_write_json(
        report_directory / "final_evaluation_summary.json",
        {
            **lineage,
            "mlflow_run_id": logged.run_id,
            "model_uri": logged.model_uri,
            "registered_model_name": registered.name,
            "registered_model_version": registered.version,
            "candidate_alias": final_config.registry.candidate_alias,
            "champion_alias": None,
            "metrics": artifacts.metrics,
            "minimum_per_class_recall": minimum_recall,
            "promotion_allowed": decision.allowed,
        },
    )
    registry.set_model_version_tags(
        name=registered.name,
        version=registered.version,
        tags={
            "promotion_gates_passed": str(decision.allowed).lower(),
            "candidate_alias": final_config.registry.candidate_alias,
        },
    )
    mlflow.MlflowClient().set_tag(logged.run_id, "registered_model_version", registered.version)
    mlflow.MlflowClient().set_tag(
        logged.run_id, "promotion_gates_passed", str(decision.allowed).lower()
    )
    atomic_write_json(
        access_audit_path,
        {
            **access_record,
            "status": "completed",
            "mlflow_run_id": logged.run_id,
            "registered_model_version": registered.version,
            "test_macro_f1": evaluation.metrics["macro_f1"],
        },
    )
    return FinalEvaluationResult(
        evaluation_run_id=run_id,
        tracking=tracking,
        mlflow_run=logged,
        registered_version=registered,
        candidate_alias=final_config.registry.candidate_alias,
        champion_version=champion_before.version if champion_before is not None else None,
        evaluation=evaluation,
        artifacts=artifacts,
        promotion_decision=decision,
        report_directory=report_directory,
    )


def _combine_training_and_validation(
    training: ModelingDataset,
    validation: ModelingDataset,
) -> ModelingDataset:
    return ModelingDataset(
        features=pl.concat([training.features, validation.features], how="vertical"),
        target=pl.concat([training.target, validation.target]),
    )


def _validate_all_split_hashes(manifest: SplitManifest, processed_dir: Path) -> None:
    for split in ("train", "validation", "test"):
        expected = manifest.output_files[split]
        path = processed_dir / f"{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Required prepared split is missing: {path}")
        observed = sha256_file(path)
        if observed != expected.sha256:
            raise FinalEvaluationError(f"{split} split hash does not match split manifest")


def _combined_training_hash(manifest: SplitManifest) -> str:
    payload = (
        f"train:{manifest.output_files['train'].sha256};"
        f"validation:{manifest.output_files['validation'].sha256}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _benchmark_latency_distribution(
    pipeline: Pipeline,
    texts: list[str],
    *,
    batch_size: int,
    repeats: int,
) -> dict[str, object]:
    pipeline.predict(texts[: min(batch_size, len(texts))])
    totals: list[float] = []
    milliseconds_per_record: list[float] = []
    for _ in range(repeats):
        started = perf_counter()
        predictions = predict_in_batches(pipeline, texts, batch_size=batch_size)
        elapsed = perf_counter() - started
        if len(predictions) != len(texts):
            raise FinalEvaluationError("Latency benchmark returned the wrong prediction count.")
        totals.append(elapsed)
        milliseconds_per_record.append(elapsed * 1000.0 / len(texts))
    values = np.asarray(milliseconds_per_record, dtype=np.float64)
    median_total = float(np.median(np.asarray(totals, dtype=np.float64)))
    return {
        "records": len(texts),
        "batch_size": batch_size,
        "repeats": repeats,
        "total_seconds_by_repeat": totals,
        "milliseconds_per_record_by_repeat": milliseconds_per_record,
        "minimum_milliseconds_per_record": float(values.min()),
        "median_milliseconds_per_record": float(np.median(values)),
        "p95_milliseconds_per_record": float(np.percentile(values, 95)),
        "p99_milliseconds_per_record": float(np.percentile(values, 99)),
        "maximum_milliseconds_per_record": float(values.max()),
        "median_total_seconds": median_total,
        "median_records_per_second": len(texts) / median_total,
    }


def _verify_logged_model(
    model_uri: str,
    *,
    labels: tuple[str, ...],
    final_config: FinalModelConfig,
) -> tuple[bool, bool, bool]:
    try:
        loaded = mlflow.sklearn.load_model(model_uri)
        load_succeeded = True
    except (MlflowException, OSError, ValueError):
        return False, False, False
    contract_passed = prediction_contract_passes(loaded, labels=labels)
    info = get_model_info(model_uri)
    signature_compatible = signature_matches_text_api(
        info.signature,
        required_input_dtype=final_config.promotion.required_input_dtype,
        required_output_dtype=final_config.promotion.required_output_dtype,
    )
    return load_succeeded, contract_passed, signature_compatible


def _champion_macro_f1(
    registry: ModelRegistryService,
    champion: RegisteredVersion | None,
) -> float | None:
    if champion is None:
        return None
    value = registry.model_version_tags(name=champion.name, version=champion.version).get(
        "test_macro_f1"
    )
    if value is None:
        raise FinalEvaluationError("Existing champion lacks required test_macro_f1 lineage tag.")
    return float(value)


def _registry_tags(
    *,
    lineage: dict[str, object],
    evidence: PromotionEvidence,
    configuration_hash: str,
    artifact_directory: Path,
) -> dict[str, str]:
    return {
        "stage": "stage7_final_evaluation",
        "test_macro_f1": str(evidence.macro_f1),
        "minimum_per_class_recall": str(evidence.minimum_per_class_recall),
        "inference_milliseconds_per_record": str(evidence.inference_milliseconds_per_record),
        "model_load_succeeded": str(evidence.model_load_succeeded).lower(),
        "prediction_contract_passed": str(evidence.prediction_contract_passed).lower(),
        "signature_compatible": str(evidence.signature_compatible).lower(),
        "split_manifest_sha256": str(lineage["split_manifest_sha256"]),
        "combined_training_data_sha256": str(lineage["combined_training_data_sha256"]),
        "test_data_sha256": str(lineage["test_data_sha256"]),
        "configuration_sha256": configuration_hash,
        "git_commit": str(lineage["git_commit"] or "unavailable"),
        "evaluation_artifact_directory": artifact_directory.as_posix(),
    }


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--final-config", type=Path, default=DEFAULT_FINAL_MODEL_CONFIG_PATH)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--split-manifest", type=Path, default=Path("data/processed/split_manifest.json")
    )
    parser.add_argument("--model-artifacts-dir", type=Path, default=Path("artifacts/models/final"))
    parser.add_argument(
        "--reports-dir", type=Path, default=Path("artifacts/reports/final_evaluation")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = Settings.load(args.config)
    final_config = FinalModelConfig.load(args.final_config)
    configure_logging(settings.log_level)
    logger = get_logger(__name__)
    try:
        result = run_final_evaluation(
            settings=settings,
            final_config=final_config,
            processed_dir=args.processed_dir,
            split_manifest_path=args.split_manifest,
            model_artifacts_dir=args.model_artifacts_dir,
            reports_dir=args.reports_dir,
            project_root=Path.cwd(),
        )
    except (
        ConnectionError,
        FileExistsError,
        FileNotFoundError,
        FinalEvaluationError,
        ValueError,
    ) as exc:
        logger.error("final_evaluation_failed", error=str(exc))
        return 1
    print(
        json.dumps(
            {
                "evaluation_run_id": result.evaluation_run_id,
                "mlflow_run_id": result.mlflow_run.run_id,
                "registered_model_name": result.registered_version.name,
                "registered_model_version": result.registered_version.version,
                "aliases": {
                    "candidate": result.registered_version.version,
                    "champion": result.champion_version,
                },
                "test_metrics": result.artifacts.metrics,
                "promotion_gates": result.promotion_decision.to_dict(),
                "test_evaluated": True,
                "repeated_test_evaluation_allowed": False,
                "report_directory": str(result.report_directory),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
