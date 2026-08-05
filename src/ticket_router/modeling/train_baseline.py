"""Train and compare validation-only sparse-text baselines."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import polars as pl
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]

from ticket_router.config import Settings
from ticket_router.data.load import ModelingDataset, load_training_split
from ticket_router.data.manifests import atomic_write_json, get_git_version
from ticket_router.data.split_manifest import SplitManifest
from ticket_router.hashing import sha256_file
from ticket_router.logging_config import configure_logging, get_logger
from ticket_router.modeling.artifacts import ModelArtifactSummary, write_model_artifacts
from ticket_router.modeling.baselines import build_baseline_pipelines, pipeline_components
from ticket_router.modeling.config import (
    DEFAULT_BASELINE_CONFIG_PATH,
    BaselineSettings,
    baseline_configuration_hash,
)
from ticket_router.modeling.evaluation import benchmark_batch_inference, evaluate_classifier

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class BaselineTrainingError(RuntimeError):
    """Raised when a safe, reproducible baseline run cannot be completed."""


@dataclass(frozen=True)
class BaselineRunResult:
    """Validation comparison returned by the CLI and tests."""

    run_id: str
    run_directory: Path
    strongest_model: str
    leaderboard_path: Path
    model_summaries: tuple[ModelArtifactSummary, ...]


def train_baselines(
    *,
    settings: Settings,
    baseline_config: BaselineSettings,
    processed_dir: Path,
    split_manifest_path: Path,
    artifacts_dir: Path,
    leaderboard_path: Path,
    project_root: Path,
    run_id: str | None = None,
    clock: Callable[[], datetime] | None = None,
) -> BaselineRunResult:
    """Fit on training only, evaluate on validation only, and write run artifacts."""
    resolved_root = project_root.resolve()
    processed_dir = processed_dir.resolve()
    split_manifest_path = split_manifest_path.resolve()
    artifacts_dir = artifacts_dir.resolve()
    leaderboard_path = leaderboard_path.resolve()
    split_manifest = SplitManifest.read(split_manifest_path)
    data_manifest_hash = sha256_file(split_manifest_path)
    _validate_data_lineage(split_manifest, processed_dir)

    training = load_training_split(processed_dir, "train")
    validation = load_training_split(processed_dir, "validation")
    label_order = tuple(
        label for label, _ in sorted(split_manifest.label_mapping.items(), key=lambda item: item[1])
    )
    _validate_datasets(training, validation, label_order)
    training_texts = training.features["model_text"].to_list()
    training_labels = training.target.to_list()
    validation_texts = validation.features["model_text"].to_list()
    validation_labels = validation.target.to_list()
    configuration_hash = baseline_configuration_hash(
        baseline_config,
        random_seed=settings.random_seed,
    )
    generated_run_id = run_id or _new_run_id(clock or _utc_now, configuration_hash)
    if not RUN_ID_PATTERN.fullmatch(generated_run_id):
        raise BaselineTrainingError(
            "run_id may contain only letters, numbers, underscores, periods, and hyphens."
        )
    run_directory = artifacts_dir / generated_run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    git_version = get_git_version(resolved_root)
    lineage: dict[str, object] = {
        "run_id": generated_run_id,
        "data_manifest_path": _portable_path(split_manifest_path, resolved_root),
        "data_manifest_sha256": data_manifest_hash,
        "baseline_configuration_sha256": configuration_hash,
        "random_seed": settings.random_seed,
        "training_file_sha256": split_manifest.output_files["train"].sha256,
        "validation_file_sha256": split_manifest.output_files["validation"].sha256,
        "training_row_count": training.target.len(),
        "validation_row_count": validation.target.len(),
        "test_evaluated": False,
        "code_version": git_version.model_dump(mode="json") if git_version else None,
    }
    pipelines = build_baseline_pipelines(baseline_config, random_seed=settings.random_seed)
    summaries: list[ModelArtifactSummary] = []
    for model_name, pipeline in pipelines.items():
        summaries.append(
            _train_one(
                model_name=model_name,
                pipeline=pipeline,
                baseline_config=baseline_config,
                run_directory=run_directory,
                training_texts=training_texts,
                training_labels=training_labels,
                validation_texts=validation_texts,
                validation_labels=validation_labels,
                label_order=label_order,
                lineage=lineage,
            )
        )
    ordered = tuple(
        sorted(
            summaries,
            key=lambda summary: (
                -summary.metrics[baseline_config.primary_metric],
                -summary.metrics["weighted_f1"],
                summary.model_name,
            ),
        )
    )
    strongest = ordered[0]
    _write_leaderboard(
        leaderboard_path,
        run_id=generated_run_id,
        summaries=ordered,
        primary_metric=baseline_config.primary_metric,
        data_manifest_hash=data_manifest_hash,
        configuration_hash=configuration_hash,
        project_root=resolved_root,
    )
    atomic_write_json(
        run_directory / "run_manifest.json",
        {
            **lineage,
            "primary_metric": baseline_config.primary_metric,
            "strongest_baseline": strongest.model_name,
            "model_order": [summary.model_name for summary in ordered],
            "leaderboard_path": _portable_path(leaderboard_path, resolved_root),
        },
    )
    return BaselineRunResult(
        run_id=generated_run_id,
        run_directory=run_directory,
        strongest_model=strongest.model_name,
        leaderboard_path=leaderboard_path,
        model_summaries=ordered,
    )


def _train_one(
    *,
    model_name: str,
    pipeline: Pipeline,
    baseline_config: BaselineSettings,
    run_directory: Path,
    training_texts: list[str],
    training_labels: list[str],
    validation_texts: list[str],
    validation_labels: list[str],
    label_order: tuple[str, ...],
    lineage: dict[str, object],
) -> ModelArtifactSummary:
    started = perf_counter()
    pipeline.fit(training_texts, training_labels)
    training_duration = perf_counter() - started
    result = evaluate_classifier(
        pipeline,
        validation_texts=validation_texts,
        validation_labels=validation_labels,
        label_order=label_order,
    )
    benchmark = benchmark_batch_inference(
        pipeline,
        validation_texts,
        batch_size=baseline_config.evaluation.inference_batch_size,
        repeats=baseline_config.evaluation.inference_repeats,
    )
    return write_model_artifacts(
        run_directory=run_directory,
        model_name=model_name,
        pipeline=pipeline,
        result=result,
        validation_texts=validation_texts,
        validation_labels=validation_labels,
        label_order=label_order,
        model_configuration={
            "model_name": model_name,
            "pipeline_class": type(pipeline).__name__,
            "pipeline_components": list(pipeline_components(pipeline)),
            "component_types": {
                name: type(component).__name__ for name, component in pipeline.steps
            },
            "baseline_configuration": baseline_config.model_dump(mode="json"),
        },
        training_duration_seconds=training_duration,
        inference_benchmark=benchmark,
        lineage=lineage,
        error_sample_size=baseline_config.evaluation.error_sample_size,
        confused_pair_count=baseline_config.evaluation.confused_pair_count,
    )


def _validate_data_lineage(manifest: SplitManifest, processed_dir: Path) -> None:
    if manifest.model_feature_columns != ("model_text",):
        raise BaselineTrainingError("Split manifest does not expose exactly model_text.")
    for split_name in ("train", "validation"):
        output = manifest.output_files.get(split_name)
        path = processed_dir / f"{split_name}.parquet"
        if output is None:
            raise BaselineTrainingError(f"Split manifest is missing '{split_name}'.")
        if not path.is_file() or sha256_file(path) != output.sha256:
            raise BaselineTrainingError(
                f"{split_name} data is missing or fails its split-manifest hash."
            )


def _validate_datasets(
    training: ModelingDataset,
    validation: ModelingDataset,
    label_order: tuple[str, ...],
) -> None:
    expected = set(label_order)
    for split_name, dataset in (("train", training), ("validation", validation)):
        if dataset.target.is_empty():
            raise BaselineTrainingError(f"{split_name} split is empty.")
        observed = set(dataset.target.to_list())
        if observed != expected:
            missing = sorted(expected - observed)
            unexpected = sorted(observed - expected)
            raise BaselineTrainingError(
                f"{split_name} labels differ from the manifest; missing={missing}, "
                f"unexpected={unexpected}."
            )
        if dataset.features["model_text"].null_count():
            raise BaselineTrainingError(f"{split_name} contains null model_text values.")


def _write_leaderboard(
    path: Path,
    *,
    run_id: str,
    summaries: tuple[ModelArtifactSummary, ...],
    primary_metric: str,
    data_manifest_hash: str,
    configuration_hash: str,
    project_root: Path,
) -> None:
    rows = [
        {
            "rank": rank,
            "run_id": run_id,
            "model_name": summary.model_name,
            "primary_metric": primary_metric,
            "macro_f1": summary.metrics["macro_f1"],
            "weighted_f1": summary.metrics["weighted_f1"],
            "accuracy": summary.metrics["accuracy"],
            "macro_precision": summary.metrics["macro_precision"],
            "macro_recall": summary.metrics["macro_recall"],
            "log_loss": summary.metrics["log_loss"],
            "inference_milliseconds_per_record": summary.metrics[
                "inference_milliseconds_per_record"
            ],
            "training_duration_seconds": summary.training_duration_seconds,
            "serialized_model_size_bytes": summary.serialized_model_size_bytes,
            "data_manifest_sha256": data_manifest_hash,
            "configuration_sha256": configuration_hash,
            "artifact_directory": _portable_path(summary.artifact_directory, project_root),
        }
        for rank, summary in enumerate(summaries, start=1)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        pl.DataFrame(rows).write_csv(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _new_run_id(clock: Callable[[], datetime], configuration_hash: str) -> str:
    timestamp = clock()
    if timestamp.tzinfo is None:
        raise ValueError("run timestamp must be timezone-aware")
    utc_timestamp = timestamp.astimezone(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"baseline-{utc_timestamp}-{configuration_hash[:10]}"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=DEFAULT_BASELINE_CONFIG_PATH,
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/processed/split_manifest.json"),
    )
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/models/baselines"))
    parser.add_argument(
        "--leaderboard",
        type=Path,
        default=Path("artifacts/reports/model_leaderboard.csv"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run all baselines and print validation metrics without opening test data."""
    args = _parser().parse_args(argv)
    settings = Settings.load(args.config)
    baseline_config = BaselineSettings.load(args.baseline_config)
    configure_logging(settings.log_level)
    logger = get_logger(__name__)
    try:
        result = train_baselines(
            settings=settings,
            baseline_config=baseline_config,
            processed_dir=args.processed_dir,
            split_manifest_path=args.split_manifest,
            artifacts_dir=args.artifacts_dir,
            leaderboard_path=args.leaderboard,
            project_root=Path.cwd(),
        )
    except (BaselineTrainingError, FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        logger.error("baseline_training_failed", error=str(exc))
        return 1
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "strongest_baseline": result.strongest_model,
                "test_evaluated": False,
                "leaderboard": str(result.leaderboard_path),
                "validation_metrics": {
                    summary.model_name: summary.metrics for summary in result.model_summaries
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
