"""Prepare deterministic, duplicate-safe train/validation/test Parquet artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import polars as pl

from ticket_router.config import Settings
from ticket_router.data.class_selection import ClassSelectionReport
from ticket_router.data.download import configuration_hash
from ticket_router.data.duplicates import NORMALIZED_TEXT_HASH_COLUMN
from ticket_router.data.manifests import NormalizationManifest, atomic_write_json, get_git_version
from ticket_router.data.split_manifest import OutputFileManifest, SplitManifest
from ticket_router.data.splitting import (
    MODEL_READY_FEATURE_COLUMNS,
    SplitError,
    create_prepared_splits,
    validate_prepared_splits,
)
from ticket_router.data.validation import DataContractError, validate_normalized_frame
from ticket_router.hashing import sha256_file
from ticket_router.logging_config import configure_logging, get_logger

TRAIN_FILENAME = "train.parquet"
VALIDATION_FILENAME = "validation.parquet"
TEST_FILENAME = "test.parquet"
TRAINING_REFERENCE_FILENAME = "training_reference.parquet"
SPLIT_MANIFEST_FILENAME = "split_manifest.json"
SPLIT_SUMMARY_FILENAME = "split_summary.json"


class PreparationError(RuntimeError):
    """Raised when verified prepared artifacts cannot be produced safely."""


def prepare_dataset(
    *,
    settings: Settings,
    normalized_path: Path,
    normalization_manifest_path: Path,
    selected_classes_path: Path,
    processed_dir: Path,
    reference_dir: Path,
    reports_dir: Path,
    project_root: Path,
    force: bool = False,
    clock: Callable[[], datetime] | None = None,
) -> SplitManifest:
    """Validate lineage, prepare splits, and write a complete manifest."""
    resolved_root = project_root.resolve()
    normalized_path = normalized_path.resolve()
    normalization_manifest_path = normalization_manifest_path.resolve()
    selected_classes_path = selected_classes_path.resolve()
    processed_dir = processed_dir.resolve()
    reference_dir = reference_dir.resolve()
    reports_dir = reports_dir.resolve()
    split_manifest_path = processed_dir / SPLIT_MANIFEST_FILENAME

    normalization_manifest = NormalizationManifest.read(normalization_manifest_path)
    selected_classes = ClassSelectionReport.read(selected_classes_path)
    source_hashes = {
        "normalized_data": sha256_file(normalized_path),
        "normalization_manifest": sha256_file(normalization_manifest_path),
        "selected_classes": sha256_file(selected_classes_path),
    }
    _validate_input_lineage(
        settings=settings,
        normalization_manifest=normalization_manifest,
        selected_classes=selected_classes,
        normalized_hash=source_hashes["normalized_data"],
    )
    current_configuration_hash = configuration_hash(settings)

    if not force and split_manifest_path.exists():
        cached = SplitManifest.read(split_manifest_path)
        _validate_cached_manifest(
            cached,
            source_hashes=source_hashes,
            configuration_digest=current_configuration_hash,
            project_root=resolved_root,
        )
        _write_split_summary(
            cached,
            summary_path=reports_dir / SPLIT_SUMMARY_FILENAME,
            manifest_path=split_manifest_path,
            project_root=resolved_root,
        )
        return cached

    expected_paths = _expected_output_paths(processed_dir, reference_dir)
    if not force and any(path.exists() for path in expected_paths.values()):
        raise PreparationError(
            "Prepared outputs exist without a reusable matching split manifest. "
            "Inspect them or rerun with --force to replace generated artifacts."
        )

    frame = pl.read_parquet(normalized_path)
    validate_normalized_frame(
        frame,
        settings=settings.project_config.analysis,
        language_filter=settings.project_config.dataset.language_filter,
    )
    prepared = create_prepared_splits(
        frame,
        selected_classes=selected_classes.selected_classes,
        preprocessing=settings.project_config.preprocessing,
        splitting=settings.project_config.splitting,
        split_ratios=settings.project_config.split_ratios,
        random_seed=settings.random_seed,
    )
    validation = validate_prepared_splits(
        prepared,
        selected_classes=selected_classes.selected_classes,
        split_ratios=settings.project_config.split_ratios,
        splitting=settings.project_config.splitting,
    )
    reference_frame = _training_reference(prepared.train)
    output_frames = {
        "train": prepared.train,
        "validation": prepared.validation,
        "test": prepared.test,
        "training_reference": reference_frame,
    }
    output_files = _write_parquet_outputs(
        output_frames,
        output_paths=expected_paths,
        project_root=resolved_root,
    )
    manifest = SplitManifest(
        preparation_timestamp_utc=_format_utc((clock or _utc_now)()),
        data_source_hashes=source_hashes,
        configuration_hash=current_configuration_hash,
        preprocessing=settings.project_config.preprocessing,
        splitting=settings.project_config.splitting,
        split_ratios=settings.project_config.split_ratios,
        random_seed=settings.random_seed,
        duplicate_group_column=NORMALIZED_TEXT_HASH_COLUMN,
        selected_input_rows=prepared.selected_input_rows,
        contradictory_group_count=prepared.contradictory_group_count,
        contradictory_rows_excluded=prepared.contradictory_rows_excluded,
        final_row_count=validation.total_records,
        split_counts=validation.split_counts,
        split_percentages=validation.split_percentages,
        per_class_counts=validation.per_class_counts,
        maximum_class_proportion_deviation=(validation.maximum_class_proportion_deviation),
        most_deviant_class=validation.most_deviant_class,
        label_mapping=selected_classes.label_mapping,
        model_feature_columns=MODEL_READY_FEATURE_COLUMNS,
        target_column=settings.project_config.dataset.target_column,
        output_files=output_files,
        preparation_code_version=get_git_version(resolved_root),
    )
    manifest.write(split_manifest_path)
    _write_split_summary(
        manifest,
        summary_path=reports_dir / SPLIT_SUMMARY_FILENAME,
        manifest_path=split_manifest_path,
        project_root=resolved_root,
    )
    return manifest


def _validate_input_lineage(
    *,
    settings: Settings,
    normalization_manifest: NormalizationManifest,
    selected_classes: ClassSelectionReport,
    normalized_hash: str,
) -> None:
    dataset_settings = settings.project_config.dataset
    if normalized_hash != normalization_manifest.output_file_sha256:
        raise PreparationError("Normalized data hash does not match normalization_manifest.json.")
    if normalized_hash != selected_classes.input_file_sha256:
        raise PreparationError(
            "Selected-class mapping was generated from a different normalized dataset."
        )
    if selected_classes.requested_class_count != dataset_settings.number_of_target_queues:
        raise PreparationError(
            "Selected-class mapping does not match configured number_of_target_queues."
        )
    if selected_classes.configured_minimum_class_count != dataset_settings.minimum_class_count:
        raise PreparationError(
            "Selected-class mapping does not match configured minimum_class_count."
        )


def _expected_output_paths(
    processed_dir: Path,
    reference_dir: Path,
) -> dict[str, Path]:
    return {
        "train": processed_dir / TRAIN_FILENAME,
        "validation": processed_dir / VALIDATION_FILENAME,
        "test": processed_dir / TEST_FILENAME,
        "training_reference": reference_dir / TRAINING_REFERENCE_FILENAME,
    }


def _training_reference(train: pl.DataFrame) -> pl.DataFrame:
    """Create a raw-text-free row-level reference for later monitoring."""
    return train.select(
        "queue",
        pl.col("model_text").str.len_chars().alias("model_text_character_count"),
        pl.col("model_text").str.split(" ").list.len().alias("model_text_word_count"),
        pl.col("subject").is_not_null().alias("subject_present"),
        pl.col("body").is_not_null().alias("body_present"),
    )


def _write_parquet_outputs(
    frames: Mapping[str, pl.DataFrame],
    *,
    output_paths: Mapping[str, Path],
    project_root: Path,
) -> dict[str, OutputFileManifest]:
    temporary_paths: dict[str, Path] = {}
    try:
        for name, frame in frames.items():
            output_path = output_paths[name]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")
            frame.write_parquet(
                temporary_path,
                compression="zstd",
                statistics=True,
                row_group_size=8192,
            )
            temporary_paths[name] = temporary_path
        for name, temporary_path in temporary_paths.items():
            os.replace(temporary_path, output_paths[name])
    finally:
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)

    return {
        name: OutputFileManifest(
            path=_portable_path(output_paths[name], project_root),
            sha256=sha256_file(output_paths[name]),
            row_count=frame.height,
        )
        for name, frame in frames.items()
    }


def _validate_cached_manifest(
    manifest: SplitManifest,
    *,
    source_hashes: dict[str, str],
    configuration_digest: str,
    project_root: Path,
) -> None:
    if (
        manifest.data_source_hashes != source_hashes
        or manifest.configuration_hash != configuration_digest
    ):
        raise PreparationError(
            "Existing prepared artifacts were generated from different data or configuration; "
            "use --force only after reviewing the change."
        )
    for output in manifest.output_files.values():
        output_path = _resolve_path(output.path, project_root)
        if not output_path.is_file() or sha256_file(output_path) != output.sha256:
            raise PreparationError(
                f"Prepared output is missing or fails its manifest hash: {output_path}"
            )


def _write_split_summary(
    manifest: SplitManifest,
    *,
    summary_path: Path,
    manifest_path: Path,
    project_root: Path,
) -> None:
    difficulties = {
        split_name: {
            "class": manifest.most_deviant_class[split_name],
            "absolute_proportion_deviation": (
                manifest.maximum_class_proportion_deviation[split_name]
            ),
        }
        for split_name in manifest.split_counts
    }
    atomic_write_json(
        summary_path,
        {
            "split_manifest_path": _portable_path(manifest_path, project_root),
            "split_manifest_sha256": sha256_file(manifest_path),
            "final_row_count": manifest.final_row_count,
            "contradictory_group_count": manifest.contradictory_group_count,
            "contradictory_rows_excluded": manifest.contradictory_rows_excluded,
            "split_counts": manifest.split_counts,
            "split_percentages": manifest.split_percentages,
            "per_class_counts": manifest.per_class_counts,
            "class_difficulties": difficulties,
            "output_hashes": {
                name: output.sha256 for name, output in manifest.output_files.items()
            },
        },
    )


def _portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _resolve_path(path_text: str, project_root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else project_root / path


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _format_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--normalized-data",
        type=Path,
        default=Path("data/interim/normalized_tickets.parquet"),
    )
    parser.add_argument(
        "--normalization-manifest",
        type=Path,
        default=Path("data/interim/normalization_manifest.json"),
    )
    parser.add_argument(
        "--selected-classes",
        type=Path,
        default=Path("artifacts/reports/selected_classes.json"),
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--reference-dir", type=Path, default=Path("data/reference"))
    parser.add_argument("--reports-dir", type=Path, default=Path("artifacts/reports"))
    parser.add_argument("--force", action="store_true", help="Replace prepared artifacts.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic data-preparation CLI."""
    args = _parser().parse_args(argv)
    settings = Settings.load(args.config)
    configure_logging(settings.log_level)
    logger = get_logger(__name__)
    try:
        manifest = prepare_dataset(
            settings=settings,
            normalized_path=args.normalized_data,
            normalization_manifest_path=args.normalization_manifest,
            selected_classes_path=args.selected_classes,
            processed_dir=args.processed_dir,
            reference_dir=args.reference_dir,
            reports_dir=args.reports_dir,
            project_root=Path.cwd(),
            force=args.force,
        )
    except (
        DataContractError,
        FileNotFoundError,
        OSError,
        PreparationError,
        SplitError,
    ) as exc:
        logger.error("data_preparation_failed", error=str(exc))
        return 1
    print(
        json.dumps(
            {
                "final_row_count": manifest.final_row_count,
                "contradictory_rows_excluded": manifest.contradictory_rows_excluded,
                "split_counts": manifest.split_counts,
                "split_percentages": manifest.split_percentages,
                "per_class_counts": manifest.per_class_counts,
                "most_deviant_class": manifest.most_deviant_class,
                "maximum_class_proportion_deviation": (manifest.maximum_class_proportion_deviation),
                "manifest": str(args.processed_dir / SPLIT_MANIFEST_FILENAME),
                "summary": str(args.reports_dir / SPLIT_SUMMARY_FILENAME),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
