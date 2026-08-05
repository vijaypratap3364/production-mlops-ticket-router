"""Validate normalized data and generate deterministic Stage 3 analysis artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from ticket_router.config import Settings
from ticket_router.data.class_selection import ClassSelectionError, select_target_classes
from ticket_router.data.download import configuration_hash
from ticket_router.data.duplicates import analyze_duplicates, contradictory_duplicate_groups
from ticket_router.data.manifests import NormalizationManifest, RawDataManifest, atomic_write_json
from ticket_router.data.reporting import build_eda_report, write_eda_artifacts
from ticket_router.data.validation import DataContractError, validate_normalized_frame
from ticket_router.hashing import sha256_file
from ticket_router.logging_config import configure_logging, get_logger

SELECTED_CLASSES_FILENAME = "selected_classes.json"
DUPLICATE_ANALYSIS_FILENAME = "duplicate_analysis.json"
EDA_JSON_FILENAME = "eda_report.json"
EDA_HTML_FILENAME = "eda_report.html"


class AnalysisError(RuntimeError):
    """Raised when lineage checks or artifact generation cannot proceed."""


def run_analysis(
    *,
    settings: Settings,
    normalized_path: Path,
    normalization_manifest_path: Path,
    raw_manifest_path: Path,
    reports_dir: Path,
    project_root: Path,
) -> dict[str, object]:
    """Run contract, duplicate, class-selection, and EDA generation."""
    normalization_manifest = NormalizationManifest.read(normalization_manifest_path)
    raw_manifest = RawDataManifest.read(raw_manifest_path)
    actual_raw_manifest_hash = sha256_file(raw_manifest_path)
    if actual_raw_manifest_hash != normalization_manifest.raw_manifest_sha256:
        raise AnalysisError(
            "Raw manifest hash does not match the normalization lineage; "
            "use the raw manifest that produced this normalized artifact."
        )
    actual_normalized_hash = sha256_file(normalized_path)
    if actual_normalized_hash != normalization_manifest.output_file_sha256:
        raise AnalysisError(
            "Normalized Parquet hash does not match normalization_manifest.json; "
            "rerun normalization or restore the verified artifact."
        )

    frame = pl.read_parquet(normalized_path)
    if frame.height != normalization_manifest.output_row_count:
        raise AnalysisError(
            "Normalized Parquet row count does not match normalization_manifest.json; "
            f"expected {normalization_manifest.output_row_count}, received {frame.height}."
        )
    analysis_settings = settings.project_config.analysis
    dataset_settings = settings.project_config.dataset
    validation = validate_normalized_frame(
        frame,
        settings=analysis_settings,
        language_filter=dataset_settings.language_filter,
    )
    duplicates = analyze_duplicates(
        frame,
        near_empty_word_threshold=analysis_settings.near_empty_word_threshold,
        template_min_group_size=analysis_settings.template_min_group_size,
    )
    config_hash = configuration_hash(settings)
    class_selection = select_target_classes(
        frame,
        target_column=dataset_settings.target_column,
        number_of_classes=dataset_settings.number_of_target_queues,
        minimum_class_count=dataset_settings.minimum_class_count,
        split_ratios=settings.project_config.split_ratios,
        input_file_path=_portable_path(normalized_path, project_root),
        input_file_sha256=actual_normalized_hash,
        configuration_hash=config_hash,
    )
    selected_frame = frame.filter(
        pl.col(dataset_settings.target_column).is_in(class_selection.selected_classes)
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    selected_classes_path = reports_dir / SELECTED_CLASSES_FILENAME
    duplicate_analysis_path = reports_dir / DUPLICATE_ANALYSIS_FILENAME
    eda_json_path = reports_dir / EDA_JSON_FILENAME
    eda_html_path = reports_dir / EDA_HTML_FILENAME
    class_selection.write(selected_classes_path)
    atomic_write_json(
        duplicate_analysis_path,
        {
            **duplicates.to_dict(),
            "contradictory_groups": contradictory_duplicate_groups(frame).to_dicts(),
            "normalized_text_hash_algorithm": "sha256(nfkc(casefold(collapse_whitespace(text))))",
            "template_text_hash_algorithm": (
                "sha256(normalized_text_with_urls_emails_and_digit_tokens_masked)"
            ),
            "split_policy": (
                "All records sharing normalized_text_hash must remain in one future split."
            ),
        },
    )
    report = build_eda_report(
        frame=frame,
        selected_frame=selected_frame,
        validation=validation,
        duplicates=duplicates,
        class_selection=class_selection,
        normalization_manifest=normalization_manifest,
        raw_manifest=raw_manifest,
        common_tokens_per_class=analysis_settings.common_tokens_per_class,
        token_min_document_frequency=analysis_settings.token_min_document_frequency,
    )
    write_eda_artifacts(report, json_path=eda_json_path, html_path=eda_html_path)

    return {
        "selected_classes": list(class_selection.selected_classes),
        "final_usable_row_count": class_selection.final_row_count,
        "imbalance_ratio": class_selection.imbalance_ratio,
        "exact_duplicate_groups": duplicates.exact_duplicate_groups,
        "exact_duplicate_rows_flagged": duplicates.exact_duplicate_rows,
        "exact_duplicate_redundant_records": duplicates.exact_duplicate_redundant_records,
        "contradictory_duplicate_groups": duplicates.contradictory_duplicate_groups,
        "artifacts": [
            _portable_path(selected_classes_path, project_root),
            _portable_path(duplicate_analysis_path, project_root),
            _portable_path(eda_json_path, project_root),
            _portable_path(eda_html_path, project_root),
        ],
    }


def _portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


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
        "--raw-manifest",
        type=Path,
        default=Path("data/raw/data_manifest.json"),
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("artifacts/reports"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Stage 3 analysis CLI."""
    args = _parser().parse_args(argv)
    settings = Settings.load(args.config)
    configure_logging(settings.log_level)
    logger = get_logger(__name__)
    try:
        result = run_analysis(
            settings=settings,
            normalized_path=args.normalized_data,
            normalization_manifest_path=args.normalization_manifest,
            raw_manifest_path=args.raw_manifest,
            reports_dir=args.reports_dir,
            project_root=Path.cwd(),
        )
    except (
        AnalysisError,
        ClassSelectionError,
        DataContractError,
        FileNotFoundError,
        OSError,
    ) as exc:
        logger.error("data_analysis_failed", error=str(exc))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
