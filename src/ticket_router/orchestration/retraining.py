"""Controlled, privacy-aware retraining decisions and dataset versioning."""

from __future__ import annotations

import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from ticket_router.config import Settings
from ticket_router.data.duplicates import normalized_text_hash
from ticket_router.data.manifests import GitVersion, atomic_write_json, get_git_version
from ticket_router.data.normalize import combine_ticket_text, normalize_nullable_text
from ticket_router.data.split_manifest import OutputFileManifest, SplitManifest
from ticket_router.data.splitting import PREPARED_COLUMNS
from ticket_router.features.text import preprocess_model_text
from ticket_router.hashing import sha256_file, sha256_json
from ticket_router.orchestration.config import RetrainingSettings
from ticket_router.orchestration.contracts import RetrainingDecision

APPROVED_COLUMNS = frozenset(
    {"request_id", "subject", "body", "queue", "label_created_at", "approved"}
)


class RetrainingDatasetError(RuntimeError):
    """Raised when an approved dataset cannot be safely versioned."""


class RetrainingDatasetManifest(BaseModel):
    """Lineage for one immutable retraining snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_version: int = 1
    dataset_id: str = Field(pattern=r"^[a-zA-Z0-9_.-]+$")
    created_at_utc: str
    source_period_start: str
    source_period_end: str
    feedback_label_count: int = Field(ge=0)
    approved_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_split_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_training_data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_validation_data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    included_record_count: int = Field(ge=0)
    excluded_record_count: int = Field(ge=0)
    exclusion_reasons: dict[str, int]
    output_training_rows: int = Field(ge=0)
    output_validation_rows: int = Field(ge=0)
    output_files: dict[str, OutputFileManifest]
    data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    orchestration_configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    code_version: GitVersion | None

    @classmethod
    def read(cls, path: Path) -> RetrainingDatasetManifest:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def evaluate_retraining_conditions(
    *,
    monitoring_summary: dict[str, object],
    recent_statuses: tuple[str, ...],
    settings: RetrainingSettings,
    manual_trigger: bool,
) -> RetrainingDecision:
    """Require sustained, multi-signal evidence unless a human triggers evaluation."""
    feedback_count = _as_int(monitoring_summary.get("feedback_count"))
    status = str(monitoring_summary.get("status", "insufficient_data"))
    consecutive = _count_consecutive_critical((status, *recent_statuses))
    if manual_trigger:
        return RetrainingDecision(
            should_retrain=feedback_count >= settings.minimum_new_feedback_labels,
            trigger="manual",
            reasons=(
                "manual retraining evaluation requested",
                _feedback_reason(feedback_count, settings.minimum_new_feedback_labels),
            ),
            feedback_count=feedback_count,
            consecutive_critical_windows=consecutive,
        )

    quality_decline, low_confidence_increase = _monitoring_signals(monitoring_summary)
    reasons = (
        f"monitoring status is {status}",
        f"consecutive critical windows: {consecutive}",
        _feedback_reason(feedback_count, settings.minimum_new_feedback_labels),
        f"macro-F1 decline: {quality_decline:.6f}",
        f"low-confidence-rate increase: {low_confidence_increase:.6f}",
    )
    sustained_critical = (
        status == "critical" and consecutive >= settings.required_consecutive_critical_windows
    )
    quality_signal = quality_decline >= settings.macro_f1_decline_tolerance
    confidence_signal = low_confidence_increase >= settings.sustained_low_confidence_rate_increase
    enough_feedback = feedback_count >= settings.minimum_new_feedback_labels
    return RetrainingDecision(
        should_retrain=sustained_critical
        and enough_feedback
        and (quality_signal or confidence_signal),
        trigger="automatic" if sustained_critical else "none",
        reasons=reasons,
        feedback_count=feedback_count,
        consecutive_critical_windows=consecutive,
    )


def prepare_retraining_dataset(
    *,
    approved_input_path: Path,
    parent_processed_dir: Path,
    parent_manifest_path: Path,
    output_root: Path,
    dataset_id: str,
    source_period_start: datetime,
    source_period_end: datetime,
    feedback_label_count: int,
    settings: Settings,
    orchestration_configuration_sha256: str,
    project_root: Path,
    clock: datetime | None = None,
) -> Path:
    """Append approved labels to training only and preserve validation/test boundaries."""
    _validate_window(source_period_start, source_period_end)
    run_dir = output_root / dataset_id
    manifest_path = run_dir / "retraining_manifest.json"
    candidate_manifest_path = run_dir / "split_manifest.json"
    if manifest_path.exists() and candidate_manifest_path.exists():
        manifest = RetrainingDatasetManifest.read(manifest_path)
        _validate_reusable_snapshot(
            manifest,
            approved_input_path=approved_input_path,
            parent_manifest_path=parent_manifest_path,
            run_dir=run_dir,
        )
        return candidate_manifest_path
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RetrainingDatasetError(
            f"incomplete retraining snapshot exists at {run_dir}; use a new dataset_id"
        )
    if not approved_input_path.exists():
        raise RetrainingDatasetError(
            "approved retraining input is unavailable; place a reviewed Parquet file at "
            f"{approved_input_path}"
        )

    parent = SplitManifest.read(parent_manifest_path)
    train_path = parent_processed_dir / "train.parquet"
    validation_path = parent_processed_dir / "validation.parquet"
    test_path = parent_processed_dir / "test.parquet"
    _verify_parent_files(parent, train_path=train_path, validation_path=validation_path)
    approved = pl.read_parquet(approved_input_path)
    missing = sorted(APPROVED_COLUMNS - set(approved.columns))
    if missing:
        raise RetrainingDatasetError("approved input is missing columns: " + ", ".join(missing))

    train = pl.read_parquet(train_path)
    validation = pl.read_parquet(validation_path)
    holdout_hashes = set(validation["normalized_text_hash"].to_list())
    if test_path.exists():
        holdout_hashes.update(
            pl.read_parquet(test_path, columns=["normalized_text_hash"])[
                "normalized_text_hash"
            ].to_list()
        )
    new_rows, exclusions = _approved_rows(
        approved,
        labels=set(parent.label_mapping),
        holdout_hashes=holdout_hashes,
        existing_ids=set(train["ticket_record_id"].to_list()),
        settings=settings,
        source_period_start=source_period_start,
        source_period_end=source_period_end,
    )
    if not new_rows:
        raise RetrainingDatasetError("no approved records remain after retraining safeguards")

    additions = pl.DataFrame(new_rows).select(PREPARED_COLUMNS)
    combined_train = pl.concat(
        [train.select(PREPARED_COLUMNS), additions], how="vertical_relaxed"
    ).sort("ticket_record_id")
    processed_dir = run_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "train": processed_dir / "train.parquet",
        "validation": processed_dir / "validation.parquet",
    }
    _atomic_write_parquet(combined_train, output_paths["train"])
    _atomic_write_parquet(validation, output_paths["validation"])
    output_files = {
        name: OutputFileManifest(
            path=_portable_path(path, project_root),
            sha256=sha256_file(path),
            row_count=(combined_train.height if name == "train" else validation.height),
        )
        for name, path in output_paths.items()
    }
    parent_output_files = dict(parent.output_files)
    parent_output_files.update(output_files)
    split_counts = dict(parent.split_counts)
    split_counts["train"] = combined_train.height
    final_rows = sum(split_counts.values())
    split_percentages = {name: count / final_rows for name, count in split_counts.items()}
    per_class_counts = dict(parent.per_class_counts)
    per_class_counts["train"] = _class_counts(combined_train)
    candidate_manifest = parent.model_copy(
        update={
            "preparation_timestamp_utc": _format_utc(clock or datetime.now(UTC)),
            "data_source_hashes": {
                **parent.data_source_hashes,
                "parent_split_manifest": sha256_file(parent_manifest_path),
                "approved_feedback_input": sha256_file(approved_input_path),
            },
            "selected_input_rows": parent.selected_input_rows + len(new_rows),
            "final_row_count": final_rows,
            "split_counts": split_counts,
            "split_percentages": split_percentages,
            "per_class_counts": per_class_counts,
            "output_files": parent_output_files,
            "preparation_code_version": get_git_version(project_root),
        }
    )
    candidate_manifest.write(candidate_manifest_path)
    retraining_manifest = RetrainingDatasetManifest(
        dataset_id=dataset_id,
        created_at_utc=_format_utc(clock or datetime.now(UTC)),
        source_period_start=_format_utc(source_period_start),
        source_period_end=_format_utc(source_period_end),
        feedback_label_count=feedback_label_count,
        approved_input_sha256=sha256_file(approved_input_path),
        parent_split_manifest_sha256=sha256_file(parent_manifest_path),
        parent_training_data_sha256=parent.output_files["train"].sha256,
        parent_validation_data_sha256=parent.output_files["validation"].sha256,
        included_record_count=len(new_rows),
        excluded_record_count=sum(exclusions.values()),
        exclusion_reasons=dict(sorted(exclusions.items())),
        output_training_rows=combined_train.height,
        output_validation_rows=validation.height,
        output_files=output_files,
        data_sha256=output_files["train"].sha256,
        orchestration_configuration_sha256=orchestration_configuration_sha256,
        code_version=get_git_version(project_root),
    )
    atomic_write_json(manifest_path, retraining_manifest.model_dump(mode="json"))
    return candidate_manifest_path


def _approved_rows(
    frame: pl.DataFrame,
    *,
    labels: set[str],
    holdout_hashes: set[str],
    existing_ids: set[str],
    settings: Settings,
    source_period_start: datetime,
    source_period_end: datetime,
) -> tuple[list[dict[str, object]], Counter[str]]:
    rows: list[dict[str, object]] = []
    exclusions: Counter[str] = Counter()
    seen_request_ids: set[str] = set()
    for source_row_number, source in enumerate(frame.iter_rows(named=True), start=1):
        request_id = str(source.get("request_id", "")).strip()
        created_at = _parse_datetime(source.get("label_created_at"))
        if source.get("approved") is not True:
            exclusions["not_approved"] += 1
            continue
        if created_at is None or not (source_period_start <= created_at < source_period_end):
            exclusions["outside_source_period"] += 1
            continue
        if not request_id or request_id in seen_request_ids:
            exclusions["missing_or_duplicate_request_id"] += 1
            continue
        seen_request_ids.add(request_id)
        subject = normalize_nullable_text(source.get("subject"))
        body = normalize_nullable_text(source.get("body"))
        queue = normalize_nullable_text(source.get("queue"))
        if queue not in labels:
            exclusions["unapproved_target_label"] += 1
            continue
        if subject is None and body is None:
            exclusions["missing_usable_text"] += 1
            continue
        text = combine_ticket_text(subject, body)
        text_group = normalized_text_hash(text)
        if text_group in holdout_hashes:
            exclusions["holdout_duplicate"] += 1
            continue
        record_id = sha256_json(
            {"request_id": request_id, "subject": subject, "body": body, "queue": queue}
        )
        if record_id in existing_ids:
            exclusions["existing_training_record"] += 1
            continue
        rows.append(
            {
                "ticket_record_id": record_id,
                "source_row_id": f"approved-feedback:{sha256_json(request_id)[:16]}",
                "source_file": "approved_labeled_tickets.parquet",
                "source_row_number": source_row_number,
                "language": settings.project_config.dataset.language_filter,
                "subject": subject,
                "body": body,
                "text": text,
                "model_text": preprocess_model_text(text, settings.project_config.preprocessing),
                "normalized_text_hash": text_group,
                "queue": queue,
            }
        )
    return rows, exclusions


def _monitoring_signals(summary: dict[str, object]) -> tuple[float, float]:
    decision = summary.get("alert_decision")
    if not isinstance(decision, dict):
        return 0.0, 0.0
    signals = decision.get("signals")
    if not isinstance(signals, list):
        return 0.0, 0.0
    observed = {
        str(item.get("name")): _as_float(item.get("observed"))
        for item in signals
        if isinstance(item, dict)
    }
    return (
        observed.get("labeled_macro_f1_decline", 0.0),
        observed.get("low_confidence_rate_increase", 0.0),
    )


def _count_consecutive_critical(statuses: tuple[str, ...]) -> int:
    count = 0
    for status in statuses:
        if status != "critical":
            break
        count += 1
    return count


def _feedback_reason(actual: int, required: int) -> str:
    return f"feedback labels: {actual} (minimum {required})"


def _as_int(value: object) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def _as_float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _parse_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
        except ValueError:
            return None
    return None


def _validate_window(start: datetime, end: datetime) -> None:
    if start.tzinfo is None or end.tzinfo is None or end <= start:
        raise RetrainingDatasetError("source period must be timezone-aware and ordered")


def _verify_parent_files(parent: SplitManifest, *, train_path: Path, validation_path: Path) -> None:
    for name, path in (("train", train_path), ("validation", validation_path)):
        if not path.exists() or sha256_file(path) != parent.output_files[name].sha256:
            raise RetrainingDatasetError(f"parent {name} data does not match its manifest")


def _validate_reusable_snapshot(
    manifest: RetrainingDatasetManifest,
    *,
    approved_input_path: Path,
    parent_manifest_path: Path,
    run_dir: Path,
) -> None:
    expected = {
        "approved input": (approved_input_path, manifest.approved_input_sha256),
        "parent manifest": (parent_manifest_path, manifest.parent_split_manifest_sha256),
    }
    expected.update(
        {
            name: (run_dir / "processed" / Path(item.path).name, item.sha256)
            for name, item in manifest.output_files.items()
        }
    )
    for name, (path, digest) in expected.items():
        if not path.exists() or sha256_file(path) != digest:
            raise RetrainingDatasetError(f"reusable snapshot {name} hash mismatch")


def _atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _class_counts(frame: pl.DataFrame) -> dict[str, int]:
    return {
        str(row["queue"]): int(row["len"])
        for row in frame.group_by("queue").len().sort("queue").iter_rows(named=True)
    }


def _portable_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
