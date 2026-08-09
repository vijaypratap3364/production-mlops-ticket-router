from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from ticket_router.config import Settings
from ticket_router.data.split_manifest import OutputFileManifest, SplitManifest
from ticket_router.data.splitting import PREPARED_COLUMNS
from ticket_router.hashing import sha256_file
from ticket_router.orchestration.config import OrchestrationConfig
from ticket_router.orchestration.retraining import (
    RetrainingDatasetManifest,
    evaluate_retraining_conditions,
    prepare_retraining_dataset,
)


def test_no_retraining_for_one_critical_window() -> None:
    config = OrchestrationConfig.load()
    decision = evaluate_retraining_conditions(
        monitoring_summary=_summary(feedback=200),
        recent_statuses=("healthy",),
        settings=config.retraining,
        manual_trigger=False,
    )

    assert decision.should_retrain is False
    assert decision.consecutive_critical_windows == 1


def test_retraining_requires_enough_feedback() -> None:
    config = OrchestrationConfig.load()
    decision = evaluate_retraining_conditions(
        monitoring_summary=_summary(feedback=10),
        recent_statuses=("critical",),
        settings=config.retraining,
        manual_trigger=False,
    )

    assert decision.should_retrain is False
    assert decision.feedback_count == 10


def test_sustained_quality_decline_triggers_retraining() -> None:
    config = OrchestrationConfig.load()
    decision = evaluate_retraining_conditions(
        monitoring_summary=_summary(feedback=200),
        recent_statuses=("critical",),
        settings=config.retraining,
        manual_trigger=False,
    )

    assert decision.should_retrain is True
    assert decision.trigger == "automatic"


def test_approved_snapshot_is_parent_linked_and_idempotent(tmp_path: Path) -> None:
    settings = Settings.load()
    processed = tmp_path / "processed"
    processed.mkdir()
    train = _prepared_frame("train", "Billing and Payments", "Original training text")
    validation = _prepared_frame("validation", "Billing and Payments", "Holdout duplicate")
    test = _prepared_frame("test", "Billing and Payments", "Sealed test text")
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        frame.write_parquet(processed / f"{name}.parquet")
    parent_path = processed / "split_manifest.json"
    _parent_manifest(settings, processed).write(parent_path)
    approved_path = tmp_path / "approved.parquet"
    pl.DataFrame(
        {
            "request_id": ["new-1", "new-2", "new-3"],
            "subject": ["New invoice", "Holdout duplicate", "Not reviewed"],
            "body": ["Please inspect charge", None, "Ignore"],
            "queue": ["Billing and Payments"] * 3,
            "label_created_at": [
                datetime(2026, 1, 2, tzinfo=UTC),
                datetime(2026, 1, 2, tzinfo=UTC),
                datetime(2026, 1, 2, tzinfo=UTC),
            ],
            "approved": [True, True, False],
        }
    ).write_parquet(approved_path)

    def prepare() -> Path:
        return prepare_retraining_dataset(
            approved_input_path=approved_path,
            parent_processed_dir=processed,
            parent_manifest_path=parent_path,
            output_root=tmp_path / "runs",
            dataset_id="fixture-v1",
            source_period_start=datetime(2026, 1, 1, tzinfo=UTC),
            source_period_end=datetime(2026, 1, 3, tzinfo=UTC),
            feedback_label_count=3,
            settings=settings,
            orchestration_configuration_sha256="a" * 64,
            project_root=tmp_path,
            clock=datetime(2026, 1, 4, tzinfo=UTC),
        )

    first = prepare()
    first_hash = sha256_file(first)
    second = prepare()
    manifest = RetrainingDatasetManifest.read(first.parent / "retraining_manifest.json")

    assert first == second
    assert sha256_file(second) == first_hash
    assert manifest.included_record_count == 1
    assert manifest.excluded_record_count == 2
    assert manifest.parent_split_manifest_sha256 == sha256_file(parent_path)
    resulting_train = pl.read_parquet(first.parent / "processed/train.parquet")
    assert resulting_train.height == 2
    assert all("new-1" not in value for value in resulting_train["source_row_id"].to_list())
    assert pl.read_parquet(first.parent / "processed/validation.parquet").equals(validation)


def _summary(*, feedback: int) -> dict[str, object]:
    return {
        "status": "critical",
        "feedback_count": feedback,
        "alert_decision": {
            "signals": [
                {
                    "name": "labeled_macro_f1_decline",
                    "severity": "critical",
                    "observed": 0.2,
                    "threshold": 0.1,
                },
                {
                    "name": "low_confidence_rate_increase",
                    "severity": "critical",
                    "observed": 0.25,
                    "threshold": 0.2,
                },
            ]
        },
    }


def _prepared_frame(source: str, queue: str, text: str) -> pl.DataFrame:
    from ticket_router.data.duplicates import normalized_text_hash

    record = {
        "ticket_record_id": (source[0] * 64),
        "source_row_id": f"{source}:1",
        "source_file": f"{source}.parquet",
        "source_row_number": 1,
        "language": "en",
        "subject": text,
        "body": None,
        "text": f"[SUBJECT] {text}\n[BODY]",
        "model_text": f"[SUBJECT] {text} [BODY]",
        "normalized_text_hash": normalized_text_hash(f"[SUBJECT] {text}\n[BODY]"),
        "queue": queue,
    }
    return pl.DataFrame([record]).select(PREPARED_COLUMNS)


def _parent_manifest(settings: Settings, processed: Path) -> SplitManifest:
    outputs = {
        name: OutputFileManifest(
            path=(processed / f"{name}.parquet").as_posix(),
            sha256=sha256_file(processed / f"{name}.parquet"),
            row_count=1,
        )
        for name in ("train", "validation", "test")
    }
    return SplitManifest(
        preparation_timestamp_utc="2026-01-01T00:00:00Z",
        data_source_hashes={"fixture": "b" * 64},
        configuration_hash="c" * 64,
        preprocessing=settings.project_config.preprocessing,
        splitting=settings.project_config.splitting,
        split_ratios=settings.project_config.split_ratios,
        random_seed=42,
        duplicate_group_column="normalized_text_hash",
        selected_input_rows=3,
        contradictory_group_count=0,
        contradictory_rows_excluded=0,
        final_row_count=3,
        split_counts={"train": 1, "validation": 1, "test": 1},
        split_percentages={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
        per_class_counts={
            name: {"Billing and Payments": 1} for name in ("train", "validation", "test")
        },
        maximum_class_proportion_deviation={"train": 0.0, "validation": 0.0, "test": 0.0},
        most_deviant_class={
            "train": "Billing and Payments",
            "validation": "Billing and Payments",
            "test": "Billing and Payments",
        },
        label_mapping={"Billing and Payments": 0},
        model_feature_columns=("model_text",),
        target_column="queue",
        output_files=outputs,
        preparation_code_version=None,
    )
