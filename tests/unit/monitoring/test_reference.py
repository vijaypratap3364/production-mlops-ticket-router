"""Monitoring reference generation tests."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl

from ticket_router.api.service import LoadedChampion, ProbabilisticPredictor
from ticket_router.hashing import sha256_file
from ticket_router.monitoring.config import MonitoringConfig
from ticket_router.monitoring.reference import build_monitoring_reference


class ReferenceModel:
    classes_ = np.asarray(["Billing", "Technical"], dtype=object)

    def predict(self, values: Sequence[str]) -> object:
        return np.asarray(["Billing" if "invoice" in value else "Technical" for value in values])

    def predict_proba(self, values: Sequence[str]) -> object:
        return np.asarray(
            [[0.8, 0.2] if "invoice" in value else [0.1, 0.9] for value in values],
            dtype=np.float64,
        )


def test_reference_generation_is_privacy_safe_and_hashed(
    tmp_path: Path,
    monitoring_config: MonitoringConfig,
) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    rows = {
        "subject": ["Invoice", "Network"],
        "body": ["invoice question", "network issue"],
        "model_text": ["invoice question", "network issue"],
        "queue": ["Billing", "Technical"],
    }
    pl.DataFrame(rows).write_parquet(processed / "train.parquet")
    pl.DataFrame(rows).write_parquet(processed / "validation.parquet")
    split_manifest = processed / "split_manifest.json"
    split_manifest.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "reference.parquet"
    manifest_path = tmp_path / "reference.json"
    champion = LoadedChampion(
        model=cast(ProbabilisticPredictor, ReferenceModel()),
        model_name="fixture-router",
        model_version="7",
        alias="champion",
        loaded_at=datetime(2026, 8, 7, tzinfo=UTC),
        labels=("Billing", "Technical"),
        input_contract={"predictive_fields": ["subject", "body"]},
    )

    manifest = build_monitoring_reference(
        processed_dir=processed,
        output_path=output,
        manifest_path=manifest_path,
        split_manifest_path=split_manifest,
        champion=champion,
        confidence_warning_threshold=0.5,
        champion_baseline_macro_f1=0.7,
        monitoring_config=monitoring_config,
        project_root=tmp_path,
        clock=datetime(2026, 8, 7, tzinfo=UTC),
    )
    frame = pl.read_parquet(output)

    assert manifest.row_count == 4
    assert manifest.reference_data_sha256 == sha256_file(output)
    assert manifest.champion_model_version == "7"
    assert {"predicted_queue", "prediction_confidence", "model_version"}.issubset(frame.columns)
    assert "actual_queue" not in frame.columns
    assert "subject" not in frame.columns
    assert "body" not in frame.columns
    assert "model_text" not in frame.columns
    assert manifest_path.is_file()
