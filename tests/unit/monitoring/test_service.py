"""Monitoring orchestration and database-summary port tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from ticket_router.db.contracts import MonitoringRun
from ticket_router.hashing import sha256_file, sha256_json
from ticket_router.monitoring.config import MonitoringConfig
from ticket_router.monitoring.contracts import CurrentPrediction, LabeledPrediction
from ticket_router.monitoring.reference import MonitoringReferenceManifest
from ticket_router.monitoring.service import execute_monitoring_run


class EmptyMonitoringData:
    def load_predictions(
        self,
        *,
        start: datetime,
        end: datetime,
        model_version: str | None = None,
    ) -> tuple[CurrentPrediction, ...]:
        return ()

    def load_labeled_predictions(
        self,
        *,
        start: datetime,
        end: datetime,
        model_version: str | None = None,
    ) -> tuple[LabeledPrediction, ...]:
        return ()


class CapturingRunRepository:
    def __init__(self) -> None:
        self.saved: MonitoringRun | None = None

    def save(self, run: MonitoringRun) -> None:
        self.saved = run

    def get(self, run_id: str) -> MonitoringRun | None:
        return self.saved if self.saved and self.saved.run_id == run_id else None


def test_insufficient_run_writes_summary_and_database_record(
    tmp_path: Path,
    monitoring_config: MonitoringConfig,
) -> None:
    reference = pl.DataFrame({"predicted_queue": ["A", "B"]})
    reference_path = tmp_path / "reference.parquet"
    reference.write_parquet(reference_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = MonitoringReferenceManifest(
        creation_timestamp_utc="2026-08-07T00:00:00Z",
        champion_model_name="fixture",
        champion_model_version="7",
        champion_alias="champion",
        row_count=2,
        columns=tuple(reference.columns),
        feature_definitions={},
        source_file_sha256={"train": "a" * 64},
        split_manifest_sha256="b" * 64,
        reference_data_sha256=sha256_file(reference_path),
        monitoring_configuration_sha256=sha256_json(monitoring_config.model_dump(mode="json")),
        confidence_warning_threshold=0.5,
        champion_baseline_macro_f1=0.7,
        code_version=None,
    )
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    repository = CapturingRunRepository()
    end = datetime(2026, 8, 7, tzinfo=UTC)

    result = execute_monitoring_run(
        reference_path=reference_path,
        reference_manifest_path=manifest_path,
        data_repository=EmptyMonitoringData(),
        run_repository=repository,
        config=monitoring_config,
        start=end - timedelta(days=7),
        end=end,
        output_root=tmp_path / "reports",
        run_id="00000000-0000-0000-0000-000000000001",
    )

    assert result.decision.status == "insufficient_data"
    assert result.summary_path.is_file()
    assert result.html_report_path is None
    assert repository.saved is not None
    assert repository.saved.drift_status == "insufficient_data"
    assert repository.saved.summary["event_count"] == 0
