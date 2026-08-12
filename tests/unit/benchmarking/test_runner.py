"""Benchmark artifact parsing, target comparison, and report tests."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from ticket_router.api.service import LoadedChampion, ProbabilisticPredictor
from ticket_router.benchmarking import runner
from ticket_router.benchmarking.config import BenchmarkConfig
from ticket_router.benchmarking.contracts import (
    LoadTestSummary,
    ReliabilityScenario,
    summarize_latencies,
)
from ticket_router.benchmarking.runner import (
    BenchmarkExecutionError,
    read_locust_summary,
    read_reliability_results,
    render_markdown_report,
    run_benchmark,
)
from ticket_router.config import Settings
from ticket_router.data.split_manifest import SplitManifest


class SerializableBenchmarkModel:
    classes_ = np.asarray(["Billing", "Returns", "Technical"], dtype=object)

    def predict_proba(self, values: Sequence[str]) -> object:
        return np.asarray([[0.7, 0.2, 0.1] for _ in values], dtype=np.float64)

    def predict(self, values: Sequence[str]) -> object:
        return np.asarray(["Billing" for _ in values], dtype=object)


def test_locust_aggregate_is_loaded_without_changing_measurements(tmp_path: Path) -> None:
    path = tmp_path / "locust_stats.csv"
    path.write_text(
        "Type,Name,Request Count,Failure Count,Median Response Time,95%,99%,Requests/s\n"
        ",Aggregated,100,2,12,40,60,8.5\n",
        encoding="utf-8",
    )

    result = read_locust_summary(path)

    assert result is not None
    assert result.request_count == 100
    assert result.failure_count == 2
    assert result.failure_rate == pytest.approx(2 / 100)
    assert result.p95_ms == 40
    assert result.throughput_per_second == 8.5


def test_invalid_locust_and_reliability_artifacts_fail_actionably(tmp_path: Path) -> None:
    locust_path = tmp_path / "locust.csv"
    locust_path.write_text("Name,Request Count\nPOST /predict,1\n", encoding="utf-8")
    with pytest.raises(BenchmarkExecutionError, match="aggregate row"):
        read_locust_summary(locust_path)

    reliability_path = tmp_path / "reliability.json"
    reliability_path.write_text("[]", encoding="utf-8")
    with pytest.raises(BenchmarkExecutionError, match="invalid contract"):
        read_reliability_results(reliability_path)


def test_reliability_artifact_preserves_observed_pass_and_failure(tmp_path: Path) -> None:
    path = tmp_path / "reliability.json"
    path.write_text(
        json.dumps(
            {
                "scenarios": [
                    {"name": "healthy", "passed": True, "expected": "200", "observed": "200"},
                    {"name": "failure", "passed": False, "expected": "200", "observed": "500"},
                ]
            }
        ),
        encoding="utf-8",
    )

    results = read_reliability_results(path)

    assert [item.passed for item in results] == [True, False]


def test_benchmark_runner_records_measurements_and_renders_complete_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings.load(env_file=None)
    split_manifest_path = tmp_path / "split_manifest.json"
    split_manifest_path.write_text("{}", encoding="utf-8")
    config = BenchmarkConfig.load().model_copy(
        update={
            "warmup_iterations": 1,
            "single_request_iterations": 10,
            "batch_sizes": (2,),
            "batch_repetitions": 3,
            "api_iterations": 10,
            "split_manifest_path": split_manifest_path,
            "output_json_path": tmp_path / "benchmark.json",
            "output_markdown_path": tmp_path / "benchmark.md",
            "locust_stats_path": tmp_path / "locust.csv",
            "reliability_results_path": tmp_path / "reliability.json",
        }
    )
    champion = LoadedChampion(
        model=cast(ProbabilisticPredictor, SerializableBenchmarkModel()),
        model_name="fixture-router",
        model_version="3",
        alias="champion",
        loaded_at=datetime(2026, 8, 12, tzinfo=UTC),
        labels=("Billing", "Returns", "Technical"),
        input_contract={"predictive_fields": ["subject", "body"]},
        training_data_hash="a" * 64,
        model_size_bytes=2048,
    )
    manifest = cast(
        SplitManifest,
        SimpleNamespace(label_mapping={"Billing": 0, "Returns": 1, "Technical": 2}),
    )
    api_summary = summarize_latencies([0.02] * 10)
    load_summary = LoadTestSummary(
        request_count=20,
        failure_count=0,
        failure_rate=0.0,
        p50_ms=25.0,
        p95_ms=80.0,
        p99_ms=100.0,
        throughput_per_second=4.0,
        source_path="locust.csv",
    )
    reliability = (
        ReliabilityScenario(name="restart", passed=True, expected="ready", observed="ready"),
    )
    monkeypatch.setattr(runner, "SplitManifest", SimpleNamespace(read=lambda _path: manifest))
    monkeypatch.setattr(runner, "_benchmark_api", lambda _config: (api_summary, 0))
    monkeypatch.setattr(runner, "read_locust_summary", lambda _path: load_summary)
    monkeypatch.setattr(runner, "read_reliability_results", lambda _path: reliability)
    monkeypatch.setattr(runner, "get_git_version", lambda _path: None)

    result = run_benchmark(
        settings=settings,
        config=config,
        champion_loader=lambda _settings: champion,
        clock=lambda: datetime(2026, 8, 12, 12, 0, tzinfo=UTC),
    )
    report = render_markdown_report(result)

    assert config.output_json_path.is_file()
    assert config.output_markdown_path.is_file()
    assert result["all_evaluated_targets_passed"] is True
    assert "Direct batch 2 p50/p95/p99" in report
    assert "Bounded load test p50/p95/p99" in report
    assert "PASS `restart`" in report
