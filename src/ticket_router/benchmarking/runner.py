"""Reproducible local champion and loopback-API performance benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import httpx
import psutil

from ticket_router.api.metrics import APIMetrics
from ticket_router.api.model_loader import ChampionLoader, load_champion
from ticket_router.api.schemas import PredictionResponse, TicketRequest
from ticket_router.api.service import PredictionService
from ticket_router.benchmarking.config import (
    DEFAULT_BENCHMARK_CONFIG_PATH,
    BenchmarkConfig,
    BenchmarkTargets,
)
from ticket_router.benchmarking.contract import validate_champion_contract
from ticket_router.benchmarking.contracts import (
    BatchBenchmark,
    LatencySummary,
    LoadTestSummary,
    ReliabilityScenario,
    TargetEvaluation,
    summarize_latencies,
)
from ticket_router.config import Settings
from ticket_router.data.manifests import atomic_write_json, get_git_version
from ticket_router.data.split_manifest import SplitManifest
from ticket_router.db.repositories import InMemoryPredictionFeedbackRepository
from ticket_router.hashing import sha256_file, sha256_json

MIB = 1024.0 * 1024.0
SYNTHETIC_TICKETS = (
    TicketRequest(subject="Invoice question", body="Please explain this synthetic local charge."),
    TicketRequest(subject="Return request", body="A demo parcel needs a local return label."),
    TicketRequest(subject="Network outage", body="The demonstration service cannot connect."),
    TicketRequest(
        subject="Long synthetic technical request",
        body=" ".join(
            ["This is non-sensitive benchmark text describing a reproducible local network issue."]
            * 80
        ),
    ),
)


class BenchmarkExecutionError(RuntimeError):
    """Raised when a benchmark prerequisite or response contract is unavailable."""


def run_benchmark(
    *,
    settings: Settings,
    config: BenchmarkConfig,
    champion_loader: ChampionLoader = load_champion,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict[str, object]:
    """Measure one immutable champion and persist unmodified observations."""
    process = psutil.Process()
    memory_before_load = process.memory_info().rss
    load_started = perf_counter()
    champion = champion_loader(settings)
    cold_load_ms = (perf_counter() - load_started) * 1000.0
    memory_after_load = process.memory_info().rss
    if champion.training_data_hash is None:
        raise BenchmarkExecutionError("champion is missing its training-data hash")
    manifest = SplitManifest.read(config.split_manifest_path)
    if set(manifest.label_mapping) != set(champion.labels):
        raise BenchmarkExecutionError(
            "champion labels do not match the prepared-data label mapping"
        )
    if any(size > settings.api_settings.maximum_batch_size for size in config.batch_sizes):
        raise BenchmarkExecutionError("benchmark batch size exceeds the configured API maximum")

    contract = validate_champion_contract(
        champion=champion,
        settings=settings,
        split_manifest=manifest,
    )
    if not contract.passed:
        raise BenchmarkExecutionError("registered champion failed the prediction contract")
    model_size_bytes = champion.model_size_bytes or contract.serialized_roundtrip_size_bytes
    model_size_source = (
        "mlflow_run_metric" if champion.model_size_bytes is not None else "local_joblib_roundtrip"
    )
    service = PredictionService(
        champion=champion,
        api_settings=settings.api_settings,
        preprocessing=settings.project_config.preprocessing,
        store=InMemoryPredictionFeedbackRepository(),
        metrics=APIMetrics(),
    )
    direct_single = _benchmark_single_service(service, config)
    direct_batches = tuple(
        _benchmark_batch_service(service, config, size) for size in config.batch_sizes
    )
    api_summary, api_errors = _benchmark_api(config)
    api_error_rate = api_errors / config.api_iterations
    api_overhead_p50_ms = api_summary.p50_ms - direct_single.p50_ms
    load_summary = read_locust_summary(config.locust_stats_path)
    reliability = read_reliability_results(config.reliability_results_path)
    evaluations = _evaluate_targets(
        targets=config.targets,
        cold_load_ms=cold_load_ms,
        direct_single=direct_single,
        direct_batches=direct_batches,
        memory_after_load_mib=memory_after_load / MIB,
        model_size_mib=model_size_bytes / MIB,
        api_summary=api_summary,
        api_overhead_p50_ms=api_overhead_p50_ms,
        api_error_rate=api_error_rate,
        load_summary=load_summary,
    )
    git_version = get_git_version(Path.cwd())
    timestamp = clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
    result: dict[str, object] = {
        "schema_version": 1,
        "benchmark_timestamp_utc": timestamp,
        "environment": _environment_metadata(),
        "git_version": git_version.model_dump(mode="json") if git_version else None,
        "model": {
            "name": champion.model_name,
            "version": champion.model_version,
            "alias": champion.alias,
            "training_data_hash": champion.training_data_hash,
            "serialized_model_size_bytes": model_size_bytes,
            "serialized_model_size_mib": model_size_bytes / MIB,
            "serialized_model_size_source": model_size_source,
            "labels": list(champion.labels),
        },
        "lineage": {
            "split_manifest_sha256": sha256_file(config.split_manifest_path),
            "benchmark_configuration_sha256": sha256_json(config.model_dump(mode="json")),
        },
        "benchmark_configuration": config.model_dump(mode="json"),
        "cold_load": {
            "milliseconds": cold_load_ms,
            "definition": (
                "load_champion in a benchmark process before inference; operating-system and "
                "MLflow artifact caches are not cleared"
            ),
            "process_rss_before_load_bytes": memory_before_load,
            "process_rss_after_load_bytes": memory_after_load,
            "process_rss_after_load_mib": memory_after_load / MIB,
            "process_rss_load_delta_bytes": memory_after_load - memory_before_load,
        },
        "direct_inference": {
            "definition": "PredictionService with in-memory persistence in the benchmark process",
            "single_request": direct_single.model_dump(mode="json"),
            "batches": [batch.model_dump(mode="json") for batch in direct_batches],
        },
        "api_inference": {
            "definition": "loopback HTTP POST /predict including validation and PostgreSQL logging",
            "url": config.api_url,
            "single_request": api_summary.model_dump(mode="json"),
            "error_count": api_errors,
            "error_rate": api_error_rate,
            "response_overhead_p50_ms": api_overhead_p50_ms,
            "response_overhead_definition": "API p50 minus in-process PredictionService p50",
        },
        "load_test": load_summary.model_dump(mode="json") if load_summary else None,
        "model_contract": contract.model_dump(mode="json"),
        "reliability_scenarios": [item.model_dump(mode="json") for item in reliability],
        "targets": [evaluation.model_dump(mode="json") for evaluation in evaluations],
        "all_evaluated_targets_passed": all(item.passed for item in evaluations),
        "notes": [
            "Measurements describe this local machine and are not public deployment SLOs.",
            "Synthetic non-sensitive requests were used; raw ticket data was not logged.",
            "No model training, test-set evaluation, or registry alias mutation occurred.",
        ],
    }
    config.output_json_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(config.output_json_path, result)
    _atomic_write_text(config.output_markdown_path, render_markdown_report(result))
    return result


def _benchmark_single_service(
    service: PredictionService,
    config: BenchmarkConfig,
) -> LatencySummary:
    for index in range(config.warmup_iterations):
        service.predict_one(
            SYNTHETIC_TICKETS[index % len(SYNTHETIC_TICKETS)], request_id=f"warm-{index}"
        )
    observations: list[float] = []
    for index in range(config.single_request_iterations):
        started = perf_counter()
        service.predict_one(
            SYNTHETIC_TICKETS[index % len(SYNTHETIC_TICKETS)],
            request_id=f"single-{index}",
        )
        observations.append(perf_counter() - started)
    return summarize_latencies(observations)


def _benchmark_batch_service(
    service: PredictionService,
    config: BenchmarkConfig,
    batch_size: int,
) -> BatchBenchmark:
    batch = tuple(SYNTHETIC_TICKETS[index % len(SYNTHETIC_TICKETS)] for index in range(batch_size))
    service.predict_many(batch)
    observations: list[float] = []
    for _ in range(config.batch_repetitions):
        started = perf_counter()
        service.predict_many(batch)
        observations.append(perf_counter() - started)
    summary = summarize_latencies(observations, operations_per_observation=batch_size)
    return BatchBenchmark(
        batch_size=batch_size,
        batch_latency=summary,
        p95_ms_per_item=summary.p95_ms / batch_size,
    )


def _benchmark_api(config: BenchmarkConfig) -> tuple[LatencySummary, int]:
    payloads = [ticket.model_dump(mode="json") for ticket in SYNTHETIC_TICKETS]
    observations: list[float] = []
    errors = 0
    with httpx.Client(base_url=config.api_url, timeout=config.api_timeout_seconds) as client:
        readiness = client.get("/ready")
        if readiness.status_code != 200 or readiness.json().get("ready") is not True:
            raise BenchmarkExecutionError("API is not ready for loopback benchmarking")
        for index in range(config.warmup_iterations):
            response = client.post("/predict", json=payloads[index % len(payloads)])
            if response.status_code != 200:
                raise BenchmarkExecutionError("API warmup prediction failed")
        for index in range(config.api_iterations):
            started = perf_counter()
            try:
                response = client.post("/predict", json=payloads[index % len(payloads)])
                if response.status_code != 200:
                    errors += 1
                else:
                    PredictionResponse.model_validate(response.json())
            except (httpx.HTTPError, ValueError):
                errors += 1
            observations.append(perf_counter() - started)
    return summarize_latencies(observations), errors


def read_locust_summary(path: Path) -> LoadTestSummary | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    aggregate = next((row for row in rows if row.get("Name") == "Aggregated"), None)
    if aggregate is None:
        raise BenchmarkExecutionError(f"Locust aggregate row is missing: {path}")
    request_count = int(aggregate["Request Count"])
    failure_count = int(aggregate["Failure Count"])
    return LoadTestSummary(
        request_count=request_count,
        failure_count=failure_count,
        failure_rate=failure_count / request_count if request_count else 0.0,
        p50_ms=float(aggregate["Median Response Time"]),
        p95_ms=float(aggregate["95%"]),
        p99_ms=float(aggregate["99%"]),
        throughput_per_second=float(aggregate["Requests/s"]),
        source_path=path.as_posix(),
    )


def read_reliability_results(path: Path) -> tuple[ReliabilityScenario, ...]:
    if not path.is_file():
        return ()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not isinstance(loaded.get("scenarios"), list):
        raise BenchmarkExecutionError(f"reliability results have an invalid contract: {path}")
    return tuple(ReliabilityScenario.model_validate(item) for item in loaded["scenarios"])


def _evaluate_targets(
    *,
    targets: BenchmarkTargets,
    cold_load_ms: float,
    direct_single: LatencySummary,
    direct_batches: tuple[BatchBenchmark, ...],
    memory_after_load_mib: float,
    model_size_mib: float,
    api_summary: LatencySummary,
    api_overhead_p50_ms: float,
    api_error_rate: float,
    load_summary: LoadTestSummary | None,
) -> tuple[TargetEvaluation, ...]:
    largest_batch = max(direct_batches, key=lambda item: item.batch_size)
    evaluations = [
        _maximum("cold_load", targets.maximum_cold_load_ms, cold_load_ms, "ms"),
        _maximum(
            "direct_single_request_p95",
            targets.maximum_single_request_p95_ms,
            direct_single.p95_ms,
            "ms",
        ),
        _maximum(
            f"direct_batch_{largest_batch.batch_size}_p95_per_item",
            targets.maximum_batch_p95_ms_per_item,
            largest_batch.p95_ms_per_item,
            "ms/item",
        ),
        _minimum(
            "direct_single_request_throughput",
            targets.minimum_direct_throughput_per_second,
            direct_single.throughput_per_second,
            "requests/s",
        ),
        _maximum(
            "process_rss_after_load",
            targets.maximum_process_rss_after_load_mib,
            memory_after_load_mib,
            "MiB",
        ),
        _maximum(
            "serialized_model_size",
            targets.maximum_serialized_model_size_mib,
            model_size_mib,
            "MiB",
        ),
        _maximum("api_single_request_p95", targets.maximum_api_p95_ms, api_summary.p95_ms, "ms"),
        _maximum(
            "api_response_overhead_p50",
            targets.maximum_api_response_overhead_p50_ms,
            api_overhead_p50_ms,
            "ms",
        ),
        _maximum("api_error_rate", targets.maximum_api_error_rate, api_error_rate, "ratio"),
    ]
    if load_summary is not None:
        evaluations.extend(
            [
                _maximum(
                    "load_test_p95",
                    targets.maximum_load_test_p95_ms,
                    load_summary.p95_ms,
                    "ms",
                ),
                _minimum(
                    "load_test_throughput",
                    targets.minimum_load_test_throughput_per_second,
                    load_summary.throughput_per_second,
                    "requests/s",
                ),
                _maximum(
                    "load_test_failure_rate",
                    targets.maximum_load_test_failure_rate,
                    load_summary.failure_rate,
                    "ratio",
                ),
            ]
        )
    return tuple(evaluations)


def _maximum(metric: str, target: float, measured: float, unit: str) -> TargetEvaluation:
    return TargetEvaluation(
        metric=metric,
        target=target,
        measured=measured,
        comparison="<=",
        unit=unit,
        passed=measured <= target,
    )


def _minimum(metric: str, target: float, measured: float, unit: str) -> TargetEvaluation:
    return TargetEvaluation(
        metric=metric,
        target=target,
        measured=measured,
        comparison=">=",
        unit=unit,
        passed=measured >= target,
    )


def _environment_metadata() -> dict[str, object]:
    virtual_memory = psutil.virtual_memory()
    return {
        "operating_system": platform.platform(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "architecture": platform.machine(),
        "processor": platform.processor()
        or os.environ.get("PROCESSOR_IDENTIFIER")
        or platform.machine(),
        "physical_cpu_cores": psutil.cpu_count(logical=False),
        "logical_cpu_cores": psutil.cpu_count(logical=True),
        "total_memory_bytes": virtual_memory.total,
    }


def render_markdown_report(result: dict[str, object]) -> str:
    model = cast(dict[str, Any], result["model"])
    environment = cast(dict[str, Any], result["environment"])
    cold = cast(dict[str, Any], result["cold_load"])
    direct = cast(dict[str, Any], result["direct_inference"])
    single = cast(dict[str, Any], direct["single_request"])
    batches = cast(list[dict[str, Any]], direct["batches"])
    api = cast(dict[str, Any], result["api_inference"])
    api_single = cast(dict[str, Any], api["single_request"])
    load_test = cast(dict[str, Any] | None, result["load_test"])
    targets = cast(list[dict[str, Any]], result["targets"])
    lines = [
        "# Stage 15 benchmark report",
        "",
        f"Measured at: `{result['benchmark_timestamp_utc']}`",
        "",
        "## Lineage and environment",
        "",
        f"- Model: `{model['name']}` version `{model['version']}` alias `{model['alias']}`",
        f"- Training dataset hash: `{model['training_data_hash']}`",
        f"- Operating system: `{environment['operating_system']}`",
        f"- Python: `{environment['python_version']}`",
        f"- Processor: `{environment['processor']}`",
        f"- CPU cores: {environment['physical_cpu_cores']} physical / "
        f"{environment['logical_cpu_cores']} logical",
        f"- Total memory: {float(environment['total_memory_bytes']) / MIB:.2f} MiB",
        "",
        "## Measurements",
        "",
        f"- Champion load: {float(cold['milliseconds']):.6f} ms",
        f"- RSS after load: {float(cold['process_rss_after_load_mib']):.6f} MiB",
        f"- Serialized model: {float(model['serialized_model_size_mib']):.6f} MiB",
        f"- Direct single p50/p95/p99: {float(single['p50_ms']):.6f} / "
        f"{float(single['p95_ms']):.6f} / {float(single['p99_ms']):.6f} ms",
        f"- Direct single throughput: {float(single['throughput_per_second']):.6f} requests/s",
        f"- API single p50/p95/p99: {float(api_single['p50_ms']):.6f} / "
        f"{float(api_single['p95_ms']):.6f} / {float(api_single['p99_ms']):.6f} ms",
        f"- API response overhead p50: {float(api['response_overhead_p50_ms']):.6f} ms",
    ]
    for batch in batches:
        latency = cast(dict[str, Any], batch["batch_latency"])
        lines.append(
            f"- Direct batch {int(batch['batch_size'])} p50/p95/p99: "
            f"{float(latency['p50_ms']):.6f} / {float(latency['p95_ms']):.6f} / "
            f"{float(latency['p99_ms']):.6f} ms; "
            f"{float(latency['throughput_per_second']):.6f} items/s"
        )
    if load_test is not None:
        lines.extend(
            [
                f"- Bounded load test p50/p95/p99: {float(load_test['p50_ms']):.6f} / "
                f"{float(load_test['p95_ms']):.6f} / {float(load_test['p99_ms']):.6f} ms",
                f"- Bounded load test: {int(load_test['request_count'])} requests, "
                f"{int(load_test['failure_count'])} failures, "
                f"{float(load_test['throughput_per_second']):.6f} requests/s",
            ]
        )
    else:
        lines.append("- Bounded load test: not run; no Locust aggregate artifact was available")
    lines.extend(
        [
            "",
            "## Target comparison",
            "",
            "| Metric | Target | Measured | Result |",
            "|---|---:|---:|---|",
        ]
    )
    for evaluation in targets:
        lines.append(
            f"| {evaluation['metric']} | {evaluation['comparison']} "
            f"{float(evaluation['target']):.6f} {evaluation['unit']} | "
            f"{float(evaluation['measured']):.6f} {evaluation['unit']} | "
            f"{'PASS' if evaluation['passed'] else 'FAIL'} |"
        )
    reliability = cast(list[dict[str, Any]], result["reliability_scenarios"])
    lines.extend(["", "## Reliability validation", ""])
    if reliability:
        for scenario in reliability:
            lines.append(
                f"- {'PASS' if scenario['passed'] else 'FAIL'} `{scenario['name']}`: "
                f"expected {scenario['expected']}; observed {scenario['observed']}"
            )
    else:
        lines.append("No separate operational reliability result file was available.")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- These are local-machine measurements, not public service-level guarantees.",
            "- Cold load does not clear operating-system or MLflow artifact caches.",
            "- API overhead includes loopback HTTP, validation, serialization, and database logging.",
            "- Synthetic non-sensitive requests were used throughout.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=DEFAULT_BENCHMARK_CONFIG_PATH,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = Settings.load(args.config)
    config = BenchmarkConfig.load(args.benchmark_config)
    try:
        result = run_benchmark(settings=settings, config=config)
    except (
        BenchmarkExecutionError,
        ConnectionError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0
