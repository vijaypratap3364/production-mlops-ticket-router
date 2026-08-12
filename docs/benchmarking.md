# Performance benchmarking and operational validation

Stage 15 measures the registered `champion` locally without training, opening the sealed test set,
or changing a model alias. The protocol uses synthetic, non-sensitive requests and stores generated
reports under ignored `artifacts/benchmarks/` paths.

## Reproduction protocol

Install every locked dependency group and start the already-bootstrapped local Compose stack:

```powershell
uv sync --locked --all-groups
docker compose up -d postgres mlflow migrate api dashboard
docker compose ps
```

Then run the checks in this order:

```powershell
uv run python -m ticket_router.benchmarking.contract
uv run python -m ticket_router.benchmarking.load_test
uv run python scripts/operational_validation.py --run-compose-disruptions
uv run python -m ticket_router.benchmarking
```

The same commands work in Bash. `make model-contract`, `make load-test`,
`make operational-validation-disruptions`, and `make benchmark` are convenience aliases. The
disruption command intentionally stops and recovers only this Compose project's MLflow and
PostgreSQL services and restarts its API. Run it only on a disposable local development stack and
inspect `docker compose ps` afterward.

`configs/benchmark.yaml` versions the sample counts, batch sizes, random seed, timeouts, output
paths, traffic defaults, hard safety caps, and targets. Remote load-test hosts are rejected by
default. The configured maximum is 25 users, a spawn rate of 10 users/second, and five minutes;
the measured default used 3 users, 1 user/second, and 30 seconds. The Locust subprocess receives
the configured seed, and each simulated user gets a stable derived seed.

## Measurement definitions

- Cold load measures `load_champion` once in a new benchmark process after Python imports. It does
  not purge operating-system or MLflow artifact caches.
- Direct inference measures the production `PredictionService` with in-memory persistence so model
  and preprocessing time are isolated from PostgreSQL.
- Batch throughput counts ticket items, not HTTP calls. The entire batch is vectorized in one model
  call.
- API inference sends loopback HTTP requests to `/predict`; it includes request validation,
  serialization, and PostgreSQL analytics logging.
- API response overhead is API p50 minus direct-service p50. It is descriptive rather than a strict
  decomposition because the two loops are measured separately.
- Process RSS is sampled before and immediately after champion loading with psutil. It includes the
  Python runtime and imported libraries, not only model allocations.
- Serialized size comes from the immutable MLflow version's recorded evaluation metric; the
  contract check independently verifies an in-memory joblib round trip.
- Percentiles use linear interpolation for direct/API observations. Load percentiles are Locust's
  approximate aggregate percentiles.

## Actual local run

The following values were measured on 2026-08-12 at 23:45:58 UTC. They describe one local run and
are not public service-level guarantees.

- OS: Windows 11 `10.0.22631`; Python 3.12.13 CPython, AMD64.
- CPU: AMD64 Family 25 Model 117, 6 physical and 12 logical cores.
- Total system memory reported by psutil: 5,896.80 MiB.
- Model: `ticket-router` version 1, alias `champion`.
- Training-data hash: `0ef52577bace28e211da26410b5c05e1c3faac43a55872f8b7ad3bac37def6d6`.
- Split-manifest hash: `42a89d794b3a175dd11f9fac3c4f537b6d5e993fdbd6dc0e48466ae91845b28c`.
- Benchmark-configuration hash: `22f93127fcb7fb3d91fd5cf2200892ce2df21634dfabe91475c334c05cc01590`.
- Git base commit: `dc16a19522bf82d60b641a08015d6e0997444806`; the worktree was dirty because
  Stage 15 was being measured before its commit.

| Measurement | p50 | p95 | p99 | Throughput |
|---|---:|---:|---:|---:|
| Direct single request | 5.639900 ms | 11.411565 ms | 12.464165 ms | 140.943413 requests/s |
| Direct batch of 8 | 20.007800 ms | 24.871245 ms | 24.958409 ms | 402.667167 items/s |
| Direct batch of 32 | 53.669200 ms | 68.952655 ms | 75.391571 ms | 567.638654 items/s |
| Loopback API single request | 15.274450 ms | 21.308660 ms | 23.411020 ms | 60.416171 requests/s |
| Bounded Locust aggregate | 23.000000 ms | 82.000000 ms | 310.000000 ms | 4.483186 requests/s |

Champion load took 3,698.732400 ms. RSS after load was 264.484375 MiB, with an
88,936,448-byte increase from the pre-load sample. The registered serialized model size was
9.565703 MiB. The API p50 overhead was 9.634550 ms, and all 40 API benchmark responses succeeded.

The authoritative Locust CSV used by the JSON report contains 131 completed requests and zero
failures. Locust's terminal shutdown summary displayed 132 requests; the one-request difference is
retained as a tool-output timing discrepancy rather than changing either observation.

## Target comparison

Targets were committed in configuration before the final measurements. They are local engineering
guardrails, not retrospectively adjusted results.

| Metric | Target | Measured | Result |
|---|---:|---:|---|
| Cold load | at most 30,000 ms | 3,698.732400 ms | Pass |
| Direct single p95 | at most 50 ms | 11.411565 ms | Pass |
| Batch-32 p95 per item | at most 10 ms/item | 2.154770 ms/item | Pass |
| Direct throughput | at least 20 requests/s | 140.943413 requests/s | Pass |
| RSS after load | at most 2,048 MiB | 264.484375 MiB | Pass |
| Serialized model | at most 100 MiB | 9.565703 MiB | Pass |
| API p95 | at most 250 ms | 21.308660 ms | Pass |
| API p50 overhead | at most 200 ms | 9.634550 ms | Pass |
| API error rate | at most 0 | 0 | Pass |
| Load-test p95 | at most 500 ms | 82 ms | Pass |
| Load-test throughput | at least 1 request/s | 4.483186 requests/s | Pass |
| Load-test failure rate | at most 0 | 0 | Pass |

All 12 evaluated targets passed. No optimization was applied because this run did not verify a
target-breaking bottleneck. Existing production-conscious choices—one-time model loading, batch
vectorization, and pooled database sessions—remain in place. The Locust p99 is materially above its
p95, so a longer soak test would be required before setting a production SLO or changing worker
configuration.

## Reliability and model contract

All nine measured reliability scenarios passed:

- malformed prediction input returned 422;
- an oversized batch returned 422;
- duplicate feedback returned 409 after the initial 201;
- a missing champion left `/health` alive while readiness and prediction returned 503;
- monitoring with no events returned `insufficient_data`;
- stopping MLflow after startup did not interrupt an already-loaded model prediction;
- with PostgreSQL stopped, prediction degraded gracefully to 200 while required feedback returned
  503;
- after an API restart, readiness recovered and prediction returned 200;
- the initial readiness prerequisite returned 200 with `ready=true`.

The champion contract passed input-schema, approved predictive-field, output-label, 2-by-10
probability-shape, probability normalization, label-mapping, top-three ordering, joblib round-trip,
and fixed-sample determinism checks. The independent joblib round trip serialized 14,185,129 bytes.

## Generated artifacts

- `artifacts/benchmarks/benchmark_results.json`: full-precision measurements, configuration,
  environment, lineage, contract results, reliability results, and target decisions.
- `artifacts/benchmarks/benchmark_report.md`: generated human-readable report.
- `artifacts/benchmarks/locust_stats.csv` and related Locust CSV/HTML files: load-test evidence.
- `artifacts/benchmarks/reliability_results.json`: operational scenario evidence.

These outputs are reproducible and intentionally ignored by Git because runtime reports can be
large or environment-specific. This document keeps the reviewed Stage 15 snapshot in the public
repository. Re-run the protocol after a model, dependency, API, database, or hardware change and
report regressions without overwriting them with preferred values.
