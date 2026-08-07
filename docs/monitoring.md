# Batch monitoring

Stage 10 monitors the champion with local PostgreSQL prediction metadata and Evidently 0.7. Raw
free-form ticket text is never sent to Evidently. The monitoring contract contains only explicit
numeric aggregates, the predicted queue, confidence, low-confidence flag, and immutable model
version. Training labels and delayed corrected labels are analysis-only quality fields, never model
inputs.

## Reference and current windows

The reference builder combines the same train and validation splits used to fit the final Stage 7
pipeline. It never opens the held-out test split. The champion alias scores those records in bounded
batches, and the generated `data/reference/monitoring_reference.parquet` contains no subject, body,
model text, target label, record ID, or source metadata. Its adjacent manifest records the file hash, source split
hashes, split-manifest hash, champion name/version/alias, creation time, code version, threshold, and
feature definitions. The macro-F1 comparison baseline comes from the champion's immutable Stage 7
`test_macro_f1` registry tag; the builder never reopens the test split. Both generated files are
ignored by Git.

Current batches come from `prediction_events` over a half-open UTC window (`start <= created_at <
end`). The CLI defaults to seven days and supports `--lookback-hours 24`, `--lookback-days 7`, or an
explicit offset-aware `--start` plus optional `--end`. It filters to the reference champion version
unless `--model-version` is explicitly supplied. This prevents deployments from being mixed
silently in one comparison. Rows created before migration `20260807_0002` are excluded because their
new aggregate fields were necessarily backfilled with zero and cannot be reconstructed without raw
text.

## Feature contract

- `subject_length`, `body_length`, and `combined_length`: Unicode code-point counts, with combined
  length excluding model field markers.
- `word_count`: whitespace-delimited count after the shared masking/normalization policy.
- `uppercase_ratio`: uppercase letters divided by all cased letters.
- `digit_ratio` and `punctuation_ratio`: counts divided by non-whitespace characters.
- `url_count` and `email_marker_count`: pattern matches before masking, without retaining matches.
- `predicted_queue`, `prediction_confidence`, and `low_confidence`: model-output monitoring fields.

Known-vocabulary coverage is intentionally omitted for now. MLflow's generic pyfunc boundary does
not guarantee access to an estimator-specific vocabulary, and the API must not grow a second model
contract merely for a nonessential metric. Sparse-vector density can be added later through a
reviewed, versioned serving contract.

## Drift, quality, and alerts

Evidently receives declared numerical and categorical columns. Numerical fields use normalized
Wasserstein distance; predicted-label and low-confidence distributions use Jensen-Shannon distance.
The JSON/HTML report contains per-column values. The concise summary separately records drifted
input-feature share, new/missing predicted classes, confidence drift, low-confidence-rate change,
and combined-length change.

Feedback is joined to its original prediction to calculate macro F1, weighted F1, per-class recall,
correction rate, nullable acceptance rate, confidence for correct versus incorrect predictions, and
quality by model version. These metrics are marked unavailable until the configured feedback minimum
is met. Drift without labels is not described as model-quality regression.

Statuses are `healthy`, `warning`, `critical`, and `insufficient_data`. Thresholds live in
`configs/monitoring.yaml`. A warning or critical status requires at least two independent signals
from drifted-feature share, prediction-distribution shift, low-confidence increase, and—only when
enough feedback exists—macro-F1 decline. No single metric can create an alert by itself.

## Commands

Start the local MLflow registry and promote an eligible candidate to `champion` before building the
reference. Apply the latest application migration and keep PostgreSQL local:

```powershell
uv run alembic upgrade head
uv run python -m ticket_router.monitoring.build_reference
uv run python -m ticket_router.monitoring.run --lookback-days 7
uv run python -m ticket_router.monitoring.run --lookback-hours 24 --minimum-events 50
uv run python -m ticket_router.monitoring.run --start 2026-08-01T00:00:00Z --end 2026-08-08T00:00:00Z
uv run python -m ticket_router.monitoring.simulate_drift
```

The deterministic simulation uses seed 42 and aggregate synthetic rows only. It plants longer
messages, a shifted prediction distribution, and lower confidence. Its outputs under
`artifacts/reports/monitoring/simulation/` are temporary generated reports and must not be committed.

Operational reports and `monitoring_runs` follow the 365-day local retention policy documented in
`docs/database.md`. A monitoring status is evidence for investigation, not permission to retrain or
move a model alias automatically.
