# Prefect orchestration and controlled retraining

Stage 11 composes the existing, tested application services; Prefect does not duplicate their
business logic. Every flow runs locally with Prefect 3, PostgreSQL, MLflow, and filesystem artifacts.
No hosted Prefect account, cloud service, paid API, or credit card is required.

## Safety boundary

`train_candidate_flow` and `conditional_retraining_flow` may register an MLflow model version and
move the `candidate` alias. They never call the champion-promotion operation. Even when every gate
passes, `champion` changes only through the separately reviewed command:

```powershell
uv run python -m ticket_router.registry.promote --approve
```

The API database intentionally contains no raw subject/body by default, so it cannot be treated as a
training corpus. A reviewer must export approved, appropriately governed labeled rows to the ignored
path `data/retraining/inbox/approved_labeled_tickets.parquet` with these columns:

| Column | Meaning |
|---|---|
| `request_id` | Stable source request identifier |
| `subject`, `body` | Reviewed ticket inputs; at least one must be usable |
| `queue` | Approved corrected target from the selected label set |
| `label_created_at` | Timezone-aware ISO timestamp or timestamp value |
| `approved` | Boolean human approval for training use |

The dataset builder filters the requested source period, rejects unknown labels and holdout duplicate
text, preprocesses with the shared Stage 4 function, appends only to training, and leaves validation
and test unchanged. It writes an immutable parent-linked manifest with included/excluded counts and
hashes under `data/retraining/runs/<dataset-id>/`. Generated tickets and artifacts remain ignored by
Git.

## Flows

- `ingest_data_flow`: pinned download, normalization, contract validation, EDA/class selection, and
  duplicate-safe splitting. Existing verified manifests are reused unless `--force` is explicit.
- `train_candidate_flow`: verifies train/validation hashes, runs train-only CV and validation
  comparison, logs MLflow runs, registers the selected candidate idempotently, and records gates.
- `monitoring_flow`: reads a bounded PostgreSQL period, calculates drift and delayed-label quality,
  writes reports, persists the monitoring run, and returns `healthy`, `warning`, `critical`, or
  `insufficient_data`.
- `conditional_retraining_flow`: runs monitoring first and requires sustained critical windows,
  enough new labels, and a material quality or low-confidence signal. A manual trigger still requires
  the configured label minimum and approved input. It stops after candidate registration and gates.

Tasks that can fail transiently on network or service boundaries use the configured retry count and
delay. Generated manifests, MLflow run IDs, and completed workflow summaries make reruns auditable
and idempotent. Model fitting and registry writes are never result-cached.

## Local server, worker, and deployments

Use two or three PowerShell terminals from the repository root. Keep Prefect state local:

```powershell
New-Item -ItemType Directory -Force .\prefect-data | Out-Null
$env:PREFECT_HOME = (Resolve-Path .).Path + "\prefect-data"
$env:PREFECT_SERVER_ANALYTICS_ENABLED = "false"
uv run prefect server start
```

In another terminal, point the CLI at the local server, create a process pool once, and register the
deployments:

```powershell
$env:PREFECT_HOME = (Resolve-Path .).Path + "\prefect-data"
$env:PREFECT_API_URL = "http://127.0.0.1:4200/api"
$env:PREFECT_SERVER_ANALYTICS_ENABLED = "false"
uv run prefect work-pool create --type process ticket-router-local
uv run python -m ticket_router.orchestration.deploy
uv run prefect worker start --pool ticket-router-local
```

The daily monitoring (`0 2 * * *`) and weekly retraining evaluation (`0 3 * * 0`) schedules are
versioned in `configs/orchestration.yaml`. `schedules.enabled` is `false` by default, so deployments
are registered paused and tests never start schedules. Review the environment and set it to `true`
before registering active local schedules.

Trigger flows directly without a server:

```powershell
uv run python -m ticket_router.orchestration ingest
uv run python -m ticket_router.orchestration train-candidate
uv run python -m ticket_router.orchestration monitor
uv run python -m ticket_router.orchestration retraining
uv run python -m ticket_router.orchestration retraining --manual-trigger
```

Or trigger registered deployments after starting the worker:

```powershell
uv run prefect deployment run "monitoring-flow/daily-monitoring"
uv run prefect deployment run "conditional-retraining-flow/weekly-retraining-evaluation"
```

The network-free engine smoke check is:

```powershell
uv run python -m ticket_router.orchestration.fixture_flow
```

Structured Prefect logs contain counts, statuses, paths, and IDs—not raw ticket text or secrets.
