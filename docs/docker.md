# Local Docker Compose runbook

Stage 13 packages the project as a local-only development stack. It uses no hosted database,
artifact store, model API, cloud account, free trial, or paid service. All published ports bind to
`127.0.0.1` and are unavailable from other machines unless the operator deliberately changes the
configuration.

## Services and profiles

The default core stack contains PostgreSQL, MLflow, the one-shot Alembic migration service, FastAPI,
and Streamlit. PostgreSQL data and MLflow artifacts use named Docker volumes. The project network is
a dedicated bridge network.

The locked MLflow server image deliberately uses the psycopg2 SQLAlchemy driver for its PostgreSQL
backend. MLflow 3.15.1 passes registry version identifiers through its REST boundary as strings;
with psycopg 3 those values are bound as `VARCHAR` and PostgreSQL rejects comparisons against the
registry's integer version columns. The application API continues to use psycopg 3 independently.
MLflow defaults to one worker in this development stack because each server worker loads the full
MLflow runtime; increase `MLFLOW_WORKERS` only when the local Docker memory allocation can support it.

Optional profiles are deliberately separate:

- `orchestration` starts the local Prefect server, creates/registers the process work pool and
  paused-by-default deployments, and starts a worker;
- `bootstrap` runs the pinned download, normalization, analysis, split preparation, candidate
  experiment, single final evaluation, registration, and explicitly approved promotion sequence;
- `smoke` runs prediction, feedback, PostgreSQL persistence, API health/readiness, and dashboard
  connectivity checks against an already prepared stack.

Normal API startup never downloads data, fits a model, registers a candidate, or moves an alias.
If `champion` is absent, `/health` remains healthy and `/ready` returns not-ready until an operator
bootstraps or promotes a gated candidate and restarts the API.

## Fresh-machine prerequisites

Install the free Docker Desktop application (or another Compose-compatible local engine), Git, and
optionally `make`. Docker must provide the `docker compose` command. No Python installation is
required for the container-only path.

Clone the repository, enter it, and create an ignored local environment file:

```powershell
Copy-Item .env.example .env
```

```bash
cp .env.example .env
```

Before starting anything, replace `change-me-local-only` and
`replace-with-a-random-local-secret` in `.env` with new local values. Do not commit `.env`, paste its
contents into issues, or pass secrets as Docker build arguments. The Compose configuration injects
runtime values only; Dockerfiles never copy environment files.

## Build and start core services

Build all four project images:

```text
docker compose build api dashboard mlflow prefect-worker
```

For release/lineage-capable builds, pass the exact source identity. The Makefile Docker targets derive
these values automatically. In PowerShell, the equivalent explicit setup is:

```powershell
$env:SOURCE_GIT_COMMIT = git rev-parse HEAD
$env:SOURCE_GIT_DIRTY = if (git status --porcelain) { "true" } else { "false" }
docker compose build api dashboard mlflow prefect-worker
```

An empty source commit remains allowed for ordinary local development, but release attestation
refuses missing or dirty Git identity.

Start the core stack. The API waits for healthy PostgreSQL and MLflow and for the Alembic migration
container to finish successfully:

```text
docker compose up -d postgres mlflow migrate api dashboard
docker compose ps
```

The API may correctly report not-ready on a fresh repository because no champion exists yet. Review
container logs without displaying `.env`:

```text
docker compose logs --tail=200 postgres mlflow migrate api dashboard
```

Equivalent convenience commands are `make docker-build`, `make docker-up`, `make docker-status`,
and `make docker-logs`.

## Explicit initial bootstrap

Bootstrap downloads the public dataset and performs the full CPU-oriented sparse-text lifecycle. It
can take substantial time and memory and requires Internet access to Hugging Face. It is not a
health-check repair and is never triggered by `api` or `dashboard` startup.

After reviewing `configs/final_model.yaml`, explicitly run:

```text
docker compose --profile bootstrap run --rm bootstrap
docker compose restart api
```

`make bootstrap` performs those two commands. The bootstrap first checks registry aliases. It exits
without retraining when a champion already exists; when only a gated candidate exists, it runs the
separate `promote --approve` command. If final evaluation was interrupted after the one-time test
audit was opened, bootstrap invokes the post-evaluation recovery command and reuses the existing
MLflow run/model version; it never reopens the test split. Other audit-without-registry states fail
safely for operator review. Promotion failures remain failures and never silently move the alias.

Watch readiness after the API restart:

```text
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/ready
```

## Smoke test

Once `/ready` succeeds, run the synthetic, non-sensitive smoke path:

```text
docker compose --profile smoke run --rm smoke-test
```

The script returns nonzero unless all of these succeed:

1. API process health and champion/database readiness;
2. one synthetic prediction;
3. accepted feedback linked to that request;
4. one matching prediction and feedback row in PostgreSQL;
5. Streamlit health, whose container health check also verifies FastAPI connectivity.

The script prints identifiers and model output but never prints ticket content, database URLs, or
credentials. `make docker-smoke` is equivalent.

## Prefect orchestration

Start the optional local server, one-shot pool/deployment initializer, and worker:

```text
docker compose --profile orchestration up -d
docker compose --profile orchestration ps
```

Schedules remain disabled because `configs/orchestration.yaml` sets `schedules.enabled: false`.
Use the local Prefect UI or explicit CLI actions to run deployments. Monitoring and retraining
evaluation can be triggered from the running worker with `make docker-monitor` and
`make docker-retrain`. Neither command promotes a champion.

## Local service addresses

These are local development addresses, not public deployments:

| Service | Local address |
|---|---|
| PostgreSQL | `127.0.0.1:5432` |
| MLflow UI/API | `http://127.0.0.1:5000` |
| FastAPI | `http://127.0.0.1:8000` |
| FastAPI OpenAPI | `http://127.0.0.1:8000/docs` |
| Streamlit | `http://127.0.0.1:8501` |
| Prefect UI, optional | `http://127.0.0.1:4200` |

Change host ports in the ignored `.env` if they conflict with another local program.

## Development override

The tracked `compose.override.example.yaml` mounts only `src/` read-only and points Python imports at
that mount. To opt in:

```powershell
Copy-Item compose.override.example.yaml compose.override.yaml
```

```bash
cp compose.override.example.yaml compose.override.yaml
```

The generated `compose.override.yaml` is ignored. It enables debug logging, so do not use it with
sensitive tickets.

## Persistence, shutdown, and reset

Stop containers while preserving PostgreSQL and MLflow state:

```text
docker compose --profile orchestration down
```

`make docker-down` is equivalent for the core stack. Named volumes and the bind-mounted ignored
`data/`, `artifacts/`, and `reports/` directories survive ordinary restarts.

`docker compose down --volumes` permanently deletes the local PostgreSQL and MLflow named volumes.
Use it only when an intentional full reset is required and after confirming no local experiment or
feedback history is needed.

## Troubleshooting

- If migration exits nonzero, inspect `docker compose logs migrate` before restarting the API.
- If `/health` passes but `/ready` fails, inspect the MLflow URI, registered-model name, `champion`
  alias, database migration status, and API logs. Do not retrain merely to make readiness green.
- If Prefect cannot start, confirm the PostgreSQL `prefect` database was created during the first
  volume initialization. Existing volumes do not rerun init scripts automatically.
- If credentials or database names change after the PostgreSQL volume was initialized, migrate or
  deliberately recreate the volume; changing `.env` alone does not rewrite existing database state.
- Use `docker compose config --quiet` to validate interpolation after `.env` is present. The expanded
  output can contain local connection strings, so do not publish it.
