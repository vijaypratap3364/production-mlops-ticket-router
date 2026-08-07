# PostgreSQL persistence and retention

Stage 9 adds SQLAlchemy 2 repositories and the initial Alembic schema for privacy-safe prediction
metadata, delayed feedback, monitoring runs, and retraining runs. PostgreSQL is the production-style
target. SQLite support exists only to provide fast, disposable migration and repository tests when
local PostgreSQL is unavailable.

## Schema

`prediction_events` uses the API `request_id` UUID as its primary key. It records the UTC timestamp,
numeric MLflow model version, predicted queue, confidence, top-k JSON, input lengths and word count,
optional language indicator, low-confidence flag, inference latency, one-way text fingerprint, and
the allowlisted `client_name`/`correlation_id` metadata. It has no `subject` or `body` columns.

`feedback_events` uses a UUID primary key and a foreign key to `prediction_events.request_id`. It
records one corrected label per prediction, nullable acceptance, a bounded optional comment, source,
UTC timestamp, and the model version copied from the original prediction. A unique constraint on
`request_id` implements a deliberate create-once policy: duplicate feedback receives HTTP 409 and
is not silently overwritten.

`monitoring_runs` stores UTC reference/current windows, drift status, local report paths, and summary
JSON. `retraining_runs` stores the trigger, source-data period, status, optional MLflow/candidate
identifiers, gate results, and timestamps. Neither table stores ticket text.

PostgreSQL uses native UUID, `TIMESTAMPTZ`, and JSONB through SQLAlchemy dialect variants. Foreign
keys, unique constraints, nonnegative numeric checks, period-order checks, and operational indexes
are created by migration `20260807_0001`.

## Privacy behavior

Raw subject and body are never persisted. The service fingerprints the shared preprocessed
`model_text` using HMAC-SHA-256 when `INPUT_HMAC_SECRET` is configured. A plain SHA-256 fallback is
available only for secret-free local development and is marked by `text_hash_algorithm`; production
deployments should always use a strong local HMAC secret.

`STORE_REDACTED_TICKET_TEXT=false` is the default. Setting it to `true` stores only the conservatively
preprocessed text after configured email, URL, and phone masking. This option is intended for a
controlled local demonstration and should remain disabled unless retention and access have been
reviewed. `STORE_RAW_TICKET_CONTENT` remains false and the schema has no raw-text columns.

Prediction analytics writes are best-effort: inference still returns when metadata persistence
fails, while `ticket_router_persistence_failures_total{operation="prediction"}` and a structured
content-free log event record the failure. Feedback writes are required because delayed labels must
join a real prediction; database outages return a structured 503.

## Alembic commands

Set `DATABASE_URL` to the local application database without putting it in source control:

```powershell
$env:DATABASE_URL = "postgresql+psycopg://ticket_router:LOCAL_PASSWORD@127.0.0.1:5432/ticket_router"
uv run alembic current
uv run alembic upgrade head
uv run alembic downgrade -1
```

Git Bash uses the same commands after `export DATABASE_URL="..."`. Create a reviewed migration after
changing ORM metadata with:

```bash
uv run alembic revision --autogenerate -m "describe schema change"
```

Inspect every generated revision before applying it. The API readiness check requires revision
`20260807_0001`; an empty or outdated database remains unready.

For an explicitly disposable local database whose name starts with `test_` or ends with `_test`:

```bash
export TEST_DATABASE_URL="postgresql+psycopg://ticket_router:LOCAL_PASSWORD@127.0.0.1:5432/ticket_router_test"
uv run pytest -m integration tests/integration/db
```

The PostgreSQL integration test downgrades, upgrades, validates, and finally downgrades that database.
Its naming guard is intended to reduce accidental use against a non-test database.

## Local reset

Resetting destroys all application prediction, feedback, monitoring, and retraining records. Only
run this against a confirmed local development database after verifying `DATABASE_URL`:

```bash
uv run alembic downgrade base
uv run alembic upgrade head
```

This does not reset the separate MLflow or Prefect databases.

## Retention policy

Until an automated cleanup flow is implemented, apply these documented local-development limits:

- prediction metadata: 90 days;
- redacted text, when explicitly enabled: 30 days;
- feedback: 365 days, or sooner when no longer needed for delayed-label evaluation;
- monitoring summaries: 365 days; generated detailed reports follow the same limit;
- retraining lineage and gate results: retain for the life of the portfolio project unless they
  contain an identified sensitive value, in which case remove it promptly through a reviewed audit.

Cleanup automation is intentionally deferred to the monitoring/orchestration stage. Even hashed and
aggregate metadata must be treated as sensitive: use separate local roles, do not expose PostgreSQL
to the public network, rotate the HMAC secret under a documented policy, and never write credentials,
database URLs, or secret values to tables or logs.
