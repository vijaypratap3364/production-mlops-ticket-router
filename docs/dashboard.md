# Streamlit dashboard

The Stage 12 dashboard is a local demonstration client for FastAPI. It never imports the fitted
pipeline, resolves an MLflow alias, constructs SQL, or opens PostgreSQL. Every page uses the typed
HTTP client in `ticket_router.dashboard.api_client`.

## Prerequisites

Install the locked environment and copy the safe environment template:

```powershell
uv sync --locked --all-groups
Copy-Item .env.example .env
```

The API requires an MLflow model assigned to the configured `champion` alias. The reviewed local
portfolio run promoted `ticket-router` version 1, but a clean checkout contains no MLflow database
or model artifact. Bootstrap the local stack or review and promote an eligible candidate:

```powershell
uv run python -m ticket_router.registry.promote --approve
```

Promotion is not part of dashboard startup. The dashboard never trains, registers, or promotes a
model.

## Start locally

Use two PowerShell or Bash terminals from the repository root. Start FastAPI first:

```text
uv run uvicorn ticket_router.api.main:app --host 127.0.0.1 --port 8000
```

Then start Streamlit:

```text
uv run streamlit run src/ticket_router/dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501`. `make api-dev` and `make dashboard-dev` are equivalent convenience
commands. Set `DASHBOARD_API_URL` if FastAPI uses another local address. The default request timeout
is ten seconds and can be changed with `DASHBOARD_REQUEST_TIMEOUT_SECONDS`.

## Pages and data availability

- **Route a ticket** sends `subject` and `body` to `/predict`, renders the top three calibrated
  results and API round-trip latency, and submits delayed labels to `/feedback`.
- **Batch routing** accepts a UTF-8 CSV, selects only `subject` and `body`, enforces the configured
  API limits, and sends one request to `/predict/batch`. The download contains row numbers and
  prediction metadata, not a duplicate copy of ticket text.
- **Monitoring** reads completed summaries from `/monitoring/history`. Drift charts require a
  drift-capable run; delayed-label scores remain unavailable until the configured feedback minimum
  is met. Missing values are labeled rather than synthesized.
- **Model information** reads champion identity, selected labels, test macro F1, model size,
  training-data hash, creation date, purpose, and limitations from `/model`.
- **System status** combines `/health`, `/ready`, and `/system/status` for API, database, champion,
  monitoring-run, and retraining-run status.

Monitoring and operational history require PostgreSQL plus applied Alembic migrations. In the
database-free demo mode, prediction/feedback storage is in memory and operational-history panels
truthfully show that no durable runs are available.

## CSV contract

Required headers:

```csv
subject,body
Invoice question,Please help me review this charge
,My local network is unavailable
```

Each row must contain usable text in at least one field. Subject/body lengths and the total row
count are checked before any request. Extra columns are shown as ignored and are never sent to the
API. Uploads over 5 MiB are rejected by the dashboard.

## Privacy and failure behavior

Ticket contents are displayed only to the current Streamlit browser session for preview and are not
included in dashboard logs. The API persists only its Stage 9 privacy-safe metadata by default.
Batch result downloads omit subject/body. Client errors expose sanitized API messages and never
attach request payloads or response bodies to exceptions.
