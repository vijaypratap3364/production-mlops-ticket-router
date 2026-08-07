# Local inference API

Stage 8 provides a local FastAPI adapter that loads the MLflow model referenced by the `champion`
alias exactly once during application lifespan. It resolves that alias to an immutable numeric
version before loading, never trains a model, and never falls back to `candidate`.

## Prerequisites

The Stage 7 candidate must pass its frozen gates and be explicitly promoted before the API can
become ready:

```bash
uv run python -m ticket_router.registry.promote --approve
```

This is a human-controlled alias change. Do not run it merely to make an endpoint green; review the
recorded gate report first. With the local Stage 7 SQLite MLflow backend, point the API at the same
store:

```bash
export MLFLOW_TRACKING_URI="sqlite:///mlruns/mlflow.db"
export MLFLOW_REGISTERED_MODEL_NAME="ticket-router"
export MLFLOW_MODEL_ALIAS="champion"
uv run uvicorn ticket_router.api.main:app --host 127.0.0.1 --port 8000
```

PowerShell equivalents are:

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlruns/mlflow.db"
$env:MLFLOW_REGISTERED_MODEL_NAME = "ticket-router"
$env:MLFLOW_MODEL_ALIAS = "champion"
uv run uvicorn ticket_router.api.main:app --host 127.0.0.1 --port 8000
```

The API is then available at `http://127.0.0.1:8000`; interactive OpenAPI documentation is at
`http://127.0.0.1:8000/docs`. The MLflow UI at port 5000 is a separate process and is not the API.

If the champion is absent or cannot load, the process stays live: `/health` returns 200 while
`/ready` returns 503. No request causes model reload or retraining.

## Example requests

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/ready
curl --fail http://127.0.0.1:8000/model

curl --fail --request POST http://127.0.0.1:8000/predict \
  --header "Content-Type: application/json" \
  --data '{"subject":"Invoice question","body":"Please explain this charge."}'

curl --fail --request POST http://127.0.0.1:8000/predict/batch \
  --header "Content-Type: application/json" \
  --data '{"items":[{"body":"The network is unavailable."},{"body":"I need a return."}]}'

curl --fail --request POST http://127.0.0.1:8000/feedback \
  --header "Content-Type: application/json" \
  --data '{"request_id":"REPLACE_FROM_PREDICTION","corrected_queue":"Technical Support","accepted":false,"comment":"Reviewed demo ticket"}'

curl --fail http://127.0.0.1:8000/metrics
```

## Contracts and privacy

Only `subject` and `body` are predictive inputs. Optional `client_name` and `correlation_id`
metadata are operational fields and never enter the model. The API uses the shared conservative
preprocessor that produced Stage 4 `model_text`, including configured email, URL, and phone masking.
The champion must provide calibrated `predict_proba`; confidence is never synthesized.

Normal request logs contain a generated request ID, route, status, and latency—not the subject,
body, preprocessing output, feedback comment, database URL, or exception message. The default
Stage 8 local store retains only privacy-safe prediction fields plus submitted feedback in memory.
PostgreSQL tables, migrations, durable prediction logging, and retention behavior remain explicitly
unfinished and will replace this adapter in the database stage. If `DATABASE_URL` is configured,
startup probes it and readiness fails deterministically when it is unavailable.

The service binds to localhost by default and has no authentication. Do not expose it to an
untrusted network without TLS, authentication, rate limiting, and a reviewed deployment design.
