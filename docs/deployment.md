# Deployment guide

The supported deployment is a zero-cost, localhost-only Docker Compose stack. It is production-
conscious—non-root images, health-gated startup, migrations, persistence, readiness, structured
errors, and explicit bootstrap—but it is not a public production deployment.

## Services

| Service | Responsibility | Local address |
|---|---|---|
| `postgres` | Application, MLflow and Prefect databases | `127.0.0.1:5432` |
| `mlflow` | Tracking UI/API, artifact metadata, registry | `http://127.0.0.1:5000` |
| `migrate` | One-shot Alembic upgrade before writes | no host port |
| `api` | Champion FastAPI inference | `http://127.0.0.1:8000` |
| `dashboard` | API-only Streamlit UI | `http://127.0.0.1:8501` |
| `prefect-server` | Optional local orchestration API/UI | `http://127.0.0.1:4200` |
| `prefect-worker` | Optional flow execution | no host port |
| `bootstrap` | Optional explicit full lifecycle | no host port |
| `smoke-test` | Optional complete-stack verification | no host port |

PostgreSQL and MLflow artifacts use named volumes. Data and generated reports use ignored local
directories. Normal API startup never downloads data, trains, evaluates, registers, or promotes.

## Fresh local deployment

```bash
git clone https://github.com/vijaypratap3364/production-mlops-ticket-router.git
cd production-mlops-ticket-router
cp .env.example .env
# Replace POSTGRES_PASSWORD and INPUT_HMAC_SECRET placeholders in .env.
docker compose config --quiet
docker compose build api dashboard mlflow prefect-worker
docker compose up -d postgres mlflow migrate api dashboard
docker compose ps
```

A clean registry has no champion, so `/health` can return 200 while `/ready` correctly returns 503.
The explicit bootstrap requires network access to the public Hugging Face dataset and performs the
full CPU model lifecycle:

```bash
docker compose --profile bootstrap run --rm bootstrap
docker compose restart api
curl --fail http://127.0.0.1:8000/ready
docker compose --profile smoke run --rm smoke-test
```

Bootstrap is idempotent around an existing champion. It does not repeatedly reopen the sealed test
set; interrupted final-registration states use the reviewed recovery workflow.

Start local Prefect only when needed:

```bash
docker compose --profile orchestration up -d
docker compose --profile orchestration ps
```

Schedules are disabled by default. Monitoring and retraining evaluation never promote a champion.

## Release attestation

After all local checks and GitHub Actions pass for the exact clean commit, bind that commit to the
existing champion before creating a version tag:

```powershell
uv run python -m ticket_router.registry.attest_release --release v1.0.0
```

This validates immutable split hashes, registry lineage, prediction behavior, and the model signature.
It records `release_git_commit` without overwriting an unavailable historical training commit, does
not load the test set for evaluation, and never changes `candidate` or `champion` aliases.

## Configuration and secrets

Versioned YAML owns non-secret reproducibility settings. `.env` owns local credentials, URLs,
ports, and privacy switches and is ignored. Do not use example credentials beyond localhost, pass
secrets as Docker build arguments, copy `.env` into images, or publish expanded Compose output.

Health checks establish startup order: PostgreSQL and MLflow become healthy, migration exits
successfully, then API/dashboard readiness is evaluated. The API resolves the `champion` alias to a
numeric version once at process start. If MLflow later becomes unavailable, the loaded model can
continue predicting; if PostgreSQL is unavailable, prediction analytics degrade without losing the
response, while feedback returns 503 because it requires a real prediction row.

## Operations

```bash
docker compose logs --tail=200 postgres mlflow migrate api dashboard
docker compose restart api
docker compose --profile orchestration down
```

Ordinary `down` preserves named volumes. `docker compose down --volumes` destroys PostgreSQL and
MLflow state and must be used only for an intentional local reset after confirming that experiment,
registry, feedback, and monitoring history is disposable.

See the detailed [Docker runbook](docker.md) for overrides, troubleshooting, MLflow driver details,
smoke behavior, and persistence.

## Production gap assessment

Before deploying outside localhost, add authentication/authorization, TLS, rate limiting, a secret
manager, least-privilege roles, network policies, backups and restore drills, high availability,
centralized logs/metrics, vulnerability and image scanning, retention automation, incident
response, resource/load validation, and a legal/privacy review. The CC BY-NC dataset should not be
assumed suitable for commercial operation.

## Verification status

Stage 13 built every project image, started the local stack, applied migrations, bootstrapped the
champion, and passed its Compose smoke path. Stage 15 additionally exercised controlled MLflow and
PostgreSQL outages plus an API restart. These were local measurements, not a public deployment.
Reviewer commands and current Stage 16 verification are recorded in [demo script](demo-script.md)
and the final handoff.
