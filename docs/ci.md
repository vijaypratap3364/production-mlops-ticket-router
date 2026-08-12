# Quality checks and continuous integration

Stage 14 makes the same core checks available locally and in GitHub Actions. Pull-request checks use
small deterministic fixtures: they do not download the full ticket dataset, run candidate training,
open the sealed test split, depend on local MLflow artifacts, or change registry aliases.

## Local quality gate

Install the exact locked environment and run the complete gate from PowerShell or Bash:

```text
uv sync --locked --all-groups
uv run ruff format .
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests scripts
uv run pytest
uv run pre-commit run --all-files
uv build
```

`uv run pytest` collects unit, integration, and end-to-end directories. Infrastructure-dependent
tests skip with an explicit reason when their opt-in environment is absent. The aggregate run
enforces 80% branch-aware coverage over `ticket_router`. Coverage omits only executable entry-point
shims that delegate immediately to a tested `main`. Generated Alembic migration files live outside
the measured package; their upgrade/downgrade and constraint behavior is instead verified against a
disposable PostgreSQL database in the integration suite.

The narrower commands are:

```text
uv run pytest tests/unit
uv run pytest tests/integration --no-cov
uv run pytest tests/e2e --no-cov
```

Unit tests are network-free and use synthetic, non-sensitive fixtures. Integration tests use
`TEST_DATABASE_URL` when a live disposable PostgreSQL database is available. To exercise the local
stack end to end, first prepare and start it as documented in `docs/docker.md`, then set:

```text
RUN_COMPOSE_E2E=1
COMPOSE_E2E_DATABASE_URL=postgresql+psycopg://<user>:<password>@127.0.0.1:5432/<database>
uv run pytest tests/e2e --no-cov
```

Keep the database URL in the shell or ignored `.env`; never commit it.

## GitHub Actions

`.github/workflows/ci.yml` runs for pull requests targeting `main` and pushes to `main`:

| Job | Boundary |
|---|---|
| Ruff | Formatting and lint checks |
| mypy | Strict static checking across source, tests, and scripts |
| Unit tests and coverage | Deterministic fixture suite with the 80% coverage gate |
| PostgreSQL integration | Integration suite against a fresh PostgreSQL 16 service database |
| Python package build | Locked source distribution and wheel build |
| Docker build | API, dashboard, MLflow, and Prefect worker images; never pushed |

The uv cache is keyed from `uv.lock`. Docker layer caches are scoped by image. Generated datasets,
trained models, MLflow artifacts, and test outputs are not cached, so a stale model cannot make a
quality check pass. Actions are pinned to immutable commit revisions.

The integration database credentials are CI-only values scoped to the disposable service. They are
not production secrets. The workflow does not call paid services and requires no cloud account.

## Release and full training boundaries

`.github/workflows/release.yml` is triggered manually or by a `v*` tag. It builds downloadable
Python distributions only. It does not publish packages or images, deploy infrastructure, or use
deployment credentials.

Full ingestion, candidate training, final evaluation, and promotion remain deliberate local
commands. They are excluded from regular CI because they require the external dataset and persistent
MLflow lineage, consume substantially more resources, and must preserve the one-time test-evaluation
contract. Add a manual full-training workflow only after its artifact storage, authorization, and
sealed-test policy have been reviewed.
