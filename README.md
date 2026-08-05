# Production MLOps Ticket Router

A local-first, production-style machine-learning system for routing English customer-support
tickets to the correct support queue. The project is designed as a public portfolio repository
and uses only open-source libraries and local services.

> **Current status:** Stage 3 data validation and analysis. Model training, serving, persistence,
> monitoring, orchestration, and the dashboard are intentionally not implemented yet.

## Scope

The eventual classifier will use only `subject` and `body` as predictive inputs and `queue` as the
target. Agent-assigned or post-submission fields—including answers, tags, priority, and ticket
type—are excluded from model features to prevent leakage.

The requested source dataset is
[`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets).
The repository does not redistribute data. `configs/base.yaml` records the immutable upstream
revision that future ingestion code must use.

## Stage 1 capabilities

- Python 3.12 package using a `src` layout.
- Exact dependency resolution recorded in `uv.lock`.
- Named dependency groups for runtime, development, monitoring, orchestration, and dashboard work.
- Typed YAML and environment settings powered by Pydantic Settings.
- Structured JSON or local console logging through structlog.
- Ruff, mypy, pytest, coverage, and pre-commit configuration.
- Cross-platform canonical `uv` commands plus optional Makefile aliases.
- Unit tests for settings, logging, package imports, and bounded cleanup behavior.

## Stage 2 capabilities

- Revision-pinned Hugging Face snapshot download with verified offline cache reuse and explicit
  `--force` replacement.
- Raw-data and normalization manifests with SHA-256 lineage, UTC timestamps, configuration hashes,
  upstream license, and Git code identity.
- English-only normalization to Parquet with stable record/source identifiers and explicit row-drop
  counts, including structurally malformed CSV records.
- Modeling feature allowlist for `subject`, `body`, and their derived `text`, with reusable leakage
  rejection.
- Network-mocked ingestion tests using a tiny synthetic CSV fixture.

## Stage 3 capabilities

- Strict Pandera contract for normalized IDs, language, source text, derived text, target, and
  configuration-driven technical length bounds.
- Exact normalized-text, contradictory-label, near-empty, and conservative template-duplicate
  analysis without reporting ticket examples.
- Deterministic top-ten queue selection based on configured support and split-feasibility rules.
- Reproducible aggregate JSON/HTML EDA and authoritative selected-class artifacts.
- Data card documenting provenance, intended use, leakage controls, privacy, and limitations.

## Prerequisites

- Git
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- Docker Desktop will be required in a later stage, but it is not needed for Stage 1.

No cloud account, hosted model API, credit card, or paid service is required.

## Setup

Clone and enter the repository, then install a managed Python 3.12 runtime and all dependency
groups:

```powershell
git clone https://github.com/vijaypratap3364/production-mlops-ticket-router.git
Set-Location production-mlops-ticket-router
uv python install 3.12
uv sync --locked --all-groups
Copy-Item .env.example .env
```

The same `uv` commands work from Bash; replace the last line with `cp .env.example .env`.
The generated `.env` is ignored by Git. Replace placeholder secrets before using any later service
stage.

Confirm the package and settings load:

```powershell
uv run python -c "import ticket_router; print(ticket_router.__version__)"
uv run python -c "from ticket_router.config import Settings; print(Settings.load().project_name)"
```

## Development commands

The `uv` commands are canonical and work in PowerShell and Bash:

| Task | Portable command | Optional Make alias |
|---|---|---|
| Install all groups | `uv sync --locked --all-groups` | `make install` |
| Format | `uv run ruff format .` then `uv run ruff check --fix .` | `make format` |
| Lint | `uv run ruff check .` | `make lint` |
| Type-check | `uv run mypy src tests scripts` | `make typecheck` |
| Test with coverage | `uv run pytest` | `make test` |
| Run all non-mutating checks | See commands below | `make check` |
| Download pinned data | `uv run python -m ticket_router.data.download` | `make download-data` |
| Normalize raw data | `uv run python -m ticket_router.data.normalize` | `make normalize-data` |
| Validate and analyze data | `uv run python -m ticket_router.data.analyze` | `make analyze-data` |
| Remove project caches | `uv run python scripts/clean.py` | `make clean` |

Run the complete Stage 1 quality gate:

```powershell
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests scripts
uv run pytest
uv run pre-commit run --all-files
```

## Configuration

Non-secret, reproducibility-sensitive values live in `configs/base.yaml`, including the seed,
pinned dataset revision, English filter, target/text columns, target queue count, provisional
minimum class support, split ratios, local MLflow URI, and model name.

Deployment-specific values come from environment variables or an ignored `.env` file. Environment
values such as `GLOBAL_RANDOM_SEED` and `MLFLOW_TRACKING_URI` override their versioned YAML defaults
through the central `Settings` object. Do not add credentials to YAML, source code, MLflow
parameters, or committed documentation.

## Repository boundaries

- `src/ticket_router/data`: future reproducible ingestion and validation.
- `src/ticket_router/features`: future leakage-safe text features.
- `src/ticket_router/modeling`: future baseline and experiment logic.
- `src/ticket_router/registry`: future candidate/champion registry operations.
- `src/ticket_router/api`: future FastAPI transport adapters.
- `src/ticket_router/db`: future SQLAlchemy persistence.
- `src/ticket_router/monitoring`: future Evidently and delayed-label quality monitoring.
- `src/ticket_router/orchestration`: future Prefect flows.
- `src/ticket_router/dashboard`: future Streamlit API client and presentation code.

Dashboard code must call application/API boundaries; modeling code must never import dashboard code.

## Data ingestion

Dataset files and generated manifests are local and ignored by Git. Reproduce ingestion with:

```powershell
uv run python -m ticket_router.data.download
uv run python -m ticket_router.data.normalize
uv run python -m ticket_router.data.analyze
```

Existing verified outputs are reused. Use `--force` only after reviewing why replacement is needed.
See `docs/data-source.md` for attribution, license terms, field selection, leakage exclusions,
manifest contents, and dataset limitations.

## Roadmap

- **Complete:** Stage 1—repository foundation, dependency lock, typed settings, logging, and tests.
- **Complete:** Stage 2—pinned download, manifests, English normalization, and leakage contract.
- **Complete:** Stage 3—Pandera validation, duplicate analysis, target selection, EDA, and data card.
- **TODO:** leakage-safe grouped train/validation/test splits and sealed-test loaders.
- **TODO:** EDA, sparse baselines, MLflow experiments, and Model Registry aliases.
- **TODO:** FastAPI, PostgreSQL/Alembic, feedback, and privacy-safe logging.
- **TODO:** Evidently monitoring, controlled retraining, Prefect, Streamlit, and Docker Compose.
- **TODO:** GitHub Actions, load tests, benchmark report, data card, and model card.

See `docs/implementation-plan.md` for the complete architecture, lifecycle, acceptance criteria, and
risk register. Unfinished sections are intentionally labeled and must not be represented as working.

## Results

The Stage 3 analysis validated 28,190 normalized English records and selected all ten eligible
observed queues. It flagged 4,495 exact duplicate groups and found no contradictory-label groups.
See `docs/data-card.md` for the measured aggregate profile. No model has been trained and no model
performance metric exists yet.

## License

**TODO—decision required:** choose the source-code license. The upstream dataset is currently marked
CC BY-NC 4.0 and retains its own terms; a future code license will not relicense the dataset.
