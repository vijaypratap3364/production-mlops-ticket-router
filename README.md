# Production MLOps Ticket Router

A local-first, production-style machine-learning system for routing English customer-support
tickets to the correct support queue. The project is designed as a public portfolio repository
and uses only open-source libraries and local services.

> **Current status:** Stage 9 PostgreSQL persistence and delayed-label storage. Explicit champion
> promotion is available but has not been approved. Monitoring, orchestration, and the dashboard
> remain intentionally unfinished.

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

## Stage 4 capabilities

- Stateless conservative text preprocessing with Unicode/whitespace normalization and configurable
  email, URL, and phone masking while preserving useful punctuation and words.
- Seeded 70/15/15 grouped-stratified splitting that prevents exact normalized-text duplicates from
  crossing splits and excludes contradictory-label groups.
- Hashed split manifest, privacy-safe training reference, aggregate split summary, and automatic
  disjointness/class-proportion validation.
- Split-aware training loaders that expose only `model_text` and keep test data behind an explicit
  final-evaluation authorization boundary.

## Stage 5 capabilities

- Most-frequent DummyClassifier, word TF-IDF plus balanced logistic regression, and word TF-IDF
  plus ComplementNB baselines fitted only on training data.
- Macro F1 model selection with validation accuracy, weighted F1, precision/recall, per-class
  metrics, confusion matrices, valid probability metrics, timing, and model-size measurements.
- Run-scoped serialized pipelines, validation predictions, lineage/environment metadata, and
  token-redacted error analysis.
- Machine-readable validation leaderboard with explicit `test_evaluated=false` lineage.

## Stage 6 capabilities

- Local MLflow tracking with a SQLite backend fallback when the configured server is unavailable;
  remote tracking remains environment-configurable for the later PostgreSQL-backed service.
- Restrained, seeded train-only cross-validation across word, character, combined word/character,
  calibrated LinearSVC, and ComplementNB sparse-text candidates.
- Validation-only candidate ranking with configurable per-class recall, major-class recall,
  latency, model-size, cross-validation stability, and train/validation-gap guardrails.
- Loadable MLflow model artifacts with safe synthetic input examples, signatures, lineage hashes,
  package versions, metrics, plots, and privacy-safe error analysis.

## Stage 7 capabilities

- Frozen Stage 6 winner refitted on train+validation and evaluated once on the sealed test split,
  with a persistent access audit that rejects repeated evaluation.
- Final metrics, per-class results, predictions, confusion matrix, latency distribution, model size,
  lineage, and token-redacted error analysis logged locally.
- MLflow Model Registry version `ticket-router/1` with alias `candidate`, load/signature/contract
  verification, and aliases instead of deprecated lifecycle stages.
- Configurable absolute and champion-relative promotion gates plus a separate explicit human command;
  `champion` remains unset until that command is intentionally approved.

## Stage 8 capabilities

- Lifespan-managed FastAPI service that resolves `champion` once, pins its numeric MLflow version,
  verifies calibrated probabilities, and never trains or reloads per request.
- Health, readiness, model metadata, single and bounded-batch prediction, delayed feedback, and
  Prometheus-compatible metrics endpoints with stable structured errors.
- Shared leakage-safe preprocessing, bounded Pydantic input contracts, calibrated top-k output,
  low-confidence warnings, request IDs, and privacy-safe structured logs.
- Dependency-injected tests that require neither a production MLflow server nor PostgreSQL; the
  database-free local mode retains an in-memory adapter.

## Stage 9 capabilities

- SQLAlchemy 2 models and repositories for privacy-safe prediction metadata, create-once delayed
  feedback, monitoring runs, and retraining lineage.
- Alembic initial migration using PostgreSQL UUID, timezone-aware timestamps, JSONB, foreign keys,
  uniqueness/check constraints, and operational indexes.
- Best-effort prediction analytics with failure metrics and safe logs; feedback remains strict,
  label-valid, foreign-key-backed, and duplicate-protected.
- HMAC-SHA-256 text fingerprints when a secret is configured, no raw subject/body columns, and
  explicit opt-in redacted-text storage with a documented retention policy.

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
| Prepare deterministic splits | `uv run python -m ticket_router.data.prepare` | `make prepare-data` |
| Train validation-only baselines | `uv run python -m ticket_router.modeling.train_baseline` | `make train-baselines` |
| Run tracked candidate search | `uv run python -m ticket_router.modeling.train_candidates` | `make experiment-candidates` |
| Final evaluation and registration | `uv run python -m ticket_router.registry.evaluate_final` | `make evaluate-final` |
| Explicit candidate promotion | `uv run python -m ticket_router.registry.promote --approve` | `make promote-candidate` |
| Run the local inference API | `uv run uvicorn ticket_router.api.main:app --host 127.0.0.1 --port 8000` | `make api-dev` |
| Apply application migrations | `uv run alembic upgrade head` | `make db-upgrade` |
| Revert one migration | `uv run alembic downgrade -1` | `make db-downgrade` |
| Build monitoring reference | `uv run python -m ticket_router.monitoring.build_reference` | `make build-monitoring-reference` |
| Monitor the last seven days | `uv run python -m ticket_router.monitoring.run --lookback-days 7` | `make monitor` |
| Verify planted drift | `uv run python -m ticket_router.monitoring.simulate_drift` | `make simulate-drift` |
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
minimum class support, split ratios, local MLflow URI, and model name. The bounded Stage 5 model and
benchmark settings live in `configs/baseline.yaml`; Stage 6 search and selection guardrails live in
`configs/experiments.yaml`; the frozen final pipeline and promotion gates live in
`configs/final_model.yaml`.

Deployment-specific values come from environment variables or an ignored `.env` file. Environment
values such as `GLOBAL_RANDOM_SEED` and `MLFLOW_TRACKING_URI` override their versioned YAML defaults
through the central `Settings` object. Do not add credentials to YAML, source code, MLflow
parameters, or committed documentation.

## Repository boundaries

- `src/ticket_router/data`: future reproducible ingestion and validation.
- `src/ticket_router/features`: future leakage-safe text features.
- `src/ticket_router/modeling`: validation-only baseline/candidate pipelines, MLflow tracking,
  evaluation, selection, and artifacts.
- `src/ticket_router/registry`: future candidate/champion registry operations.
- `src/ticket_router/api`: champion-backed FastAPI transport, prediction/feedback service, and metrics.
- `src/ticket_router/db`: SQLAlchemy models, sessions, repositories, privacy hashes, and migrations.
- `src/ticket_router/monitoring`: aggregate feature extraction, Evidently drift reports,
  delayed-label quality, and multi-signal alert policy.
- `src/ticket_router/orchestration`: future Prefect flows.
- `src/ticket_router/dashboard`: future Streamlit API client and presentation code.

Dashboard code must call application/API boundaries; modeling code must never import dashboard code.

## Data ingestion

Dataset files and generated manifests are local and ignored by Git. Reproduce ingestion with:

```powershell
uv run python -m ticket_router.data.download
uv run python -m ticket_router.data.normalize
uv run python -m ticket_router.data.analyze
uv run python -m ticket_router.data.prepare
uv run python -m ticket_router.modeling.train_baseline
uv run python -m ticket_router.modeling.train_candidates
```

Existing verified outputs are reused. Use `--force` only after reviewing why replacement is needed.
See `docs/data-source.md` for attribution, license terms, field selection, leakage exclusions,
manifest contents, and dataset limitations.

## Roadmap

- **Complete:** Stage 1—repository foundation, dependency lock, typed settings, logging, and tests.
- **Complete:** Stage 2—pinned download, manifests, English normalization, and leakage contract.
- **Complete:** Stage 3—Pandera validation, duplicate analysis, target selection, EDA, and data card.
- **Complete:** Stage 4—conservative preprocessing, grouped splits, manifests, and sealed-test loaders.
- **Complete:** Stage 5—validation-only sparse baselines, evaluation artifacts, and leaderboard.
- **Complete:** Stage 6—MLflow tracking, bounded candidate search, guardrails, and candidate selection.
- **Complete:** Stage 7—single-use final evaluation, candidate registration, and promotion gates.
- **Pending human action:** Explicit initial promotion from `candidate` to `champion`.
- **Complete:** Stage 8—FastAPI champion inference, feedback contract, metrics, and privacy-safe logs.
- **Complete:** Stage 9—PostgreSQL schema, Alembic migration, repositories, and API persistence.
- **Complete:** Stage 10—privacy-safe Evidently monitoring, delayed-label quality, and alert policy.
- **TODO:** Controlled retraining, Prefect, Streamlit, and Docker Compose.
- **TODO:** GitHub Actions, load tests, benchmark report, and remaining operational documentation.

See `docs/implementation-plan.md` for the complete architecture, lifecycle, acceptance criteria, and
risk register. Unfinished sections are intentionally labeled and must not be represented as working.
See `docs/api.md` for champion prerequisites, local startup, endpoint contracts, and curl examples.
See `docs/database.md` for the schema, migration workflow, reset precautions, and retention policy.
See `docs/monitoring.md` for feature definitions, thresholds, report interpretation, and commands.

## Results

The Stage 3 analysis validated 28,190 normalized English records and selected all ten eligible
observed queues. It flagged 4,495 exact duplicate groups and found no contradictory-label groups.
Stage 4 produced 19,729 training, 4,232 validation, and 4,229 sealed-test records without record-ID
or exact-text-group overlap. The Stage 5 incumbent scored 0.58822 validation macro F1. Stage 6
selected calibrated word TF-IDF + LinearSVC at 0.67812 validation macro F1 after train-only
cross-validation and guardrail checks. Its single authorized Stage 7 test evaluation scored 0.69681
macro F1 and registered `ticket-router` version 1 as `candidate`; `champion` remains unset. See
`docs/data-card.md`, `docs/baseline-modeling.md`, `docs/candidate-experimentation.md`, and
`docs/model-card.md` for measured details.

## License

**TODO—decision required:** choose the source-code license. The upstream dataset is currently marked
CC BY-NC 4.0 and retains its own terms; a future code license will not relicense the dataset.
