# Contributing

Contributions that improve reproducibility, leakage prevention, tests, local operations, privacy,
or documentation are welcome. This is a zero-cost local portfolio project; proposals must not add
paid APIs, hosted inference, required cloud infrastructure, or credit-card-gated services.

## Development setup

```bash
git clone https://github.com/vijaypratap3364/production-mlops-ticket-router.git
cd production-mlops-ticket-router
uv python install 3.12
uv sync --locked --all-groups
uv run pre-commit install --install-hooks
```

Copy `.env.example` to an ignored `.env` only when local services are needed. Replace placeholders
locally and never attach `.env`, data, model files, databases, or reports to a pull request.

## Architecture and data rules

- Keep domain, data, ML, registry, persistence, monitoring, and orchestration logic under
  `src/ticket_router`; FastAPI and Streamlit remain adapters.
- The dashboard may call FastAPI but may not import modeling code or query PostgreSQL.
- Only `subject` and `body` may be source predictors; `queue` is the separate target. Preserve the
  feature allowlist and reject post-submission fields.
- Fit learned preprocessing and vocabulary on training data only. Do not inspect the sealed test
  split for development or rerun final evaluation to tune a model.
- Do not move `candidate` or `champion` aliases in tests or ordinary development commands.
- Use synthetic, non-sensitive test fixtures and type hints for production code.

Read `AGENTS.md`, the [implementation plan](docs/implementation-plan.md), and relevant
[ADRs](docs/adr/README.md) before changing an architectural boundary.

## Quality gate

Add meaningful tests for every behavior change. Run narrower tests while iterating and the complete
gate before opening a pull request:

```bash
uv sync --locked --all-groups
uv run ruff format .
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests scripts load_tests
uv run pytest
uv run pre-commit run --all-files
git diff --check
```

Integration tests require a disposable PostgreSQL database with a guarded test name. The full-data
pipeline, registered champion, and Compose e2e path are opt-in and must never run as a hidden side
effect of a unit test.

## Pull requests

Keep changes focused and explain:

- the problem and chosen design;
- affected data/model/privacy contracts;
- commands actually run and their exit results;
- tests added or changed;
- generated artifacts intentionally omitted;
- known limitations and follow-up work.

Do not fabricate metrics or write “passed” for a command that was not observed. If a result was not
verified locally, label it unverified. Add or update an ADR for decisions that change data leakage,
model selection, registry promotion, persistence/privacy, service boundaries, or deployment.

## Licenses and attribution

Contributions to original project code are accepted under the repository's MIT License. The
upstream synthetic dataset is a separate CC BY-NC 4.0 work and must not be committed or relicensed.
