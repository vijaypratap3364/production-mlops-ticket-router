# Repository Instructions for Coding Agents

These rules apply to the entire repository.

## Architecture rules

- Keep business and ML logic inside `src/ticket_router`; API and dashboard modules are adapters.
- Maintain one-way boundaries: dashboard → API/application service → domain/model/persistence adapter.
- Never import `ticket_router.dashboard` from modeling, features, data, registry, database, monitoring,
  or orchestration code. Dashboard code must not load model artifacts or access PostgreSQL directly.
- Data pipelines may use only `subject` and `body` as predictive inputs and `queue` as the target.
  Agent-assigned/post-submission fields must remain physically separated from model inputs.
- Fit preprocessing only on training data. The final test split remains sealed until candidate
  selection is frozen.
- Put secrets and deployment overrides in environment variables or an ignored `.env`; keep
  reproducibility-sensitive, non-secret values in versioned YAML.
- Keep generated datasets, models, databases, Docker volumes, MLflow artifacts, and temporary reports
  out of Git. Prefer reproducible generation over Git LFS.
- Use type hints for production code and keep modules focused on one responsibility.

## Commands Codex should run

Use Python 3.12 and the locked environment. Before handing off a behavior change, run:

```text
uv sync --locked --all-groups
uv run ruff format .
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests scripts load_tests
uv run pytest
uv run pre-commit run --all-files
```

Use the portable `uv` commands as the source of truth; Makefile targets are convenience aliases.
Run narrower tests while iterating, then the full gate before the final commit. Inspect
`git diff --check`, the staged diff, and ignored files before every commit.

## Evidence and safety rules

- Never fabricate metrics, benchmark numbers, command output, test results, or deployment status.
- Never say a command passed unless its successful exit status was observed in this workspace.
- Never commit or print real secrets, credentials, private keys, unredacted ticket text, or `.env`.
- Do not download or commit the dataset unless the active stage explicitly calls for ingestion.
- Do not call paid APIs, hosted model inference, or paid cloud infrastructure.
- Preserve correct user changes and keep unrelated work out of stage commits.
- Do not push, rewrite history, force-push, or move model aliases without explicit user authorization.

## Testing rules

- Every behavior change requires tests in the same stage/commit series.
- Unit-test pure logic; use integration tests for PostgreSQL, MLflow, and service boundaries; reserve
  end-to-end tests for the local Compose stack.
- Tests use small synthetic, non-sensitive fixtures and must not require the full dataset or network.
- Add regression tests before fixing a defect when practical.
- Keep unfinished functionality explicitly marked in documentation; placeholders must not imply a
  working implementation.
