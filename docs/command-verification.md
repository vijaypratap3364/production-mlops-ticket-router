# README command verification

## 2026-08-13 native-workflow and release-gate follow-up

- No Docker Desktop startup or local `docker`/`docker compose` command was performed.
- Repository inspection found Compose execution only in explicit `docker-*` Make targets and the
  opt-in `--run-compose-disruptions` operational-validation branch. There is no `.vscode`, `.agents`,
  `.codex`, PowerShell, batch, or startup hook that launches Docker.
- `uv sync --locked --all-groups`, Ruff format/check, mypy across `src tests scripts load_tests`, and
  pre-commit completed successfully.
- `uv run pytest` completed with 232 passed, 2 intentionally skipped, and 80.38% branch coverage.
- `uv build` created the source distribution and wheel. `uv build --locked` is unsupported by the
  installed uv version and is not the documented build command.
- GitHub Actions run `31761752043` succeeded for commit
  `ac6062b6aee087bc90e8ee42118979d5004a2ad9`, including remote package, quality, PostgreSQL
  integration, and four container-build jobs.
- The localhost MLflow health probe was unavailable. Native `mlruns/mlflow.db` contained a different
  version-1 run under `candidate` only, so release attestation and `v1.0.0` creation remained blocked
  rather than substituting mismatched lineage.

This record distinguishes commands executed during the Stage 16 documentation pass from commands
validated through non-mutating help/contracts or prior stage evidence. It prevents a reviewer guide
from implying that every expensive or destructive lifecycle command was rerun.

Verification occurred on 2026-08-12/13 in the existing Windows workspace, the closest available
environment to a clean checkout. The worktree began clean at Stage 15 commit `ddb2a58`; ignored data,
MLflow, PostgreSQL, and model artifacts from earlier verified stages were present.

## Executed successfully in Stage 16

- `git ls-remote --exit-code ... HEAD` resolved the documented public repository at
  `ddb2a585a78f3034dbba63874ea64229abc39259`.
- `uv python install 3.12` reported that Python 3.12 was already installed.
- `uv sync --locked --all-groups` resolved 242 and checked 240 packages.
- Package import, documentation contract, CI contract, and import tests passed in an 18-test focused
  run before the full gate.
- `docker compose config --quiet` succeeded.
- `docker compose up -d postgres mlflow migrate api dashboard` completed with PostgreSQL and MLflow
  health gates, successful migration, healthy API, and dashboard startup.
- `/health`, `/ready`, and `/model` returned successful contracts; `/ready` reported model and
  database ready, and `/model` returned `ticket-router` champion version 1.
- One synthetic prediction returned `Billing and Payments` with champion version 1.
- Streamlit, MLflow, and Prefect health endpoints responded successfully.
- `docker compose --profile smoke run --rm smoke-test` passed prediction, feedback, PostgreSQL
  persistence, API readiness, and dashboard connectivity.
- `uv run python -m ticket_router.orchestration.fixture_flow` completed its six-row, five-queue
  network-free Prefect fixture flow.
- Pinned download cache reuse reported 61,764 raw rows at revision
  `ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb`.
- Normalization reproduced 28,190 English rows and normalized-data SHA-256
  `abac81257c3e21b462217ba02455b11e4bc7427b13e0adc5c0f6a3cc946751e7`.
- Analysis reproduced ten selected queues, imbalance ratio 20.388471, 4,495 exact duplicate groups,
  8,990 flagged duplicate rows, and no contradictory groups.

All documented Python entry points were also invoked with `--help` and exited successfully:
download, normalize, analyze, prepare, baseline/candidate training, final evaluation, promotion,
monitoring reference/run/simulation, orchestration, champion contract, load test, operational
validation, benchmark, Alembic, Uvicorn, and Streamlit.

## Broken or not rerun

- `uv run python -m ticket_router.data.prepare` exited nonzero in the existing historical workspace.
  The current analysis artifact's serialized hash differs from the hash recorded by the Stage 4
  split manifest, although normalized-data hash, configuration hash, row counts, classes, and label
  mapping are unchanged. The command correctly refused to overwrite the sealed prepared artifacts.
  `--force` was deliberately not used because rewriting train/validation/test would invalidate the
  reviewed model lineage. A clean-checkout prepare path could not be proven in this pass without
  repeating the full lifecycle; this is an unresolved reproducibility caveat, not hidden as a pass.
- Full baseline/candidate training was not repeated. It is CPU/time intensive, and its verified
  generated Stage 5/6 artifacts and reviewed documentation already exist.
- Final evaluation was not repeated because the persistent audit correctly prohibits another test
  evaluation. Promotion was not repeated because champion version 1 is already assigned and alias
  movement must remain explicit.
- Monitoring reference, current-window monitoring, drift simulation, and Stage 15 benchmarks were
  not regenerated in this documentation pass. Their CLI contracts passed, and their actual earlier
  results remain in reviewed docs/generated artifacts.
- `docker compose build ...` and the bootstrap profile were not repeated. Existing Stage 13 images
  started successfully and the complete smoke test passed; bootstrap would unnecessarily redownload
  or retraverse the expensive model lifecycle.
- Standalone Uvicorn and Streamlit host commands were not launched because the verified Compose
  services already occupied ports 8000 and 8501. Their CLI entry points passed and the containerized
  services were healthy.
- The Bash `cp .env.example .env` step was not run on Windows because an ignored `.env` already
  existed; the tracked template exists and its ignore rule was checked.

## Final Stage 16 quality gate

- Ruff format and format-check: 211 files clean.
- Ruff lint: passed.
- Strict mypy: no issues across 200 source/test/script files.
- Pytest: 221 passed, 2 skipped, 26 third-party deprecation warnings.
- Branch coverage: 80.54%, above the required 80% threshold.
- Pre-commit: lock check, Ruff lint, and Ruff format hooks passed.
- `uv build`: produced the source distribution and platform-independent wheel successfully.

The two skips are the opt-in live Compose e2e test and the PostgreSQL-only migration variant that is
disabled without its explicit test environment. The Compose smoke path was executed separately and
passed as recorded above.
