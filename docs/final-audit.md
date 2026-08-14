# Final production-readiness audit

## Release-remediation follow-up

Follow-up date: 2026-08-13 (America/Chicago)

The audit below is retained as the immutable evidence snapshot that blocked the first release
attempt. Its four failed items have since been remediated in the release-candidate changes:

- CI no longer combines `UV_FROZEN=true` with `uv sync --locked`, and a workflow contract test
  prevents that invalid combination from returning.
- The locked environment now contains `cryptography 50.0.0`, `GitPython 3.1.59`, `MLflow 3.15.1`,
  and `PyArrow 24.0.0`. `pip-audit` reports no known vulnerability in the installed environment.
- Real `ticket_router.data.prepare` reruns now accept serialization-only selected-class report drift
  after verifying the normalized-data hash, configuration hash, label mapping, selected counts, and
  every sealed output hash. The audited split-manifest hash remained byte-for-byte unchanged.
- API and worker images receive an explicit commit/dirty identity, and a new release-attestation
  gate ties a clean release commit to the immutable champion while preserving the honest historical
  `git_commit=unavailable` value. The gate does not reevaluate test data or move an MLflow alias.

Current native release-candidate evidence is: 232 tests passed and 2 intentionally skipped with
80.38% branch coverage; Ruff and mypy passed; pre-commit passed; the Python source distribution and
wheel built; and the installed dependency scan returned `No known vulnerabilities found`. The
earlier Docker, migration, champion-backed smoke, registry, and monitoring evidence below remains
historical evidence and was not rerun after the laptop switched to the lightweight native workflow.

GitHub Actions run
[`31761752043`](https://github.com/vijaypratap3364/production-mlops-ticket-router/actions/runs/31761752043)
succeeded for native-workflow remediation commit
`ac6062b6aee087bc90e8ee42118979d5004a2ad9`. Package build, Ruff, mypy, unit/coverage,
PostgreSQL integration, and the four GitHub-hosted container-build jobs all passed. Any later release
commit must pass its own exact-commit CI run before tagging.

All four original **Failed** findings are resolved. The tag is still withheld because the required
champion release attestation has not executed: `http://127.0.0.1:5000/health` was unavailable during
the native-only follow-up. The local SQLite MLflow store is not equivalent; it contains a different
run with only the `candidate` alias. The audit does not create or move an alias to fabricate release
evidence, and it does not start Docker without explicit user authorization.

Audit date: 2026-08-12 (America/Chicago), with commands completing on 2026-08-13 UTC  
Audited commit: `10bffa30063268a52f10f7f0204d298203986e97` on `main`  
Repository: `vijaypratap3364/production-mlops-ticket-router`

## Readiness summary at the original audited commit

At audited commit `10bffa30063268a52f10f7f0204d298203986e97`, the repository was a strong,
working local portfolio system but was not ready for a `v1.0.0` tag. The champion-backed Compose
stack, API, PostgreSQL persistence, MLflow registry, Prefect, Streamlit, monitoring, tests, and
documented benchmark worked locally. The four release-blocking gaps identified then are retained
below as historical evidence and are closed by the release-remediation follow-up above.

No model was retrained and no MLflow alias was moved during this audit. Existing PostgreSQL and
MLflow named volumes were preserved. The Compose services were shut down cleanly at the end.

## Passed

### Repository and static quality

- The worktree was clean and synchronized with `origin/main` at the start:
  `## main...origin/main`.
- Python is constrained to `>=3.12,<3.13`; the audit ran on CPython 3.12.13.
- `uv sync --locked --all-groups` exited 0: 242 packages resolved and 240 installed packages
  checked.
- `uv run ruff format .`, `uv run ruff format --check .`, and `uv run ruff check .` exited 0.
  Ruff reported 211 files unchanged/already formatted and no lint findings.
- `uv run mypy src tests scripts load_tests` exited 0 with: `Success: no issues found in 200
  source files`.
- `uv run pre-commit run --all-files` exited 0; lock, Ruff lint, and Ruff format hooks passed.
- `uv build` exited 0 and created both the source distribution and wheel.
- `git diff --check` had no tracked change to inspect before this report was added.

### Tests and coverage

`uv run pytest` exited 0 with the following actual result:

- 223 tests collected.
- 221 passed.
- 2 skipped: the opt-in Compose e2e test and the PostgreSQL-dependent migration test when its test
  database variable is absent.
- 26 third-party deprecation warnings.
- Branch-aware coverage: **80.54%**, above the configured 80% threshold.
- Runtime: 84.01 seconds reported by pytest (91.2 seconds command wall time).

The tests cover the data, leakage, model, registry, API, database, monitoring, orchestration,
dashboard, benchmark, Compose-contract, and documentation layers. Critical behavior does not depend
on a notebook; no `.ipynb` file exists.

### Data controls and lineage

- The selected dataset contains 28,190 normalized English records and 10 configuration-selected
  queues.
- The sealed split contains 19,729 train, 4,232 validation, and 4,229 test records.
- An independent audit loaded all three Parquet files and found zero `ticket_record_id` overlap and
  zero `normalized_text_hash` overlap for train/validation, train/test, and validation/test.
- All 10 labels occur in every split.
- SHA-256 checks passed for train, validation, test, and training-reference Parquet files.
- The split manifest records seed 42, source/configuration hashes, output hashes, preprocessing,
  label mapping, and the contradictory-duplicate policy.
- `model_text` is the sole exposed model feature. Source review found one implementation of
  `preprocess_model_text`, reused by preparation, serving, monitoring/retraining preparation, and
  contract checks.
- Static search found no dashboard import from data, features, modeling, registry, database,
  monitoring, or orchestration modules.
- The test-access audit records one authorized final evaluation, `repeated_test_evaluation_allowed:
  false`, and `recovery_reloaded_test_data: false`.

### Model, experiment tracking, and registry

- MLflow was reachable locally and registered model `ticket-router` resolved both `candidate` and
  `champion` aliases to immutable version 1 and run `042a28f803c24c79b5c1051f2e72a880`.
- Registry/run metadata includes split, training, configuration, and test-data hashes; per-class and
  aggregate test metrics; model contract status; model signature status; latency; and model size.
- Promotion audit evidence shows explicit approval and all seven configured gates passed. Normal
  Prefect candidate/retraining flows set `champion_promotion_performed=False`; the explicit promotion
  CLI requires `--approve`.
- The actual one-time held-out test metrics are:

| Metric | Actual result |
| --- | ---: |
| Macro F1 | 0.6960570778151813 |
| Weighted F1 | 0.6884321005416468 |
| Accuracy | 0.6897611728541027 |
| Macro precision | 0.7855124437229934 |
| Macro recall | 0.6413172874560391 |
| Minimum per-class recall | 0.46825396825396826 |
| Log loss | 1.0421782107530768 |
| Model size | 10,030,367 bytes (9.5657 MiB) |

### API, database, and privacy

- Before shutdown, `/health`, `/ready`, `/model`, MLflow `/health`, Prefect `/api/health`, and
  Streamlit `/_stcore/health` all returned HTTP 200. `/ready` reported model and database ready, and
  `/model` reported `ticket-router` version 1 at alias `champion`.
- The host smoke test and the later containerized smoke test both passed health, readiness,
  prediction, feedback, PostgreSQL persistence, and dashboard connectivity. The final containerized
  smoke request used model version 1.
- Alembic reported `20260807_0002 (head)` before and after `alembic upgrade head`.
- A persisted smoke prediction had a 64-character one-way text hash, `redacted_text IS NULL`, and
  only approved metadata key `client_name`. Its feedback row carried the same model version.
- API schemas enforce 2,000 subject characters, 20,000 body characters, usable text, label validity,
  and a configurable maximum batch size of 100 by default.
- Structured errors hide stack traces. Request logs contain method, route, status, duration, and
  request ID, not subject/body. Static search found no normal logger call containing raw ticket
  fields.
- SQLAlchemy statements are constructed with typed expressions; the few textual SQL statements are
  fixed health/migration/smoke queries and bind request IDs as parameters.
- No CORS middleware is configured, so cross-origin browser access is not opened by default.
- Dashboard HTTP calls and local probes use finite timeouts.

### Security and public-repository hygiene

- `.env`, datasets, MLflow artifacts, databases, model files, reports, and Docker volume state are
  ignored. Tracked data/artifact content is limited to placeholders and the reviewed resume metrics
  JSON.
- A high-risk secret-pattern scan of current tracked files returned no match; a Git-history patch
  scan found zero matches for the same AWS, GitHub, hosted-model-token, and private-key patterns.
- Compose requires `POSTGRES_PASSWORD` and `INPUT_HMAC_SECRET`; it does not silently supply them.
  `.env.example` contains explicit local-only placeholders.
- PostgreSQL, MLflow, API, Streamlit, and Prefect ports bind to `127.0.0.1`.
- All four project images declare non-root runtime user `ticket-router`; runtime inspection confirmed
  it for API, dashboard, MLflow, Prefect server, and worker containers.
- No paid API or hosted cloud dependency is configured.

### Monitoring and orchestration

- `uv run python -m ticket_router.orchestration.fixture_flow` exited 0 with six fixture rows, five
  queues, and fixture hash `a97d4abdfeeeb015e31336e4ee631138100053537195b818f1858dea888309d6`.
- Champion monitoring-reference generation exited 0 and produced 23,961 privacy-safe rows tied to
  champion version 1, split-manifest hash
  `42a89d794b3a175dd11f9fac3c4f537b6d5e993fdbd6dc0e48466ae91845b28c`, and reference hash
  `b17bb2f5f006fcd5d07f7ae7673b9ab42a2fab4b8be6fb19c1cb35b2d6ebec72`.
- The deterministic planted-drift simulation exited 0 and returned `critical`: 9/9 input features
  drifted, predicted-label Jensen-Shannon score was 0.35715393566914383, and low-confidence rate
  increased by 0.932. Three independent critical signals fired.
- The real seven-day monitoring command completed and persisted run
  `76c0e44c-56c3-4d2f-bd8a-a574c3994bf2`. Its reference hash matched the new champion reference.
- Monitoring clearly separated unlabeled drift from delayed-label quality.

### Docker and operational validation

- All four images were rebuilt during the audit. The aggregate build command exceeded its
  five-minute caller timeout, but subsequent image inspection showed new API, dashboard, MLflow, and
  worker image creation timestamps between 02:55 and 02:57 UTC and non-root user `ticket-router`.
- `docker compose --profile orchestration up -d --force-recreate ...` exited 0.
- Final pre-shutdown state: PostgreSQL, MLflow, API, dashboard, and Prefect server were healthy;
  Prefect worker was running; migration and Prefect-init jobs exited 0.
- `docker compose --profile smoke run --rm smoke-test` exited 0.
- Services were shut down with `docker compose --profile orchestration down --remove-orphans`.
  `docker compose ... ps -a` was empty afterward. PostgreSQL and MLflow named volumes remain present.

### Existing benchmark evidence

The Stage 15 benchmark was not rerun because it already contains immutable champion and environment
lineage and the audit did not change code or model aliases. Its actual local-Windows measurements are:

| Measurement | Actual result |
| --- | ---: |
| Champion load | 3,698.7324 ms |
| Direct single-request p95 | 11.411565 ms |
| Direct throughput | 140.9434 requests/s |
| API single-request p95 | 21.308660 ms |
| API p50 overhead | 9.634550 ms |
| API benchmark errors | 0 / 40 |
| Load-test p95 | 82 ms |
| Load-test throughput | 4.483186 requests/s |
| Load-test failures | 0 / 131 |
| Reliability scenarios | 9 / 9 passed |

These values describe one local Windows machine, not a production SLO or public deployment.

## Failed

There are no unresolved code, dependency, reproducibility, lineage-design, or CI findings from the
original **Failed** section. The subsections below describe the immutable audit snapshot; each was
resolved by the release-remediation changes summarized at the top of this report. Release execution
itself remains pending under **Not verified**.

### Current GitHub Actions run is red

Read-only GitHub inspection found the CI run for the audited commit failed:

- Run: <https://github.com/vijaypratap3364/production-mlops-ticket-router/actions/runs/31660859284>
- Commit: `10bffa30063268a52f10f7f0204d298203986e97`
- Package build and all four Docker build jobs passed.
- Ruff, mypy, unit-test, and PostgreSQL integration jobs failed before their checks because the
  workflow sets `UV_FROZEN=true` and also invokes `uv sync --locked`. CI reported: `the argument
  --locked cannot be used with UV_FROZEN`.

This is a release blocker even though equivalent local checks passed.

### Known dependency vulnerabilities

`uv run --with pip-audit pip-audit --path .venv/Lib/site-packages --skip-editable` exited 1 and
reported six known vulnerabilities in two installed transitive packages:

- `cryptography 49.0.0`: `PYSEC-2026-3552`; fixed in 50.0.0.
- `GitPython 3.1.57`: five GHSA advisories; fixed in 3.1.58.

The audit did not alter or re-lock dependencies. These findings must be remediated and rescanned
before a production/release claim.

### Prepared-data regeneration is not idempotent against sealed lineage

`uv run python -m ticket_router.data.prepare` exited 1 with the intentional safety error that
existing artifacts were generated from different data or configuration. Direct hash comparison
isolated the mismatch:

- normalization manifest: matches.
- normalized Parquet: matches.
- `selected_classes.json`: current `5bd92c417238ac73ed00880a409a40289a260a963dcacb45b260645e549e89bb`;
  sealed split-manifest value
  `4ff222971f5c14aabf7c25c8f986ddcd8d616d853e6e92408f7fcf379ddcf60d`.

The safety behavior is correct, and all existing processed-file hashes remain valid, but the documented
regeneration path is not reproducible in the current workspace without force. Force was deliberately
not used because it could invalidate sealed model/test lineage.

### Champion Git lineage is incomplete

The registered version and MLflow run contain data/configuration hashes, but their `git_commit` tag is
`unavailable`. The Stage 7 test-access audit and promotion audit also contain null Git commit/dirty
fields because evaluation/promotion ran in a container without Git metadata. This does not invalidate
the measured model, but it violates the intended complete model-to-code lineage contract.

## Warning

- The real seven-day monitoring run returned `critical` for 468 locally accumulated events: 7/9
  input features drifted, prediction-distribution score was 0.5252590267107526, low-confidence rate
  increased by 0.29339262840577474, and five classes were absent. These events include synthetic
  smoke/load traffic and are not representative production traffic, but a deployed operator should
  still treat the status as requiring review.
- Delayed-label metrics were unavailable because only 23 feedback records existed, below the
  configured minimum of 50. Acceptance rate was 1.0, but no F1 was fabricated.
- NumPy emitted constant-column divide warnings while Evidently generated the real drift report. The
  run completed, but warning-free handling of constant monitoring features would improve operations.
- Prefect worker was running but has no container healthcheck. The server has a healthcheck and the
  worker process is restart-managed, but worker liveness is less observable than other services.
- MLflow used approximately 1.152 GiB in the final snapshot, high for a student laptop with a
  3.249-GiB Docker memory limit. Other observed container memory was API 249.8 MiB, Prefect worker
  152.3 MiB, Prefect server 123.9 MiB, PostgreSQL 75.56 MiB, and dashboard 54.34 MiB.
- MLflow, Prefect, `/metrics`, and the API have no authentication layer. Loopback-only bindings make
  this acceptable for the documented local demo, but the stack must not be exposed to an untrusted
  network as-is.
- `uv build --locked` is not a valid command for the installed uv version and exited 2; plain
  `uv build` is the verified command and is what CI uses.
- Coverage meets the global gate, but dashboard pages, orchestration flow bodies, download CLI paths,
  and some benchmark/MLflow command paths have materially lower module-level coverage.
- The test suite reports third-party deprecations from Starlette/TestClient, Joblib/NumPy, Evidently,
  and Litestar. They do not currently fail tests but should be tracked during upgrades.

## Not verified

- A destructive clean-checkout rehearsal was not performed. The sandbox denied deletion of the
  existing `.venv`, so environment recreation from an absent virtual environment is not claimed.
- Full ingestion, analysis, split generation, candidate search, final training, and first-champion
  bootstrap were not rerun from empty data/MLflow/PostgreSQL volumes. Repeating final test evaluation
  would also violate the sealed-test policy.
- The full conditional-retraining path was not executed against newly approved delayed labels. Unit
  tests verify new dataset versioning, candidate registration, and no automatic champion promotion;
  the live database had zero retraining-run records and insufficient labels.
- GitHub Actions remediation is now verified by successful run `31761752043` for commit
  `ac6062b6aee087bc90e8ee42118979d5004a2ad9`. Any subsequent release commit still requires its own
  green run.
- The release workflow has not been executed for a tag.
- The champion release-attestation command has not succeeded. The required audited registry was
  unavailable at `http://127.0.0.1:5000`; the native local MLflow store has different lineage and was
  deliberately not treated as equivalent.
- No public deployment, TLS termination, authentication, multi-host load, disaster recovery,
  backup/restore, or long-duration soak test was performed. The system remains a local, zero-cost
  portfolio deployment.

## Recommended future improvements

1. Add a Prefect-worker healthcheck and investigate the MLflow memory footprint under the local
   Docker limit.
2. Keep synthetic smoke/load events separate from representative monitoring windows; collect at least
   50 reviewed feedback labels before interpreting delayed-label quality.
3. Add authentication, TLS, authorization, and a reverse proxy only if the stack is ever exposed
   beyond loopback.
4. Perform and document one disposable clean-machine bootstrap rehearsal, including ingestion,
   migrations, candidate registration, explicit initial promotion, smoke, monitoring reference, and
   teardown.
5. Raise targeted coverage in dashboard rendering, orchestration branches, and external-service error
   paths without adding assertion-free tests.
6. Update pinned GitHub Actions before their Node.js 20 compatibility period ends; the current green
   run emitted advance deprecation warnings while GitHub forced those actions onto Node.js 24.

Suggested release tag after the remaining runtime gate passes: **`v1.0.0`**. Do not create it until
the champion release attestation succeeds and CI is green for the exact commit being tagged.
