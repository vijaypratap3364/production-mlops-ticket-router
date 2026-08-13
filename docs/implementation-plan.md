# Production MLOps Ticket-Routing System: Implementation Plan

Status: historical pre-implementation baseline; Stages 1–16 are implemented
Target repository name: `production-mlops-ticket-router`
Target runtime: Python 3.12, local CPU, zero-cost open-source services
Prepared: 2026-08-03

Current operational contracts live in the README, cards, runbooks, ADRs, and generated manifests.
When they are more specific than a planned statement below, use the implemented documentation.

## 1. Scope, principles, and current repository state

The system will classify an English customer-support ticket into one of ten support queues using only the ticket's `subject` and `body`. The label is `queue`. The training, serving, monitoring, and retraining paths will run locally and will not call hosted inference APIs or provision cloud resources.

Repository inspection found an empty, non-Git workspace containing only Codex-generated `work/` and `outputs/` directories. No source, configuration, tests, repository instructions, Git history, or remote were present. Git was therefore initialized with `main` as the default branch. The generated workspace directories are excluded from Git and are not part of the proposed application.

Non-negotiable engineering rules:

- Never report a metric unless it was produced by a recorded run over a named data split.
- Never report a command or check as passing unless its exit status was observed.
- Treat the final test set as sealed after split creation. Model choice, features, hyperparameters, thresholds, and class-selection logic must be frozen before one authorized final evaluation.
- Use only `subject` and `body` as model inputs. `queue` is the target. Post-submission or agent-assigned fields such as `answer`, `response`, `tags`, `priority`, `type`, `queue`, and business metadata must never enter a fitted feature pipeline.
- Use deterministic seeds where supported and record unavoidable sources of nondeterminism.
- Pin source-data revisions and dependency versions, and record hashes for source manifests, validated data, split manifests, configuration, code commit, feature pipeline, and model artifact.
- Do not persist raw ticket text in production prediction logs by default.
- Keep domain, data, model, persistence, API, orchestration, and UI concerns separate.
- Establish sparse, interpretable baselines before considering any more complex model. No external AI API or paid service is permitted.
- Make `uv run ...` commands canonical and cross-platform. Makefile targets and PowerShell helpers will only wrap those commands.

## 2. Dataset facts to verify and pin

The requested source is [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets). On 2026-08-03, its dataset card advertised one viewer split with 61,765 rows, English and German content, CSV files, and a CC BY-NC 4.0 license. The repository contained multiple CSV variants, so implementation must not assume that an unpinned default loader selects the intended file. These are upstream observations, not locally reproduced data-profile results.

The first data task will use `huggingface_hub` to resolve and record the full immutable Hub commit SHA, enumerate repository files, inspect schemas and row counts, and select the intended current multi-language source through configuration. The source selection, filename/config, full revision, ETag or file hash, library version, retrieval timestamp, and license metadata will be written to a manifest. Ingestion must fail when the revision is `main`, missing, or still set to the placeholder in `.env.example`.

No class names, class counts, English row count, missing-value rates, duplicate rates, or model metrics are asserted in this plan. Those belong to generated, hashed reports after ingestion.

License note: CC BY-NC 4.0 is appropriate only for uses compatible with its non-commercial restriction and attribution requirements. The public repository will not redistribute the dataset. It will include attribution, the source link, DOI, license notice, and reproducible download commands. Whether the repository's own code uses MIT or Apache-2.0 requires confirmation and does not relicense the dataset.

## 3. Proposed repository structure

The tree below is the intended end state. Generated directories contain only `.gitkeep` placeholders; their contents remain ignored.

```text
production-mlops-ticket-router/
├── .github/
│   └── workflows/
│       └── ci.yml                     # Optional GitHub CI mirroring local checks
├── alembic/
│   ├── versions/                      # Reviewed database migration revisions
│   ├── env.py                         # SQLAlchemy/Alembic integration
│   └── script.py.mako
├── configs/
│   ├── base.yaml                      # Seed, paths, shared names, hash policy
│   ├── data.yaml                      # Hub revision/file, language, class and split rules
│   ├── experiments.yaml               # Baselines, search spaces, selection metric
│   ├── monitoring.yaml                # Windows, minimum samples, drift/quality rules
│   └── retraining.yaml                # Trigger, gate, cooldown, and promotion policy
├── data/
│   ├── raw/.gitkeep                   # Immutable downloaded source; never committed
│   ├── interim/.gitkeep               # Validated/filtering intermediates
│   ├── processed/.gitkeep             # Model-ready split files and manifests
│   ├── reference/.gitkeep             # Privacy-safe monitoring reference profiles
│   └── retraining/inbox/.gitkeep      # Approved labeled snapshots from secure source
├── docker/
│   ├── api.Dockerfile                 # Minimal FastAPI runtime image
│   ├── mlflow.Dockerfile              # Local MLflow server image
│   ├── prefect.Dockerfile             # Worker/training/monitoring image
│   └── streamlit.Dockerfile           # Demo UI image
├── docs/
│   ├── implementation-plan.md         # This plan
│   ├── architecture.md                # Runtime/data/model diagrams and ADR links
│   ├── data-card.md                   # Provenance, license, schema, limits, splits
│   ├── model-card.md                  # Champion behavior, metrics, limits, ethics
│   ├── benchmarking.md                # Reproducible benchmark protocol and results
│   ├── privacy-and-security.md         # Threat model, retention, redaction, secrets
│   ├── runbook.md                     # Local operation, backup, recovery, rollback
│   ├── monitoring-runbook.md          # Drift/quality triage and false-positive handling
│   └── decisions/                     # Numbered architecture decision records
├── reports/
│   └── .gitkeep                       # Generated EDA/evaluation/drift outputs ignored
├── scripts/
│   ├── bootstrap.ps1                  # Optional Windows wrapper around canonical uv commands
│   └── wait_for_services.py           # Bounded, cross-platform readiness helper
├── src/
│   └── ticket_router/
│       ├── __init__.py
│       ├── cli.py                     # Thin command entry points; delegates to services
│       ├── config.py                  # Typed settings/config loading and validation
│       ├── constants.py               # Stable names only; no environment secrets
│       ├── hashing.py                 # Canonical config/data/artifact digest utilities
│       ├── logging.py                 # Structured logging and content-redaction filter
│       ├── domain/
│       │   ├── entities.py            # Ticket, prediction, feedback domain types
│       │   ├── labels.py              # Label-space validation loaded from artifacts
│       │   └── errors.py              # Domain-specific exceptions
│       ├── data/
│       │   ├── download.py            # Revision-pinned Hub retrieval and source manifest
│       │   ├── schemas.py             # Raw, analysis-only, and model-input Pandera schemas
│       │   ├── validate.py            # Data contracts, quarantine, and validation report
│       │   ├── language.py            # Strict `language == en` filtering and audits
│       │   ├── deduplicate.py         # Exact/near-duplicate grouping before splitting
│       │   ├── class_selection.py     # Config-driven eligible-class ranking and manifest
│       │   ├── split.py               # Deterministic stratified group split and seal
│       │   ├── load.py                # Split-aware readers that enforce access policy
│       │   └── profile.py             # Train-only EDA and leakage audit inputs
│       ├── features/
│       │   ├── text.py                # Subject/body normalization and field markers
│       │   ├── sparse.py              # Word/character TF-IDF pipeline builders
│       │   └── metadata.py            # Pipeline/version/signature metadata, not predictors
│       ├── modeling/
│       │   ├── baselines.py           # Dummy, NB, logistic, LinearSVC/SGD candidates
│       │   ├── experiment.py          # Reproducible MLflow experiment runner
│       │   ├── search.py              # Bounded validation/CV model search
│       │   ├── metrics.py             # Metric definitions and confidence intervals
│       │   ├── evaluate.py            # Validation and separately guarded final evaluation
│       │   ├── selection.py           # Frozen candidate-selection rule
│       │   ├── registry.py            # MLflow versions, tags, aliases, lineage
│       │   ├── promotion.py            # Candidate/champion gates and audit record
│       │   └── serialization.py        # MLflow pyfunc/signature/dependency packaging
│       ├── services/
│       │   ├── prediction.py          # Framework-neutral prediction use case
│       │   ├── feedback.py            # Feedback validation and persistence use case
│       │   ├── model_loader.py         # Alias-aware load, cache, readiness, safe reload
│       │   └── health.py               # Dependency health checks
│       ├── persistence/
│       │   ├── base.py                # SQLAlchemy declarative base
│       │   ├── session.py             # Engine/session lifecycle
│       │   ├── models.py              # ORM mappings only
│       │   ├── repositories.py        # Prediction/feedback/report persistence adapters
│       │   └── uow.py                 # Transaction boundary
│       ├── api/
│       │   ├── app.py                 # FastAPI factory and lifespan
│       │   ├── dependencies.py        # Injected settings, predictor, repositories
│       │   ├── schemas.py             # Versioned Pydantic request/response contracts
│       │   ├── errors.py              # Stable HTTP error mapping
│       │   └── routes/
│       │       ├── health.py           # Liveness/readiness/version endpoints
│       │       ├── predictions.py      # Prediction HTTP adapter
│       │       └── feedback.py         # Delayed-label feedback HTTP adapter
│       ├── monitoring/
│       │   ├── reference.py            # Training-reference profile creation
│       │   ├── data_quality.py         # Null/empty/length/OOV/schema checks
│       │   ├── drift.py                # Evidently drift calculation over safe features
│       │   ├── model_quality.py        # Delayed-label quality windows
│       │   ├── policy.py               # Severity and trigger decisions
│       │   └── reporting.py            # Local HTML/JSON reports and DB summaries
│       ├── retraining/
│       │   ├── dataset.py              # Approved labeled snapshot assembly
│       │   ├── policy.py               # Eligibility, cooldown, and minimum-label rules
│       │   ├── gates.py                # Challenger-versus-champion acceptance gates
│       │   └── audit.py                # Retraining/promotion audit metadata
│       └── orchestration/
│           ├── tasks.py                # Small idempotent Prefect tasks
│           ├── ingest_flow.py          # Download, validate, select, split, profile
│           ├── train_flow.py           # Fit, validate, select, register candidate
│           ├── monitor_flow.py         # Prediction/data drift and delayed quality
│           └── retrain_flow.py         # Policy check, challenger train/evaluate/register
├── streamlit_app/
│   ├── app.py                         # UI composition; calls API only
│   ├── client.py                      # Typed HTTP client for FastAPI
│   └── pages/
│       ├── predict.py                 # Ticket entry and routing result
│       ├── feedback.py                # Corrected-label demonstration
│       └── monitoring.py              # Read-only local status and report links
├── tests/
│   ├── fixtures/                      # Tiny synthetic, non-sensitive test records
│   ├── unit/                          # Pure business/data/model unit tests
│   ├── contract/                      # Pandera, Pydantic, config, migration contracts
│   ├── integration/                   # PostgreSQL, MLflow, registry, API boundaries
│   ├── e2e/                           # Docker Compose smoke path
│   └── load/locustfile.py             # Local-only latency/concurrency scenarios
├── .dockerignore                      # Excludes data, artifacts, secrets, and Git metadata
├── .env.example                      # Safe local placeholders only
├── .gitignore                        # Excludes secrets and generated state
├── .pre-commit-config.yaml           # Ruff and lightweight repository checks
├── alembic.ini                       # Migration configuration without credentials
├── compose.yaml                      # All local services, networks, health checks, volumes
├── LICENSE                           # Code license only; dataset separately attributed
├── Makefile                          # Optional aliases for canonical uv commands
├── pyproject.toml                    # Package, tools, dependency groups, CLI entry point
├── README.md                         # Portfolio overview and zero-cost quick start
└── uv.lock                           # Committed reproducible dependency resolution
```

Module boundaries are intentional: API routes and Streamlit pages perform validation and presentation only; they do not preprocess text, call estimators directly, construct SQL, or implement promotion rules. Domain services depend on interfaces implemented by persistence/model adapters. Prefect flows orchestrate the same tested functions exposed by the CLI rather than duplicating business logic.

## 4. Configuration and reproducibility design

Configuration will be layered as immutable typed objects:

1. Versioned YAML contains non-secret experiment, data, monitoring, and policy settings.
2. Environment variables contain deployment-specific URLs, ports, credentials, and privacy switches.
3. CLI overrides are explicit, printed with secrets redacted, and included in the resolved-config hash.

Each run records:

- Git commit and dirty-worktree flag;
- Python/platform, package lock hash, and container image label when applicable;
- full Hugging Face revision and source-file digest;
- validated dataset manifest and row-level stable IDs;
- selected-class manifest and its ranking/tie-break details;
- split-manifest digest and duplicate-grouping version;
- resolved redacted configuration plus SHA-256 digest;
- random seeds for Python, NumPy, scikit-learn, split generation, and searches;
- feature-pipeline parameters and vocabulary/artifact digest;
- MLflow run ID, model version, signature, aliases, and evaluation artifact paths.

Canonical commands will be Python CLI commands such as `uv run ticket-router data ingest`, `uv run ticket-router train`, `uv run ticket-router evaluate-final`, and `uv run ticket-router monitor`. Their exact interfaces will be finalized during implementation. The README will show PowerShell syntax and the same platform-neutral `uv` commands. Make targets will be conveniences, never the sole documented path.

## 5. Data flow and leakage controls

```mermaid
flowchart LR
    HF["Hugging Face dataset at pinned revision"] --> RAW["Immutable raw snapshot + manifest"]
    RAW --> VAL["Pandera validation + quarantine report"]
    VAL --> EN["English-only filter"]
    EN --> DEF["Top-10 label-space definition"]
    DEF --> DUP["Duplicate grouping"]
    DUP --> SPLIT["Stratified group split + sealed manifests"]
    SPLIT --> TRAIN["Train split: EDA, fit, CV"]
    SPLIT --> VALID["Validation split: selection and gates"]
    SPLIT --> TEST["Sealed test: one final evaluation"]
    TRAIN --> MLF["MLflow runs and artifacts"]
    VALID --> MLF
    MLF --> CAND["Registered model alias: candidate"]
    CAND --> TEST
    TEST --> REVIEW["Promotion review"]
    REVIEW --> CHAMP["Registered model alias: champion"]
```

### 5.1 Ingestion and validation

- Download through the Hub client using an explicit full commit SHA and configured file/config. Local caching is allowed; cache location and checksums are recorded.
- Preserve raw bytes as immutable local input. Reruns with the same revision either verify the existing digest or fail; they do not silently replace it.
- Create a stable `ticket_id` from source identity plus canonical row index and a separate normalized-content digest for duplicate grouping. Digests used in manifests are not reused as public production-log identifiers.
- Validate required columns, types, allowed language representation, null policy, non-empty `subject`/`body` policy, target presence, unexpected schema changes, and row-count sanity with Pandera.
- Quarantine invalid records locally and report counts/reasons without printing their text. Any tolerance is configured and acceptance is explicit.
- Retain analysis-only columns only in a physically separate analysis frame or file with an allowlist. The model-input constructor accepts exactly `subject` and `body`, and tests fail if forbidden columns reach a fitted pipeline.

### 5.2 English filter and label-space selection

- Filter on the dataset's validated language field normalized to the exact English code discovered during inspection; record before/after counts and reject unknown language encodings rather than guessing.
- Define `top_k: 10`, `min_class_count`, minimum rows per split, split fractions, and tie-break rule in `configs/data.yaml`.
- Do not choose `min_class_count` until the initial profile proves feasible support. The eventual value and rationale are committed; queue names are generated into a manifest, not copied into source code.
- Eligibility requires both the configured total minimum and enough duplicate groups to populate every split. Rank eligible queues by descending count, with normalized queue name as the deterministic tie-breaker, and take ten. Fail if fewer than ten qualify.
- Label counts may be used once before sealing solely to define the target universe and support stratification. No text-label relationship, metric, or model result may influence class eligibility. The generated manifest records this limited pre-split operation.

### 5.3 Duplicate control and splitting

- Normalize only for duplicate detection (Unicode normalization, whitespace normalization, case folding) and group exact duplicate `subject + body` pairs before splitting.
- Produce a near-duplicate audit using a documented, deterministic method. If template leakage is material, configure a similarity threshold and group near-duplicates before finalizing the split; never revise that threshold after looking at model results.
- Use a deterministic stratified group procedure so a duplicate group cannot cross train, validation, or test. Target fractions are proposed as 70/15/15, subject to class-support inspection.
- Validate disjoint ticket IDs, disjoint content groups, all ten labels in each split, count/fraction tolerances, and deterministic regeneration from the same inputs.
- Write separate split manifests containing stable IDs, not duplicated raw text. Include hashes and a seal timestamp.
- The standard data loader refuses test access. Only the `evaluate-final` command can open the test manifest, requires a registered `candidate` version and frozen selection-config hash, checks that no final evaluation already exists for that candidate/data combination, and records an audit event.

### 5.4 EDA

The reproducible EDA job runs after split creation and uses the training split only. It produces local HTML/JSON and image artifacts for class balance, missing/empty text, subject/body lengths, token/character statistics, duplicate/template prevalence, vocabulary coverage, suspicious target tokens, and potential PII patterns. Dataset-definition counts and structural validation are reported separately so the EDA does not normalize test inspection. Notebook exploration, if added, consumes only prepared training data and is not the authoritative pipeline.

## 6. Text features and model experimentation

Preprocessing will be a fitted scikit-learn `Pipeline`/`ColumnTransformer` serialized with the estimator. It will:

- coerce validated missing subject/body values according to the committed policy;
- normalize Unicode and whitespace conservatively;
- preserve punctuation/case unless a measured validation experiment justifies changing them;
- add explicit field markers or separate vectorizer branches for subject and body;
- avoid target-derived preprocessing and avoid fitting vocabulary/IDF outside training folds.

Experiment order:

1. `DummyClassifier` with stratified and most-frequent strategies to establish honest lower bounds.
2. Word TF-IDF plus Multinomial Naive Bayes.
3. Word TF-IDF plus regularized multinomial logistic regression.
4. Word and/or character n-gram TF-IDF with LinearSVC, SGDClassifier, or logistic regression, constrained by CPU/memory measurements.
5. Probability calibration only if a product requirement needs comparable confidence and validation proves the benefit; calibration folds remain inside training data.

No transformer or deep-learning experiment is planned. Adding one would require a separate documented decision proving that sparse baselines are insufficient and that the local CPU/storage cost remains reasonable.

The proposed primary selection metric is validation macro F1 because queue balance may be uneven and every selected queue matters. Secondary metrics are weighted F1, accuracy, per-class precision/recall/F1 and support, confusion matrix, top-k accuracy where scores exist, latency, artifact size, and peak memory. Log loss, Brier score, and calibration error are reported only for models with valid probabilities. Confidence intervals use a seeded bootstrap where statistically appropriate. No acceptance threshold will be invented before results exist.

Searches are bounded and declared in configuration. Cross-validation is stratified and group-aware inside the training split. The validation split selects one frozen candidate after CV/model development. Test performance is never used to choose a model, feature, threshold, or retry.

## 7. MLflow experiment tracking and model lifecycle

MLflow runs locally with PostgreSQL as its backend store and a Docker volume as its artifact store. It is not exposed publicly.

Every experiment logs parameters, hashes, dataset/split lineage, environment, metrics, per-class tables, figures, fitted pipeline, inferred signature, input example containing synthetic/non-sensitive text, dependency snapshot, and evaluation timing. Parent runs represent an experiment; child runs represent folds/trials.

Lifecycle:

1. Train and compare only on training/CV and validation data.
2. Select one frozen run by the committed selection rule and register a new immutable model version.
3. Validate signature, dependencies, artifact digest, lineage tags, smoke inference, and validation gates.
4. Assign the mutable `candidate` alias to that version and record who/what moved it.
5. Run the guarded final-test evaluation once for that version and data seal.
6. Review the final report. If all predefined gates are met, explicitly promote the same version to `champion`; otherwise leave the existing champion unchanged and document the result. The test cannot be used to return to experimentation on the same split.
7. Serving resolves `champion` at startup, logs its numeric version, and keeps serving the last healthy in-memory model if an alias refresh fails.

MLflow's old stage concept will not be used. Aliases (`candidate`, `champion`) plus tags (`status`, `data_hash`, `config_hash`, `git_sha`, `approved_by`, `approval_time`) provide the lifecycle. Alias changes go through one promotion service using optimistic checks/advisory locking so concurrent retraining runs cannot race.

If there is no incumbent champion, initial promotion requires all absolute gates defined before final evaluation. Future promotion compares the challenger with the champion on the same frozen validation/regression data and requires no unacceptable per-class regression, acceptable latency/size, complete lineage, and migration/API compatibility. Manual approval remains required by default.

## 8. Inference API design

FastAPI is an HTTP adapter over the prediction and feedback services. Proposed versioned endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health/live` | Process liveness only; no expensive dependency calls |
| `GET` | `/health/ready` | Champion loaded, database reachable, schema current |
| `GET` | `/version` | API version, model name/version/alias, preprocessing hash |
| `POST` | `/v1/predictions` | Validate an English ticket and return queue scores |
| `POST` | `/v1/feedback` | Attach a delayed corrected queue to a prediction |

Prediction request fields:

- `subject: str` and `body: str`, with explicit max lengths, at least one non-blank;
- optional caller-generated `request_id` for idempotency, otherwise server UUID;
- optional `language` constrained to `en` if accepted at all. The local sparse model is not a reliable language detector, so undeclared language remains a documented limitation.

Prediction response fields:

- `prediction_id`, `predicted_queue`, optional calibrated `confidence` and class-score map;
- `model_name`, numeric `model_version`, resolved alias, and response timestamp;
- `low_confidence` only if a threshold was selected on validation data and frozen in the model metadata;
- no echo of raw input by default.

Feedback request fields:

- `prediction_id`, `corrected_queue`, `source` from a small enum, and optional non-sensitive reason code;
- no free-form text by default. The corrected queue must belong to the model's recorded label space.

API behavior:

- Pydantic enforces bounds before preprocessing. Oversized/malformed input returns stable 4xx errors.
- Request IDs are unique and retries are idempotent. Database logging and response behavior are defined so a logging outage cannot silently lose events; proposed default is fail-closed with 503 for this portfolio demonstration, configurable only by an explicit ADR.
- Database operations are parameterized through SQLAlchemy, with bounded pools and transaction scopes.
- Model load occurs during lifespan startup. Readiness stays false until signature and smoke checks pass.
- Logs are structured and redact request bodies, authorization values, database URLs, secrets, and exception payloads that contain text.
- Localhost binding is the default for host-published service ports. Public authentication is out of scope; if non-local exposure is later requested, authentication, TLS, rate limiting, and a reverse proxy become required first.

## 9. PostgreSQL schema

Use UUID primary keys, `TIMESTAMPTZ` in UTC, explicit foreign keys, check constraints, indexed window fields, and Alembic migrations. MLflow and Prefect use separate databases/roles in the same local PostgreSQL container; application tables are isolated in the `ticket_router` database.

### `prediction_event`

- `id UUID` primary key; `request_id TEXT` unique; `created_at TIMESTAMPTZ` indexed.
- `predicted_queue TEXT`, `confidence DOUBLE PRECISION NULL`, `class_scores JSONB NULL`.
- `model_name TEXT`, `model_version INTEGER`, `model_alias TEXT`, `mlflow_run_id TEXT`, `preprocessor_hash TEXT`.
- `subject_length INTEGER`, `body_length INTEGER`, `combined_length INTEGER`, `empty_subject BOOLEAN`, and privacy-safe vocabulary/OOV statistics needed for monitoring.
- `input_hmac TEXT NULL`, produced with a rotating secret for local duplicate-rate analysis. Plain SHA hashes are not used because predictable text can be attacked offline.
- `latency_ms DOUBLE PRECISION`, `client_name TEXT NULL`, `status TEXT`, and `error_code TEXT NULL`.
- No raw `subject` or `body` columns in the default schema.

### `feedback_event`

- `id UUID` primary key; `prediction_id UUID` foreign key and indexed.
- `corrected_queue TEXT`, `source TEXT`, `reason_code TEXT NULL`, `created_at TIMESTAMPTZ`.
- `label_available_at TIMESTAMPTZ`, `is_approved_for_retraining BOOLEAN`, reviewer/audit fields.
- Append-only history; a view selects the latest approved correction rather than overwriting evidence.

### `monitoring_run`

- Run ID, type (`data_quality`, `prediction_drift`, `model_quality`), model version, reference/current window bounds, counts, status/severity, config hash, summary JSONB, local artifact URI, and timestamps.

### `retraining_run`

- Prefect flow/run IDs, trigger type and evidence, source snapshot/split/config hashes, candidate MLflow run/version, incumbent version, gate results JSONB, status, start/end timestamps, and failure summary without ticket content.

### `model_promotion_event`

- Immutable audit of model name/version, previous/new alias targets, decision, gate-report hash, actor (`manual` or flow identity), reason code, Git SHA, and timestamp.

Retention and deletion procedures will be documented. Even privacy-safe metadata can be sensitive in aggregate, so least-privilege roles separate migration, API, MLflow, Prefect, and read-only monitoring access.

## 10. Monitoring workflow

Evidently runs as a library inside a scheduled/local Prefect flow; no hosted Evidently service is used. Reports are saved to ignored local storage and summarized in PostgreSQL for the Streamlit view.

Reference data is created from the final training split and fitted pipeline without raw text. It contains distributions or aggregate sketches for input lengths, empty-field rates, vocabulary/OOV coverage, sparse-vector density, predicted class, maximum score/confidence, score margin, and safe operational fields. Where an individual row is not necessary, only aggregates are stored.

Monitoring flow:

1. Acquire a PostgreSQL advisory lock and establish an idempotent `(model_version, window_start, window_end, config_hash)` run key.
2. Read a bounded current prediction window after a minimum sample count is met.
3. Run schema/data-quality checks for missing fields, length changes, invalid values, empty rates, unseen/low-vocabulary input, and inference errors.
4. Compare prediction-class, confidence/margin, length, OOV, and other safe distributions with the version-matched training reference using Evidently/custom statistics.
5. Persist local JSON/HTML artifacts plus a compact DB summary; emit structured local warnings. No email/SaaS alerting is required.
6. Join predictions to latest approved feedback when labels arrive and compute the same primary/secondary quality metrics over configured rolling windows, including per-class support.
7. Mark results `insufficient_data` rather than drawing conclusions when sample or per-class label support is inadequate.
8. Evaluate trigger policy, but never auto-promote a model.

Drift is a diagnostic signal, not proof of quality loss. Thresholds and minimum counts will be set after reference variability is measured, committed in config, and evaluated over consecutive windows to reduce noise. Reports are version-specific so a deployment change does not masquerade as drift.

## 11. Delayed-label quality and the raw-text boundary

Feedback is sufficient to calculate delayed-label accuracy/F1 because it joins a corrected queue to a stored prediction. It is not, by itself, sufficient to retrain because raw ticket text is intentionally absent from production logs.

The default retraining design therefore consumes a separately supplied, approved labeled snapshot in `data/retraining/inbox/`, keyed to prediction IDs where possible. In a real system this would come from the access-controlled source ticket system. For the local portfolio, it can be generated from a reproducible demonstration fixture or an explicitly approved data refresh. It is validated, hashed, access-limited, and never committed.

An optional local-demo raw-content mode is possible, but must default to false, require a separate schema/migration and retention policy, and encrypt content at rest with a locally supplied secret. Enabling it should be a deliberate future decision, not an implicit consequence of feedback collection.

## 12. Controlled retraining workflow

Retraining can be requested manually or recommended by policy when one of these evidence types persists for the configured number of windows:

- delayed-label macro F1 or per-class recall degradation with adequate support;
- material input/prediction drift plus sufficient newly approved labels;
- scheduled dataset refresh at a new explicitly pinned Hub revision;
- operator-requested reproducibility check.

Workflow:

1. Acquire a global retraining lock; reject duplicate/cooldown-violating requests.
2. Snapshot and hash approved labeled input. Validate schema, English policy, label mapping, duplicates, and privacy authorization.
3. Preserve the original sealed test set. New data is partitioned according to a versioned temporal/group-aware policy into training and future evaluation windows; it is never appended blindly across splits.
4. Refit all feature steps using training data only. Run the bounded experiment suite using the same recorded selection protocol.
5. Compare the challenger with the numeric champion version on a frozen, representative regression/validation set and newly labeled evaluation window where valid. Report uncertainty and per-class support.
6. Apply predefined data, quality, regression, latency, artifact-size, lineage, API-signature, and smoke gates.
7. Register a passing challenger and move only the `candidate` alias.
8. Require an explicit promotion command/review to move `champion`. Record an immutable promotion event.
9. API workers perform a safe reload or restart, verify readiness, and retain the previous champion for rollback.
10. On failure, keep the champion unchanged, persist the reason, and avoid automatic retry loops.

The original final test set is for the original development claim, not an endlessly reusable promotion gate. After meaningful production feedback accumulates, define a new versioned evaluation protocol/snapshot rather than repeatedly tuning against the old test.

## 13. Prefect 3 orchestration

Prefect 3 is self-hosted in Docker with a local server and worker. Four flows wrap idempotent, directly testable task functions:

- `ingest_flow`: resolve pinned source, download, validate, select classes, group duplicates, split, seal, and create train-only profile.
- `train_flow`: verify manifests, run baselines/search, select by validation rule, package, register, gate, and alias `candidate`.
- `monitor_flow`: build version/window-specific data-quality, drift, and delayed-quality reports, then evaluate retraining recommendation policy.
- `retrain_flow`: validate approved snapshot, train challenger, compare/gate, register candidate, and stop before champion promotion.

Retries are allowed only for transient I/O/service failures, with bounded backoff. Deterministic validation or gate failures are non-retryable. Tasks use cache keys derived from input/config/code hashes; states and output paths are explicit. Schedules are optional and local. The CLI can run every flow synchronously without a Prefect server so local checks do not depend on orchestration availability.

## 14. Streamlit demonstration interface

Streamlit calls FastAPI over HTTP and never imports the fitted model or writes the database directly. It provides:

- a prediction page for subject/body input, routing result, version, and responsible confidence messaging;
- a feedback page constrained to the model label space and linked to the session prediction ID;
- a read-only monitoring page showing model/version, data sufficiency, recent drift/quality status, and links to local reports.

The UI warns that the dataset/model is a portfolio demonstration, predictions require human oversight, content may be sensitive, and raw text is not stored by default. It must not render stack traces, secrets, or raw database content.

## 15. Docker Compose services

`compose.yaml` will define:

| Service | Responsibility | Persistent local state |
|---|---|---|
| `postgres` | Application, MLflow, and Prefect databases with separate DBs/roles | Named `postgres_data` volume |
| `db-init` | Idempotently create service DBs/roles; one-shot | None |
| `migrate` | Apply reviewed application Alembic migrations; one-shot | None |
| `mlflow` | Tracking server, registry, artifact proxy | PostgreSQL + `mlflow_artifacts` volume |
| `api` | Champion inference and feedback API | No model baked into image; loads from MLflow |
| `streamlit` | Demo UI that calls API | None |
| `prefect-server` | Local orchestration API/UI | PostgreSQL |
| `prefect-worker` | Ingestion/training/monitoring/retraining execution | Mounted ignored data/report directories |

Evidently is a library in the worker, not a separately billed or hosted service. Compose health checks and `depends_on` readiness conditions are used, but application-level retry/backoff still handles startup races. Services run as non-root where practical, use pinned image/dependency versions, mount only needed paths, avoid embedding `.env` in images, and bind published ports to `127.0.0.1` by default. PostgreSQL and MLflow are not Internet-facing.

Optional Compose profiles can separate `core` (PostgreSQL, MLflow, API, Streamlit) from `orchestration` (Prefect) to reduce student-laptop resource usage. Exact CPU/memory limits will be benchmarked locally rather than fabricated.

## 16. Testing strategy

### Unit tests

- Typed config parsing, secret redaction, canonical hashing, and deterministic seeds.
- Pandera validation, English filtering, allowlisted model inputs, class ranking/ties, minimum support, and failure paths.
- Duplicate grouping, split determinism/disjointness/stratification tolerance, and sealed-test access denial.
- Text normalization and fitted pipeline behavior, including no fit on validation/test.
- Metric calculations, selection rules, promotion gates, monitoring sufficiency, retraining cooldown/idempotency.
- Domain prediction/feedback services with fake adapters.

### Contract and leakage tests

- Raw versus analysis-only versus model-input schemas are distinct.
- Forbidden columns cannot enter `fit`, `transform`, MLflow input examples, API schemas, logs, or saved feature names.
- Vocabulary/IDF and calibration are fitted only within training/folds.
- Exact duplicate groups never cross splits; split manifests are pairwise disjoint and hash-stable.
- Test loader is inaccessible to training/search/EDA modules and final evaluation is single-use per candidate/data seal.
- Model artifact signature and label map match the API response contract.

### Integration tests

- Alembic upgrade from empty database and downgrade policy where supported.
- SQLAlchemy repositories, idempotent request/feedback behavior, UTC timestamps, constraints, and concurrent writes against ephemeral PostgreSQL.
- FastAPI TestClient and real DB dependency wiring with a tiny synthetic model fixture.
- Local MLflow run logging, model registration, alias changes, artifact loading, and failed-promotion rollback.
- Prefect task/flow execution in ephemeral mode with external calls mocked or tiny local fixtures.
- Evidently report generation and version/window persistence from non-sensitive fixtures.

### End-to-end and non-functional tests

- Compose config validation, image builds, health/readiness, migration, model registration, API prediction, feedback, monitoring summary, and clean shutdown using tiny generated fixtures.
- Reproducibility test: two same-seed runs yield identical manifests/predictions where supported; any model-level tolerance is explicit.
- Security tests for oversized payloads, invalid labels, SQL-like content, log redaction, missing secrets, and non-development insecure defaults.
- Locust scenarios measure local throughput, p50/p95/p99 latency, error rate, and concurrency without claiming target values until measured. Reports remain local/ignored.

Tests never require the full dataset, network, paid service, or committed trained artifact. Tiny synthetic English ticket fixtures are clearly labeled as test data. Full-data benchmark tests are opt-in and produce hashed local reports.

## 17. Local quality checks and CI

Canonical local checks, later exposed as Make aliases, will be:

```text
uv sync --locked --all-groups
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest -m "not e2e and not load"
uv run pre-commit run --all-files
docker compose config --quiet
docker compose build
```

These commands are proposed interfaces, not claims that they currently exist or pass.

GitHub Actions remains optional and zero-cost within the account's available public-repository allowance. CI will:

1. Check out without credentials persisted.
2. Install Python 3.12 and `uv`; verify lockfile synchronization.
3. Cache only dependency downloads, never data/models/secrets.
4. Run Ruff formatting/lint, mypy, unit/contract tests, and coverage reporting with a justified threshold added after tests exist.
5. Start a PostgreSQL service for integration tests; use synthetic fixtures and an ephemeral local MLflow artifact directory.
6. Validate migrations, `docker compose config`, and build images with no push.
7. Optionally run a Compose smoke test on `main` if runtime remains reasonable.
8. Upload only non-sensitive test/coverage artifacts with short retention.

Full dataset ingestion, training, final-test evaluation, promotion, scheduled Prefect work, and load tests do not run in CI. No workflow will push images, deploy infrastructure, or require paid credentials.

## 18. Security, privacy, and public-repository precautions

- `.env`, credentials, private keys, database files, service volumes, downloaded data, models, MLflow artifacts, generated reports, logs, caches, and raw/processed data are ignored before the first commit.
- `.env.example` contains local placeholders only. Production-like modes refuse placeholder passwords/HMAC secrets. Secrets are read from environment or local untracked files, never config YAML or MLflow parameters.
- `.dockerignore` will prevent build-context leakage. Docker images will not copy `.git`, data, artifacts, reports, tests not needed at runtime, or environment files.
- Default prediction storage contains derived numeric features and outputs, not raw subject/body. Application/access logs do not log bodies. Tracebacks are sanitized at the boundary.
- Keyed HMAC, not plain content hash, supports duplicate-rate monitoring; the key is rotatable and excluded from source control.
- Input lengths, request size, concurrency, database pools, timeouts, and artifact paths are bounded. Model artifacts are loaded only from the expected registered model and validated signature/version.
- Database roles use least privilege; migrations do not run under the API role. SQLAlchemy parameterization and Pydantic validation protect common injection paths.
- Dependency lockfile, automated update review, `pip-audit` or another free local vulnerability check, container base-image review, and secret scanning (`detect-secrets` or equivalent) will be evaluated for CI/pre-commit without outsourcing data.
- Generated reports use aggregate examples or explicitly synthetic text and are reviewed before any portfolio result is committed.
- Dataset license/provenance and synthetic-data limitations are prominent. This is decision support, not autonomous routing for safety-critical support operations.
- Services bind locally. Any later public deployment is a new scope requiring authentication, TLS, rate limiting, secure secret management, backups, privacy review, and a separate zero-cost feasibility decision.

## 19. Documentation artifacts

- `README.md`: project purpose, architecture snapshot, prerequisites, zero-cost local quick start, canonical PowerShell/uv/Make commands, demo flow, and honest result links.
- `docs/architecture.md`: component, data-flow, deployment, model-lifecycle, and trust-boundary diagrams.
- `docs/data-card.md`: Hub revision/file, retrieval and hash procedure, license/attribution, schema, English filter, class selection, split statistics, duplicate handling, known quality issues, and no redistribution.
- `docs/model-card.md`: candidate/champion identity, training lineage, real measured metrics with support/uncertainty, intended use, exclusions, class-specific failure modes, privacy, and rollback.
- `docs/benchmarking.md`: machine specs, software/container versions, exact commands, config/data hashes, cold/warm definitions, repeated-run protocol, accuracy/latency/size results, and limitations.
- `docs/privacy-and-security.md`: data inventory, threat model, trust boundaries, redaction/storage/retention, secrets, incident response, and optional raw-content mode.
- `docs/runbook.md`: boot, migrations, seed/setup, ingestion, training, alias promotion/rollback, service health, backup/restore, and troubleshooting.
- `docs/monitoring-runbook.md`: window sufficiency, report interpretation, drift-versus-quality distinction, triage, retraining recommendation, and false-alert handling.
- ADRs: dataset source variant/revision, class threshold, split and duplicate strategy, primary metric, confidence policy, logging failure behavior, feedback raw-text boundary, Compose profiles, and promotion governance.
- Generated EDA, evaluation, model comparison, drift, data-quality, and load-test reports: reproducible locally and ignored unless a reviewed aggregate/static summary is deliberately incorporated into documentation.

## 20. Stage-by-stage implementation and acceptance criteria

### Stage 0 — Repository foundation

Deliver: Git on `main`, ignore policy, safe env template, `pyproject.toml`, lockfile, pre-commit, README skeleton, license decision, base directories/placeholders, and config schemas.

Accept when:

- Git status contains no environment file, generated workspace content, dataset, model, database, or volume.
- `uv sync --locked --all-groups` succeeds on Python 3.12 on the tested machine and the exact command/output is recorded.
- Ruff, mypy, pytest, and pre-commit configurations run locally; results are reported only after execution.
- A secret scan finds no real credential, and `.env.example` contains placeholders only.

### Stage 1 — Reproducible ingestion and validation

Deliver: pinned Hub revision/source config, downloader, raw manifest, Pandera schemas, quarantine/report, and license attribution.

Accept when:

- Two ingestions of the same revision produce the same source and manifest hashes.
- `main`/unpinned revisions are rejected.
- Required/forbidden field handling, English code, nulls, schema drift, and corrupt rows have tests.
- Data remains ignored and the source card documents verified row/schema/license facts without committing records.

### Stage 2 — Label definition, leakage audit, and splits

Deliver: top-ten class manifest, duplicate report/groups, stratified group manifests, seal metadata, guarded loaders.

Accept when:

- Exactly ten eligible queues are selected by count and deterministic tie-break from committed config, with no class names in source.
- Every split contains adequate support; ratios meet documented tolerance.
- Ticket/content group IDs are disjoint; regeneration is hash-identical.
- Model inputs contain exactly subject/body, forbidden-column tests pass, and routine code cannot load test data.

### Stage 3 — Train-only EDA and sparse baselines

Deliver: reproducible EDA, Dummy/NB/logistic/linear sparse experiments, local MLflow logging.

Accept when:

- EDA reads only the training manifest and contains no raw sensitive examples.
- Every model uses a single fitted pipeline and fold-local preprocessing.
- MLflow runs contain lineage/config/code hashes, real metrics, artifacts, and timing.
- Baseline comparison is reproducible from a documented command; no result is manually fabricated.

### Stage 4 — Candidate selection and registry

Deliver: bounded experiment runner, frozen selection rule, packaged model signature, registry version, candidate alias, promotion gates.

Accept when:

- The selected run follows the configured validation rule without test access.
- A clean process can load the registered version and reproduce predictions within defined tolerance.
- Signature, label map, dependency snapshot, hashes, and smoke tests are complete.
- Alias movement is audited and concurrent/failed updates leave a valid target.

### Stage 5 — Guarded final evaluation and champion

Deliver: single-use final evaluator, final report, model card update, explicit promotion/rollback command.

Accept when:

- Candidate/config/data hashes are frozen before test access.
- Final evaluation runs once for that tuple and records the access/audit event.
- Promotion uses predefined gates; failure leaves champion unchanged.
- Report states actual per-class support, uncertainty/limits, and exact reproduction command.

### Stage 6 — Database and FastAPI

Deliver: migrations, repositories/services, API contracts, champion loader, safe logging, health/readiness.

Accept when:

- Migrations apply to an empty PostgreSQL instance and schema constraints/integration tests pass.
- API predicts using the numeric champion version and returns a traceable prediction ID.
- Default DB/logs contain no raw subject/body; redaction/security tests pass.
- Feedback is idempotent, label-valid, and joins to prediction history.
- Startup/reload/database failure modes have deterministic readiness/error behavior.

### Stage 7 — Monitoring and delayed labels

Deliver: versioned safe reference profile, Evidently quality/drift, delayed-label metrics, reports, policy summary.

Accept when:

- Monitoring is idempotent by model/window/config and uses the matching reference.
- Insufficient samples produce no unsupported drift/quality claim.
- Synthetic known-shift tests trigger expected signals; stable fixtures do not exceed configured tolerances.
- Delayed-label metrics reconcile exactly with approved feedback and include per-class support.
- Reports contain no raw text or secrets.

### Stage 8 — Controlled retraining

Deliver: approved-snapshot intake, trigger/cooldown, challenger comparison, candidate registration, manual champion gate, rollback audit.

Accept when:

- Retraining cannot start without adequate, validated, hashed, authorized labeled data.
- Duplicate/concurrent runs are locked and failures cannot change champion.
- Challenger comparison uses frozen evaluation inputs and predefined regression/operational gates.
- Passing runs move only candidate; champion requires explicit approval; rollback is tested.

### Stage 9 — Prefect orchestration

Deliver: four flows, idempotent tasks/cache keys, local server/worker deployments, synchronous CLI fallback.

Accept when:

- Flows can run locally without hosted services and task states link to artifacts/hashes.
- Transient retries are bounded; deterministic failures do not loop.
- Repeated identical flow input is safely cached/idempotent.
- Schedules are disabled by default or explicitly enabled locally with documented resource impact.

### Stage 10 — Streamlit demo

Deliver: prediction, feedback, and monitoring pages through the API client.

Accept when:

- UI has no direct model/database import and handles API errors without exposing internals.
- Prediction/feedback round trip is demonstrated using synthetic/non-sensitive text.
- Version, privacy warning, human-oversight disclaimer, and monitoring sufficiency are visible.

### Stage 11 — Docker Compose and end-to-end operation

Deliver: hardened Dockerfiles, Compose services/profiles, health checks, volumes, migrations, local quick start.

Accept when:

- `docker compose config --quiet` and image builds succeed when actually run.
- A fresh local start reaches healthy state and completes migration → candidate/champion seed → predict → feedback → monitor smoke path.
- Data, artifacts, PostgreSQL, and MLflow persist in named/local ignored volumes across restart.
- Published services bind only to localhost by default; no paid/external service or card is required.

### Stage 12 — CI, benchmarking, and portfolio documentation

Deliver: GitHub Actions, local-equivalent checks, Locust protocol, finalized cards/runbooks/diagrams, reproducible results.

Accept when:

- Every CI check has a documented local `uv` equivalent and CI uses synthetic fixtures only.
- The workflow performs no deployment/push and requires no paid secret.
- Benchmark results include actual hardware/software, repetitions, commands, hashes, and raw local artifact references.
- README claims match recorded runs; fresh-clone instructions are successfully exercised on at least one documented environment.

## 21. Initial Git and GitHub workflow

- The first commit should contain only inspected planning/repository-hygiene artifacts and no generated data/state.
- Subsequent commits should be small, stage-oriented, and created only after relevant local checks run.
- Before each commit, inspect `git diff --check`, `git diff --cached`, `git status --ignored`, and a secret scan when configured.
- Do not add a GitHub remote or push until the user creates/configures the intended `production-mlops-ticket-router` remote.
- Before any push, verify the remote URL and branch protection expectations. Never state that a push occurred unless `git push` succeeds.
- Prefer reproducible generation over Git LFS. Use LFS only after a necessary tracked artifact is proven too large for normal GitHub and cannot reasonably be regenerated.

## 22. Major technical risks and mitigations

| Risk | Impact | Planned mitigation |
|---|---|---|
| Multiple mutable upstream CSV variants | Silent dataset changes or wrong schema | Inspect files, choose via config, pin full Hub SHA and digest, fail on `main` |
| CC BY-NC 4.0 dataset license | Limits reuse/commercial interpretation | Do not redistribute, attribute clearly, separate code license, confirm portfolio use |
| Synthetic/template-heavy tickets | Inflated offline metrics and weak real-world generalization | Disclose provenance, group duplicates, near-duplicate audit, train-only EDA, cautious model card |
| Post-submission leakage | Unrealistic routing performance | Physical feature allowlist, forbidden-field tests, pipeline accepts only subject/body |
| Duplicate leakage across splits | Inflated validation/test results | Content grouping before stratified split and disjointness tests |
| Top-ten selection instability | Non-reproducible label space | Pinned data, configured minimum, deterministic ranking/ties, generated manifest |
| Sparse model confidence misuse | Misleading UI/automation | Calibrate only with held-out training folds/validation evidence; otherwise expose scores without probability claim |
| Text drift without raw logging | Limited semantic diagnosis | Monitor safe proxies/aggregates; investigate through separately authorized source snapshots |
| Feedback lacks retraining features | Cannot train from privacy-safe prediction DB alone | Separate approved labeled snapshot connector; raw storage stays opt-in and off |
| Delayed/biased labels | Misleading quality and retraining gates | Minimum support, label source/latency metadata, per-class reporting, reviewer approval |
| Reusing the final test | Adaptive overfitting to portfolio benchmark | Single-use audit guard and new versioned evaluation protocol for later cycles |
| MLflow alias races or broken reload | Wrong model served | Central promotion service, lock/optimistic check, numeric version logs, last-known-good rollback |
| Local stack resource use, especially Windows Docker | Poor developer experience | CPU-friendly models, Compose profiles, bounded search, measured resource documentation |
| Localhost service exposure/secrets | Data or registry compromise | Local binding, separate roles, placeholder rejection, redacted logs, no public deployment |
| Reproducibility across BLAS/platforms | Small numerical/model differences | Locked deps, seeds, platform capture, prediction/metric tolerances where exact identity is unsupported |

## 23. Decisions requiring confirmation

1. Confirm that CC BY-NC 4.0 is acceptable for the intended non-commercial public portfolio and choose the code license (proposed: MIT, while clearly separating dataset terms).
2. After the first source profile, approve the exact upstream CSV/config and full Hub revision to pin. The currently advertised repository has multiple variants, so this cannot be responsibly guessed.
3. Approve the data-derived `min_class_count`, per-split minimum, and duplicate-similarity policy after counts are measured. `top_k = 10` and deterministic count/name ranking are already fixed by the requirement/plan.
4. Confirm the proposed 70/15/15 stratified-group split and validation macro F1 as the primary selection metric after feasibility is demonstrated.
5. Confirm that champion promotion remains manual even after all automated gates pass (recommended for a portfolio and safer by default).
6. Confirm the raw-text boundary: default no raw content in prediction logs; retraining receives a separate approved labeled snapshot. Optional encrypted raw demo storage would be a separate opt-in feature.
7. Confirm whether Docker Compose should start the full stack by default or use lightweight `core` and optional `orchestration` profiles (profiles are recommended for student hardware).
8. Confirm the eventual GitHub remote URL. No remote or push should occur before that confirmation.

## 24. Definition of project completion

The project is complete only when a fresh clone can, using documented local zero-cost commands, install locked dependencies; retrieve the pinned public dataset; validate/filter/select/split without leakage; reproduce EDA and sparse baselines; track/register a candidate; perform guarded evaluation and explicit champion promotion; serve predictions; log privacy-safe events and feedback; generate drift and delayed-quality reports; demonstrate controlled retraining without automatic promotion; run through Prefect and Streamlit; start through Docker Compose; pass the actually executed test/quality suite; and reproduce documented benchmark results without committed data, model blobs, secrets, paid APIs, hosted databases, or cloud infrastructure.
