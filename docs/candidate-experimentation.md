# Stage 6 candidate experimentation

## Scope and leakage boundary

Stage 6 compares five maintainable sparse-text candidate families. Search and vocabulary fitting
use only 19,729 training records. Three-fold `StratifiedKFold` cross-validation is shuffled with
seed 42. The selected parameters are refitted on the complete training split, then evaluated once
on 4,232 validation records. The 4,229-row test split was neither loaded nor evaluated.

The candidate families are word TF-IDF + logistic regression, character `char_wb` TF-IDF +
logistic regression, a word/character `FeatureUnion` + logistic regression, word TF-IDF +
LinearSVC, and word TF-IDF + ComplementNB. The selected LinearSVC parameters are calibrated with
three training-only folds and sigmoid calibration so probability metrics and confidence analysis
are valid. No validation record participates in vectorizer fitting, parameter search, or
calibration.

`configs/experiments.yaml` limits each family to two trials. The known Stage 5 logistic-regression
configuration is guaranteed as one word-model trial, preventing a small randomized search from
accidentally discarding the proven incumbent. All other trials are sampled deterministically from
the versioned search space.

## Reproduction and MLflow

Run from the repository root after data preparation:

```text
uv sync --locked --all-groups
uv run python -m ticket_router.modeling.train_candidates
```

The experiment uses `MLFLOW_TRACKING_URI` when it is available. During the measured run,
`http://127.0.0.1:5000` was unavailable, so tracking fell back to the ignored local SQLite file
`mlruns/mlflow.db`. This is a zero-cost local backend and can later be replaced by the planned
PostgreSQL MLflow backend without changing experiment code. To inspect the local runs:

```text
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
```

Experiment name: `ticket-router-candidate-search`

Experiment run: `cand-20260805T224856Z-8df1c27d`

Configuration SHA-256:
`8df1c27dccf8cfd9e868141a4a552b908961b002da31e8c5c7aa6e74550db80a`

Split-manifest SHA-256:
`e7b162f5f439d4327648524a01eae0666dd4d2e755e5600327103098fe6285de`

Training-file SHA-256:
`9c9c7dfe47107dd59fdafd4a421d376c701ecf91815af553111436d1deddd011`

The run was executed from Git commit `1195b02b2f0e1d1cb033921881fd36f2923ef64a` with Stage 6
changes uncommitted, so lineage records `git_dirty=true`. Re-running after committing will produce
a different, cleaner code identity and may produce different local timing measurements.

## Measured validation comparison

| Candidate | MLflow run ID | CV macro F1 mean ± std | Validation macro F1 | Weighted F1 | Accuracy | Eligible |
|---|---|---:|---:|---:|---:|---|
| Calibrated word TF-IDF + LinearSVC | `03ada1859756492bb9619c5e873099dc` | 0.66750 ± 0.00734 | 0.67812 | 0.66194 | 0.66304 | Yes |
| Word + character TF-IDF + logistic regression | `65980ffb10524e6494c7658fad8ebc71` | 0.58914 ± 0.00772 | 0.58807 | 0.58094 | 0.58105 | Yes |
| Word TF-IDF + logistic regression | `160b74f25d6941139bfdaaa4460c9fa4` | 0.59787 ± 0.00844 | 0.58822 | 0.57410 | 0.57491 | Yes |
| Character TF-IDF + logistic regression | `a85bd352c04c41f2a1c568ebb497f92c` | 0.45602 ± 0.00800 | 0.44840 | 0.46031 | 0.45794 | No |
| Word TF-IDF + ComplementNB | `de1b80996428457b9b2362a21a1c3a8c` | 0.41869 ± 0.00240 | 0.43819 | 0.46317 | 0.47519 | No |

The selected calibrated LinearSVC also measured macro precision 0.76711, macro recall 0.62379,
log loss 1.10239, 0.11446 milliseconds per record for local batch inference, 31.9214 seconds to fit
the final calibrated pipeline, and 9,797,750 serialized bytes. These are observations on the local
Windows workspace, not production service-level guarantees.

## Selection and rejection reasons

Validation macro F1 is primary. A candidate must also satisfy all versioned guardrails: minimum
per-class recall 0.30, minimum recall 0.40 for validation classes with at least 400 records,
inference at most 1.0 millisecond per record, serialized size at most 100 MiB, CV macro-F1 standard
deviation at most 0.03, and CV-to-validation macro-F1 drop at most 0.10.

Calibrated LinearSVC ranked first and passed every guardrail. Its lowest validation class recall was
0.50952 (`Returns and Exchanges`), above the global threshold. The word logistic and combined
logistic candidates also passed but were rejected by ranking because their validation macro F1 was
lower. Character logistic was ineligible because `Product Support` and `Customer Service` recall
fell below the major-class threshold. ComplementNB was ineligible because five classes fell below
0.30 recall and two major classes fell below 0.40.

## Logged and generated artifacts

Each MLflow run contains model parameters and tags, Git/data/configuration lineage, package
versions, aggregate and per-class metrics, cross-validation results, classification report,
confusion matrix, benchmark, model size, fitted pipeline, inferred model signature, and a synthetic
input example. The input example contains no dataset ticket text.

Ignored local reports are generated under
`artifacts/reports/candidates/cand-20260805T224856Z-8df1c27d/`:

- `candidate_comparison.json`
- `cross_validation_summary.json`
- `mlflow_run_comparison.json`
- `validation_error_analysis.json`
- `confidence_distribution.json`
- `calibration_points.json`
- `calibration_plot.png`
- `experiment_lineage.json`

`artifacts/reports/model_leaderboard.csv` is regenerated with Stage 5 and Stage 6 rows. Candidate
model directories and all report/model artifacts remain ignored because they are reproducible and
may be large. A reload check successfully loaded the selected model from
`runs:/03ada1859756492bb9619c5e873099dc/model` and produced two predictions from synthetic inputs.

## Limitations and next boundary

The dataset is synthetic and template-heavy; validation performance may not transfer to real
support traffic. Confidence calibration is measured on the same validation split used for final
candidate comparison, so it is diagnostic rather than a claim of production calibration. Stage 7
subsequently performed the single authorized final evaluation and registered the model as
`candidate`; see `docs/model-card.md`. Serving remains a later-stage boundary.
