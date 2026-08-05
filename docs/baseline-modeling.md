# Stage 5 baseline modeling

## Scope and selection rule

Stage 5 establishes maintainable local sparse-text baselines. Every model was fitted on the 19,729
training records and evaluated once on the 4,232 validation records. The sealed test Parquet was
not loaded or evaluated. Macro F1 is the primary selection metric because the ten queues are
imbalanced and performance on smaller queues matters.

This stage compares one fixed configuration rather than performing a hyperparameter search:

- a most-frequent `DummyClassifier`;
- word unigram/bigram TF-IDF followed by balanced one-vs-rest logistic regression; and
- the same non-negative TF-IDF representation followed by `ComplementNB`.

ComplementNB is technically appropriate here because TF-IDF emits non-negative sparse features.
The vectorizer is part of each serialized non-dummy pipeline and is fitted only through
`pipeline.fit(training_text, training_labels)`. The dummy pipeline contains a stateless shape
adapter and the classifier. Validation and test text never participate in vocabulary construction.

## Reproduction

From the repository root after completing Stages 2 through 4:

```text
uv sync --locked --all-groups
uv run python -m ticket_router.modeling.train_baseline
```

`configs/baseline.yaml` fixes TF-IDF, classifier, inference benchmark, and error-sample settings.
The global seed remains 42. Each run records the baseline configuration hash, split-manifest hash,
input file hashes, Git state, Python/package versions, timings, model hashes, and test-access flag.

## Measured validation results

Run: `baseline-20260805T220008.955971Z-999ab3460b`

Data manifest SHA-256:
`e7b162f5f439d4327648524a01eae0666dd4d2e755e5600327103098fe6285de`

Baseline configuration SHA-256:
`999ab3460bbed3ce8b6130f88a64b38b1bbc91d3218e611bc3e9357fbfaca64f`

| Model | Macro F1 | Weighted F1 | Accuracy | Macro precision | Macro recall | Log loss |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + logistic regression | 0.58822 | 0.57410 | 0.57491 | 0.61546 | 0.57284 | 1.50904 |
| TF-IDF + ComplementNB | 0.42447 | 0.45845 | 0.47472 | 0.68601 | 0.37050 | 1.84598 |
| Most-frequent dummy | 0.04478 | 0.12921 | 0.28852 | 0.02885 | 0.10000 | 25.64448 |

The TF-IDF logistic regression pipeline is the strongest baseline by the predefined primary
metric. Accuracy was reported but was not used for selection.

## Local operational measurements

The measurements below came from Python 3.12.13 and scikit-learn 1.9.0 on the local Windows 11
workspace. Inference used batches of 256 over the entire validation split, with one warm-up batch
and the median of three repetitions. These figures are local observations, not service-level
targets.

| Model | Training seconds | Median ms/record | Serialized bytes |
|---|---:|---:|---:|
| TF-IDF + logistic regression | 6.9013 | 0.08043 | 4,376,493 |
| TF-IDF + ComplementNB | 1.8329 | 0.08310 | 6,013,196 |
| Most-frequent dummy | 0.0342 | 0.00083 | 972 |

## Error analysis

For the strongest baseline, the lowest-recall queues were `IT Support` (0.47495), `Returns and
Exchanges` (0.49048), `General Inquiry` (0.50000), `Product Support` (0.50126), and `Customer
Service` (0.50235). The most common directional confusions were:

1. `Product Support` predicted as `Technical Support`: 168 records.
2. `Technical Support` predicted as `Product Support`: 129 records.
3. `IT Support` predicted as `Technical Support`: 128 records.
4. `Technical Support` predicted as `IT Support`: 114 records.

The highest-confidence incorrect logistic-regression prediction had confidence 0.53405. Confidence
values are uncalibrated model outputs and must not be interpreted as operational certainty.
Generated examples preserve punctuation structure but replace all text tokens with `[TOKEN]`; raw
ticket text is not written to error-analysis or prediction artifacts.

## Generated artifacts

The run directory is `artifacts/models/baselines/<run-id>/<model-name>/`. Each model receives:

- `pipeline.joblib`
- `metrics.json`
- `classification_report.json`
- `confusion_matrix.png`
- `per_class_metrics.csv`
- `validation_predictions.parquet`
- `model_configuration.json`
- `training_metadata.json`
- `inference_benchmark.json`
- `error_analysis.json`

The aggregate ranking is written to `artifacts/reports/model_leaderboard.csv`. Generated reports,
predictions, and model binaries are ignored by Git and must be reproduced locally.

## Limitations and next step

The source data is synthetic and template-heavy, so these validation metrics may overstate
real-world behavior. Exact duplicate groups are isolated across splits, but semantic near-duplicate
templates may remain. No extensive tuning, probability calibration, MLflow tracking, registry
operation, candidate selection, or test evaluation was performed in this stage.
