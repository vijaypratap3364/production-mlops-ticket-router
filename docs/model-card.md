# Model card: ticket-router version 1

## Model details

The model routes an English customer-support ticket to one of ten support queues. It is a local,
CPU-oriented sparse-text classifier intended to demonstrate a production-style MLOps lifecycle.
It does not generate answers, resolve tickets, assign agents, or make staffing decisions.

MLflow registered-model name: `ticket-router`

Registered version: `1`

Current aliases: `candidate` → version `1`; `champion` → version `1` after explicit human-approved
bootstrap promotion.

Final MLflow run: `042a28f803c24c79b5c1051f2e72a880`

## Architecture and prediction contract

The fitted scikit-learn `Pipeline` contains word TF-IDF followed by a calibrated LinearSVC:

- word unigrams and bigrams;
- `min_df=2`, `max_df=1.0`, and at most 50,000 features;
- sublinear term frequency;
- LinearSVC with `C=1.5`, no class weighting, `max_iter=5000`, and seed 42;
- three-fold training-only sigmoid calibration.

The registered MLflow signature is one variable-length string tensor in and one variable-length
string tensor out. Each input string is the conservatively cleaned `model_text` derived solely from
the submitted `subject` and `body`. The output is exactly one queue label. The FastAPI layer
must construct `model_text` through the versioned preprocessing code and must not pass queue,
answer, response, priority, type, tags, resolution, or agent-generated fields to the model.

## Dataset and classes

The source is the synthetic
[`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)
dataset at revision `ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb`, marked CC BY-NC 4.0.
Only English records were retained. Exact normalized-text duplicate groups were kept within one
split, and contradictory duplicate groups would have been excluded.

The ten selected queues are:

1. Technical Support
2. Product Support
3. Customer Service
4. IT Support
5. Billing and Payments
6. Returns and Exchanges
7. Service Outages and Maintenance
8. Sales and Pre-Sales
9. Human Resources
10. General Inquiry

The final fit combined the previously separated training and validation splits: 19,729 + 4,232 =
23,961 records. The untouched 4,229-record test split was opened once after the pipeline,
parameters, label space, and promotion thresholds were frozen.

## Training and selection procedure

Stage 6 used three-fold stratified cross-validation on training data and one validation comparison
across five sparse candidate families. Word TF-IDF + calibrated LinearSVC was selected by validation
macro F1 after passing recall, stability, latency, and size guardrails. Its validation metrics were:

| Metric | Value |
|---|---:|
| Macro F1 | 0.678889 |
| Weighted F1 | 0.664230 |
| Accuracy | 0.665406 |
| Macro precision | 0.771256 |
| Macro recall | 0.623157 |
| Log loss | 1.102612 |

No test result influenced model-family selection, parameters, class selection, preprocessing, or
promotion thresholds.

## Final test results

The following values are from the single authorized test evaluation on 2026-08-11:

| Metric | Value |
|---|---:|
| Macro F1 | 0.696057 |
| Weighted F1 | 0.688432 |
| Accuracy | 0.689761 |
| Macro precision | 0.785512 |
| Macro recall | 0.641317 |
| Multiclass log loss | 1.042178 |
| Mean maximum confidence | 0.569149 |
| Mean confidence, correct predictions | 0.616981 |
| Mean confidence, incorrect predictions | 0.462805 |

Local batch inference over all 4,229 records used seven repetitions with batches of 256. Median
latency was 0.114693 ms/record and median throughput was 8,718.92 records/second. The serialized
local pipeline was 10,030,367 bytes. These measurements describe the local evaluation environment;
they are not production service-level objectives.

### Per-class test performance

| Queue | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Technical Support | 0.6360 | 0.8049 | 0.7106 | 1,220 |
| Product Support | 0.6420 | 0.6444 | 0.6432 | 793 |
| Customer Service | 0.6404 | 0.6495 | 0.6449 | 639 |
| IT Support | 0.6538 | 0.5411 | 0.5921 | 499 |
| Billing and Payments | 0.9059 | 0.8203 | 0.8609 | 434 |
| Returns and Exchanges | 0.8516 | 0.5190 | 0.6450 | 210 |
| Service Outages and Maintenance | 0.9214 | 0.7818 | 0.8459 | 165 |
| Sales and Pre-Sales | 0.8310 | 0.4683 | 0.5990 | 126 |
| Human Resources | 0.9310 | 0.6506 | 0.7660 | 83 |
| General Inquiry | 0.8421 | 0.5333 | 0.6531 | 60 |

The weakest recall is `Sales and Pre-Sales` at 0.4683, followed by `Returns and Exchanges`,
`General Inquiry`, and `IT Support`. Smaller classes have wider uncertainty and require monitoring.
High precision but lower recall for several minority queues means the model is conservative about
assigning them and may over-route ambiguous tickets to larger support queues.

## Calibration behavior

The LinearSVC scores are converted to probabilities through three-fold sigmoid calibration fitted
only within the combined training data. Test log loss was 1.042178. Mean maximum confidence was
0.5691 compared with accuracy of 0.6898; correct predictions averaged 0.6170 confidence and errors
averaged 0.4628. This evidence suggests useful confidence separation but does not justify treating
the scores as guaranteed probabilities or automating high-impact decisions without monitoring.

## Promotion gates and lineage

All frozen initial-promotion gates passed:

- macro F1 0.696057 ≥ 0.60;
- minimum class recall 0.468254 ≥ 0.45;
- median latency 0.114693 ms/record ≤ 1.0 ms/record;
- logged artifact loaded successfully;
- synthetic prediction-contract test passed;
- MLflow signature matched the planned API contract.

There was no prior champion, so no relative regression comparison was required. Passing gates did
not move the champion alias automatically. Stage 13 later invoked the explicit command with human
approval and recorded `champion` version 1:

```text
uv run python -m ticket_router.registry.promote --approve
```

Key lineage hashes:

- split manifest: `42a89d794b3a175dd11f9fac3c4f537b6d5e993fdbd6dc0e48466ae91845b28c`
- combined train+validation: `0ef52577bace28e211da26410b5c05e1c3faac43a55872f8b7ad3bac37def6d6`
- test data: `19a4b1c0db4ed2d8f4fb989c924692a5726bb7b9507581fa6509c777003cad47`
- final configuration: `76301b418083667f27c839783f582baf5f97e87d2a27ab43215c1af18c3d44f8`
- serialized model: `e20b1a355748b3d613b33dd3723157f8c2f641cfc031b1edf7b3a53aa8b9293f`
- source Git commit: unavailable in the persisted recovery artifact; all data/configuration hashes
  and the MLflow run ID remain recorded

The first post-evaluation registration attempt encountered a Windows console-encoding error after
the evaluation and model were already logged. Registration recovery used only the saved evaluation
artifacts and READY logged model; it did not reload or re-evaluate the test data. The permanent
access audit records this recovery.

## Inappropriate uses and limitations

Do not use this model for emergency triage, safety-critical decisions, legal or medical routing,
employee evaluation, customer eligibility, fully autonomous ticket closure, or languages other
than English. Do not interpret a predicted queue or probability as a resolution or factual answer.

The dataset is synthetic and template-heavy. Even with exact-duplicate isolation, semantic template
similarity may inflate offline performance. Real tickets can contain organization-specific products,
slang, multilingual text, novel queues, and distribution shifts absent from this benchmark. The
CC BY-NC dataset terms also limit commercial reuse independently of the repository's MIT code
license.

## Privacy and security

Only subject/body-derived cleaned text is accepted. Email addresses, URLs, and configured phone-like
sequences are masked during preprocessing. Raw ticket content is not written to MLflow input
examples, error-analysis reports, or prediction logs by default. Error examples contain token-redacted
structural excerpts. MLflow models use cloudpickle and must be loaded only from this trusted local
registry; untrusted pickle artifacts can execute code.

## Monitoring and retraining expectations

Deployment must monitor input schema failures, empty/length distributions, safe text proxies,
prediction and confidence distributions, queue drift, latency, model version, and delayed-label
macro/per-class metrics. Particular attention is required for Sales and Pre-Sales, Returns and
Exchanges, General Inquiry, and IT Support recall.

Retraining should be considered only after sufficient reviewed labels, material data/prediction
drift, sustained quality degradation, new queue definitions, or preprocessing changes. Retraining
must use a new approved labeled snapshot, preserve a new sealed evaluation protocol, compare against
the numeric champion version, pass the same or stricter gates, register as `candidate`, and require
explicit human promotion. The Stage 7 test split must not become a recurring tuning dataset.
