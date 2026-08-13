# Interview notes

These are concise evidence-based answers. Customize the personal-contribution answer so it remains
truthful to how you built and reviewed the project.

## What problem does the system solve?

It routes an incoming English support ticket to one of ten queues from its subject and body. The
business goal is to reduce manual triage and avoid handoffs while preserving human review for
uncertain predictions. The scope includes the lifecycle around the classifier—reproducible data,
evaluation, registry, serving, feedback, monitoring, and controlled retraining—not only a notebook.

## Why was macro F1 selected?

The measured largest-to-smallest queue ratio is 20.3885. Accuracy and weighted F1 can look strong
while minority queues fail. Macro F1 gives every queue equal weight and balances precision with
recall. Selection also used a minimum per-class recall gate so one weak class could not be hidden by
the average.

## How was leakage prevented?

The feature contract accepts only subject, body, and text derived from them. It explicitly rejects
queue, answers/responses, priority, type, tags, agent fields, and resolutions. Exact normalized-text
duplicates are grouped before splitting, contradictory-label groups are excluded, and group hashes
cannot cross train/validation/test. TF-IDF vocabulary and hyperparameter search fit on training
only. Validation performs candidate comparison, and a persistent audit permits final test access
only after the configuration is frozen.

## Why use TF-IDF and a linear classifier?

Support tickets are high-dimensional sparse text, where TF-IDF plus a linear decision boundary is a
strong, interpretable, CPU-friendly baseline. It trains and serves locally, serializes compactly,
and makes feature leakage easier to audit than an unnecessary deep model. The calibrated LinearSVC
beat word/character logistic-regression and ComplementNB alternatives on validation macro F1 while
meeting recall, stability, latency, and size guardrails.

## How was the champion selected?

Five model families used restrained train-only stratified cross-validation. The calibrated word
TF-IDF + LinearSVC candidate achieved 0.678889 validation macro F1 and passed guardrails. Its frozen
pipeline was refit on train plus validation and evaluated once on 4,229 sealed test tickets,
achieving 0.696057 macro F1 and 0.468254 minimum class recall. MLflow registered version 1 as
`candidate`; every absolute gate passed, and a separate explicit approval assigned `champion`.

## What does MLflow track?

It tracks experiment/run names, model family and parameters, random seed, Git identity when
available, dataset/training/configuration hashes, Python/package versions, CV and validation/test
metrics, per-class reports, confusion matrix, calibration/confidence artifacts, training and
inference timing, model size, signature, synthetic input example, and the fitted pipeline. Registry
versions link back to runs and lineage artifacts.

## How is drift detected?

The monitoring reference is derived from final training data and scored by the champion. Current
events come from a bounded PostgreSQL time window and are filtered by model version. Evidently
compares explicit privacy-safe features: lengths, word count, character ratios, URL/email counts,
predicted queue, confidence, and low-confidence status. The report distinguishes feature and
prediction drift from delayed-label quality. Status requires multiple signals rather than one
threshold.

## What triggers retraining?

Examples are sustained critical multi-signal drift, enough approved delayed labels plus macro-F1
decline, sustained low-confidence increase with supporting evidence, or a manual request. The flow
still requires a configured minimum label count and creates a hashed parent-linked dataset.
`insufficient_data` or a single small drifted window does not trigger training.

## Why is promotion human-controlled?

Drift does not necessarily mean lower quality, delayed labels may be biased, and automated training
can produce an eligible but operationally inappropriate model. Human review provides an explicit
checkpoint for lineage, class weaknesses, privacy, metrics, and business context. Prefect may
register/update `candidate`, but no monitoring, retraining, CI, or startup path can move
`champion`.

## What happens if PostgreSQL fails?

The model remains in memory. A successful prediction still returns even if optional analytics
persistence fails; the service increments a persistence-failure metric and logs a content-free
error. Feedback returns structured 503 because it must reference an existing prediction and cannot
be safely accepted without durable storage. Stage 15 verified both behaviors during a controlled
outage.

## What would change at larger scale?

I would separate services and credentials, add TLS/auth/rate limiting, use managed or replicated
PostgreSQL and object storage, introduce a queue for asynchronous analytics writes, autoscale API
workers after memory/load tests, centralize logs/metrics/traces, add backup/restore and incident
runbooks, enforce automated retention, and use governed real ticket data. I would benchmark again
before changing workers or models, and keep the registry/human approval boundary.

## What are the dataset’s limitations?

It is synthetic and CC BY-NC 4.0, contains repetitive templates and overlapping source variants,
trusts provided language labels, has a source-specific queue taxonomy, and is highly imbalanced.
Offline results may overstate performance on organic tickets. The weakest test class recall is
0.468254, so the system should remain decision support with monitoring and review.

## What did I personally implement?

Suggested answer—edit this to match your actual contribution: “I designed and implemented the
end-to-end local system: revision-pinned ingestion and manifests, Pandera contracts, duplicate-safe
splits, sparse baselines and candidate search, MLflow lineage/registry gates, FastAPI serving,
SQLAlchemy/Alembic persistence, Evidently monitoring, Prefect retraining controls, Streamlit,
Docker Compose, CI, tests, benchmarking, and documentation. I also ran and recorded the actual
data, model, reliability, and latency results.”

## What was the hardest engineering decision?

The hardest decision was protecting evaluation integrity while keeping the workflow recoverable.
Exact duplicates required group-aware stratification, and the held-out test needed a persistent
single-use audit so a failed registration step could recover without evaluating again. The solution
separates frozen evaluation, immutable MLflow version registration, gate recording, and explicit
alias promotion. It is more ceremony than a notebook, but it prevents optimistic leakage and
test-driven tuning.
