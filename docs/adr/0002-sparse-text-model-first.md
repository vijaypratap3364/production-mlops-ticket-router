# ADR 0002: sparse-text models before deep learning

- Status: accepted
- Date: 2026-08-06

## Context

The routing target has ten imbalanced classes and 28,190 synthetic English records. The project is
zero-cost, must train and serve locally on student hardware, and must make leakage, preprocessing,
latency, and model size easy to audit. Deep neural models would add download/compute cost and
operational complexity before establishing whether classical sparse text is sufficient.

## Decision

Establish DummyClassifier, word TF-IDF + LogisticRegression, and ComplementNB baselines. Compare a
focused Stage 6 set of word, character, combined word/character, calibrated LinearSVC, and
ComplementNB candidates. Fit vectorizer vocabulary inside scikit-learn pipelines on training data
only. Select by validation macro F1 with per-class recall, CV stability, latency, size, and load
guardrails. Do not introduce deep learning unless measured errors and scale justify it.

## Consequences

Training and inference remain CPU-friendly, reproducible, understandable, and compact. The selected
calibrated word TF-IDF + LinearSVC achieved 0.678889 validation macro F1 and 0.696057 on the single
authorized test evaluation. Sparse features can be weaker on semantic paraphrase, multilingual
input, and long-range context; those limitations are documented rather than hidden by complexity.

## Alternatives considered

- Transformer fine-tuning: rejected for the initial system because it adds compute, model size, and
  lifecycle complexity without a baseline-backed need.
- Hosted embeddings or LLM classification: rejected because it violates zero-cost/local-only and
  privacy constraints.
- Rules only: retained as a possible operational fallback, but insufficient for a portfolio ML
  lifecycle and likely brittle across ten queues.
