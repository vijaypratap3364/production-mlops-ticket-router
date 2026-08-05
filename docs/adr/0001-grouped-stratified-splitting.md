# ADR 0001: deterministic grouped-stratified data splitting

- Status: accepted
- Date: 2026-08-05

## Context

The pinned synthetic source contains exact normalized-text duplicates. A normal row-level random
split could place identical ticket text in training and evaluation data, inflating later model
metrics. A normalized-text group can also theoretically contain conflicting queue labels, in which
case choosing one label would silently fabricate certainty.

## Decision

- Normalize exact-text group keys with Unicode NFKC, case folding, collapsed whitespace, and
  SHA-256. Grouping uses the raw normalized combined `text`, before PII masking.
- Exclude every record in a group assigned to more than one `queue`. Do not use majority voting,
  first-label selection, or random tie-breaking.
- Allocate groups independently within each selected queue. Sort larger groups first and use a
  SHA-256 ordering key derived from random seed 42, queue, and group hash for deterministic ties.
- Assign each group to the split with the greatest remaining target deficit for that queue. Target
  ratios are 70% train, 15% validation, and 15% test.
- Require all labels in all splits, disjoint record IDs and group hashes, an absolute split-size
  tolerance of 0.01, and an absolute per-class proportion tolerance of 0.01.
- Keep the test Parquet sealed from routine training utilities. Only the explicit final-evaluation
  loader can read it, and that loader requires an authorization flag.
- Preserve normalized `text` and stateless cleaned `model_text` as separate local columns. Routine
  modeling receives only `model_text`; no vocabulary or other fitted state is learned here.

## Consequences

This policy prevents exact-duplicate leakage and makes regeneration stable for the same data,
configuration, seed, and code. Ratios are approximate because groups are indivisible. Excluding a
contradictory group reduces usable support but avoids guessing a label. The final test set must not
be inspected for model selection; preparation-time integrity counts are not model-performance
evaluation.
