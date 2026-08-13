# ADR 0004: privacy-safe prediction persistence

- Status: accepted
- Date: 2026-08-09

## Context

Monitoring and delayed-label evaluation require production metadata, but support tickets can contain
personal, account, or confidential text. Retaining raw subject/body by default would increase harm,
retention obligations, and repository/demo risk. The API must also continue predicting if optional
analytics storage is temporarily unavailable.

## Decision

The PostgreSQL schema has no raw subject/body columns. Persist model/version, predicted label,
confidence/top-k, lengths and safe derived ratios/counts, latency, approved request metadata, and a
one-way text fingerprint. Prefer HMAC-SHA-256 with a local secret; plain SHA-256 is a development
fallback. Optional conservatively redacted text is disabled by default. Prediction metadata writes
are best-effort; feedback requires a durable source prediction, is create-once, and returns 503 on
database failure.

## Consequences

The service supports aggregate monitoring and joined delayed labels while materially reducing raw-
text exposure. It cannot retrain directly from PostgreSQL, so a governed human-approved export must
provide subject/body when retraining is justified. Fingerprints and aggregates can still be
sensitive and require access/retention controls. Masking is not anonymization.

## Alternatives considered

- Store all raw text for error analysis: rejected on data-minimization grounds.
- Store no prediction events: rejected because monitoring, reliability, and delayed-label lineage
  would be unavailable.
- Fail inference when analytics logging fails: rejected because routing can succeed safely without
  optional persistence; feedback remains strict because accepting an orphan label would be unsafe.
