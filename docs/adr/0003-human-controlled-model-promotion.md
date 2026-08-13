# ADR 0003: human-controlled champion promotion

- Status: accepted
- Date: 2026-08-08

## Context

Monitoring, delayed labels, and scheduled retraining can identify or produce a candidate, but drift
does not prove quality regression, feedback can be biased, and aggregate gates cannot capture every
business or privacy concern. Automatically moving a serving alias from a scheduled workflow would
make deployment a side effect of noisy observations.

## Decision

MLflow versions are immutable and use aliases rather than deprecated lifecycle stages. Training and
Prefect workflows may register a version and assign `candidate`. Absolute and champion-relative
gates evaluate macro F1, per-class recall, latency, loadability, prediction contract, and signature.
No flow, schedule, CI job, API startup, or dashboard action may assign `champion`. Promotion requires
the separate explicit command `uv run python -m ticket_router.registry.promote --approve` after a
human reviews the recorded evidence.

## Consequences

Serving changes are deliberate, auditable, and reversible by explicitly assigning a previously
verified immutable version. The process adds a manual step and can delay deployment, but avoids
turning drift or training success into unreviewed production behavior. The first champion was
promoted only after all absolute gates passed; future candidates also face regression tolerance
against the existing champion.

## Alternatives considered

- Automatic promotion on gate pass: rejected because thresholds cannot express all operational and
  governance context.
- Deprecated MLflow stages: rejected in favor of aliases supported by the installed MLflow version.
- Loading `candidate` when `champion` is absent: rejected because it silently bypasses approval.
