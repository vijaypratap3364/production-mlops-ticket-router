# ADR 0005: API-only dashboard boundary

- Status: accepted
- Date: 2026-08-10

## Context

Streamlit can import Python modules directly, which makes it tempting to load the model or query
PostgreSQL from UI code. That would duplicate preprocessing, bypass FastAPI validation and metrics,
create multiple model-loading paths, expose credentials to the presentation layer, and make local
and container behavior diverge.

## Decision

The dashboard is an adapter and communicates through a typed FastAPI client only. It may send
single/batch predictions and feedback and read model, monitoring, and system-status projections. It
must not import modeling/registry services, load MLflow artifacts, construct SQL, or access
PostgreSQL. Shared business rules live below the API route layer; the API owns request validation,
preprocessing, model lifecycle, persistence, errors, logs, and metrics.

## Consequences

One contract serves Streamlit and other clients, operational behavior is observable in one place,
and the UI can fail with sanitized errors when the API is unavailable. The dashboard requires a
running API and has an extra local HTTP hop, measured in Stage 15 rather than assumed away.

## Alternatives considered

- Direct model import in Streamlit: rejected because it creates inconsistent preprocessing and
  per-process registry/model lifecycle.
- Direct read-only SQL for monitoring pages: rejected because it leaks persistence contracts and
  credentials into UI code.
- Duplicate a second lightweight API client per page: rejected in favor of one reusable typed
  client and centralized error handling.
