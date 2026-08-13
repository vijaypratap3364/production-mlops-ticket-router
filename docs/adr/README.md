# Architecture decision records

ADRs capture decisions that materially affect evaluation integrity, model lifecycle, privacy,
service boundaries, or operations. An accepted ADR is not immutable, but changing it requires a new
record that links to and supersedes the earlier decision.

| ADR | Status | Decision |
|---|---|---|
| [0001](0001-grouped-stratified-splitting.md) | Accepted | Group exact duplicates during deterministic stratified splitting |
| [0002](0002-sparse-text-model-first.md) | Accepted | Prefer sparse TF-IDF linear candidates before deep learning |
| [0003](0003-human-controlled-model-promotion.md) | Accepted | Separate automated candidate registration from human champion promotion |
| [0004](0004-privacy-safe-prediction-persistence.md) | Accepted | Persist safe metadata and delayed labels, not raw ticket text |
| [0005](0005-api-only-dashboard-boundary.md) | Accepted | Keep Streamlit behind the FastAPI application boundary |

New records should state context, decision, consequences, alternatives, status, and date. Do not
edit an old accepted record to hide a changed decision.
