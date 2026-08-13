# Privacy statement

This project demonstrates privacy-conscious engineering, but it is not a legal compliance
certification. It currently uses a synthetic, noncommercially licensed public dataset and local
synthetic requests for tests and benchmarks.

## Data minimization

Only `subject` and `body` are used to predict `queue`. Agent answers, responses, priority, ticket
type, tags, assigned-agent fields, resolutions, and other post-submission values are excluded from
normalized model inputs. Raw and processed datasets remain local and ignored by Git.

Text preprocessing applies Unicode and whitespace normalization and masks email addresses, URLs,
and optionally phone-like sequences. Masking reduces accidental exposure but is not guaranteed to
remove every kind of personal or confidential information.

## Prediction persistence

The PostgreSQL `prediction_events` table has no raw `subject` or `body` columns. By default it stores
only:

- request/model identity, UTC timestamp, predicted queue, confidence and top-k;
- lengths, word count, ratios, URL/email counts and low-confidence status;
- inference latency and an optional language indicator;
- approved operational metadata (`client_name` and `correlation_id`);
- a one-way text fingerprint.

Configure `INPUT_HMAC_SECRET` to use HMAC-SHA-256 fingerprints. Plain SHA-256 exists only as a
secret-free local fallback and can permit dictionary attacks against predictable text. Raw ticket
storage is unsupported. `STORE_REDACTED_TICKET_TEXT=false` is the default; the opt-in redacted value
still requires access and retention review because masking is imperfect.

Feedback stores the corrected queue, nullable acceptance, bounded optional comment, source, and
original model version. Comments can contain sensitive information if users paste it, so the demo
asks for non-sensitive comments only.

## Logs, reports, and UI

Normal API and dashboard logs omit ticket subject/body, preprocessed text, feedback comments,
credentials, database URLs, and raw exception traces. Reports contain aggregates, class labels,
hashes, statuses, and privacy-safe examples or tokens. Streamlit may display the user's current
input in their browser session but does not include it in downloaded batch results or normal logs.

Monitoring sends only declared derived features and model outputs to local Evidently code. It does
not pass raw free-form text into generic drift metrics. MLflow input examples are synthetic and
artifacts remain in a local ignored volume.

## Retention and access

Documented local limits are:

- prediction metadata: 90 days;
- explicitly enabled redacted text: 30 days;
- feedback: 365 days or less when no longer needed;
- monitoring reports: 365 days;
- retraining lineage: project lifetime unless sensitive content is discovered.

Automated deletion is a future improvement, so the operator is responsible for enforcement. Keep
PostgreSQL, MLflow, Prefect, data, artifacts, and reports on a controlled local machine. Do not
publish volumes or expose ports beyond localhost. Rotate HMAC/database secrets and use separate
least-privilege roles before production-like use.

## User rights and real-data use

The synthetic source does not provide a production data-subject workflow. Before using real
tickets, establish a lawful basis, notice/consent where required, purpose limits, access and
deletion procedures, incident response, regional requirements, retention automation, human review,
and a governed label-correction process. Do not assume the upstream CC BY-NC license permits a
commercial deployment.

See [data source](data-source.md), [database schema and retention](database.md), and
[security policy](../SECURITY.md).
