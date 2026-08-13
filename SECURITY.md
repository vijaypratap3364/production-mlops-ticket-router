# Security policy

## Supported scope

The current repository is a local portfolio and educational system. Security fixes target the
latest `main` branch. No public deployment, managed service, or production support commitment is
provided.

## Report a vulnerability

Do not publish a suspected vulnerability, credential, ticket payload, database dump, or exploit in
a public issue. Prefer GitHub private vulnerability reporting for this repository when it is
enabled. Otherwise, contact the maintainer privately through the repository owner's GitHub profile
and include only the minimum information needed to establish a secure channel.

Useful reports contain:

- the affected commit and component;
- a minimal reproduction using synthetic data;
- impact and prerequisites;
- whether secrets, raw text, model artifacts, or local service access may be exposed;
- a suggested remediation, if known.

Never test against infrastructure or data you do not own or have explicit permission to use.

## Security boundaries

- Every service binds to localhost by default. The stack has no public-network authentication, TLS
  termination, web application firewall, or distributed rate limiting.
- `.env`, credentials, data, databases, model artifacts, MLflow state, Docker volumes, and generated
  reports are ignored. Dockerfiles do not copy secrets into images.
- Raw ticket subject/body are not stored in the PostgreSQL schema. Redacted-text persistence is
  disabled by default; production-like fingerprints should use a secret HMAC key.
- The dashboard accesses the system only through FastAPI. It does not load a model or open a
  database connection.
- A model version cannot become champion through monitoring or retraining. Promotion requires a
  separate explicit human command after gate review.
- CI uses deterministic fixtures and build checks. It does not download the full dataset, evaluate
  the sealed test set, publish images, deploy services, or move registry aliases.

## Before any non-local deployment

Add and review TLS, authentication and authorization, rate limiting, a managed secret mechanism,
least-privilege database roles, encrypted backups, restore testing, audit logging, dependency and
image scanning, network segmentation, retention enforcement, incident response, and applicable
privacy/legal controls. The upstream dataset's CC BY-NC 4.0 license must also be reassessed for the
intended use.

See [privacy](docs/privacy.md), [deployment](docs/deployment.md), and
[database retention](docs/database.md).
