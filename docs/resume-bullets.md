# Resume bullets

Use these only with the accompanying scope and limitations. Every number is traceable through
`artifacts/resume/project_metrics.json` to a verified generated artifact and reviewed repository
document.

- Built a leakage-safe 10-class support-ticket routing system over 28,190 synthetic English records,
  implementing hashed data lineage and exact-duplicate-grouped 19,729/4,232/4,229
  train/validation/test splits for reproducible local deployment.
- Engineered train-only sparse-text search, single-use sealed-test evaluation, and MLflow
  candidate/champion gates; promoted calibrated TF-IDF + LinearSVC version 1 after it achieved
  0.696057 test macro F1 and 0.468254 minimum per-class recall across 4,229 test tickets.
- Delivered a local FastAPI, Streamlit, PostgreSQL, MLflow, Evidently, and Prefect Compose stack with
  privacy-safe feedback and monitoring; measured 21.308660 ms loopback API p95, zero failures across
  a bounded 131-request Locust run, and passing behavior in all 9 reliability scenarios.

The dataset is synthetic and CC BY-NC 4.0. The benchmark is one local-machine result, not a public
production SLO.
