# 90-second portfolio demo

## Before recording

Build and bootstrap once before the timed demo. Use only synthetic text and keep `.env`, database
URLs, terminal history containing secrets, raw datasets, and artifact directories out of view.

```bash
docker compose up -d postgres mlflow migrate api dashboard
docker compose restart api
curl --fail http://127.0.0.1:8000/ready
```

Open these tabs in advance:

1. MLflow experiments at `http://127.0.0.1:5000`;
2. MLflow registered model `ticket-router`;
3. Streamlit at `http://127.0.0.1:8501`;
4. generated drift HTML under `artifacts/reports/monitoring/simulation/`, after running the
   simulation once if necessary.

Keep one terminal at the repository root. The commands below assume the local example database
names; adapt them to your ignored `.env` without showing credentials.

## Timed narration

### 0–10 seconds — start and orient

Run or show:

```bash
docker compose up -d postgres mlflow migrate api dashboard
docker compose ps
```

Say: “This is a completely local support-ticket routing stack: PostgreSQL, MLflow, FastAPI, and
Streamlit. The API loads a reviewed champion; normal startup never trains a model.”

### 10–23 seconds — MLflow experiments and champion

In MLflow, show the Stage 6 candidate runs and their macro F1 comparison, then open registered model
`ticket-router` and point to version 1 with `candidate` and `champion` aliases.

Say: “Five sparse candidates were tracked with data/configuration hashes, signatures, plots,
latency, size, and per-class metrics. Calibrated word TF-IDF plus LinearSVC won validation macro F1.
The champion alias moved only after test, recall, latency, load, and contract gates passed.”

### 23–39 seconds — route one ticket

On **Route a ticket**, enter synthetic content:

- Subject: `Invoice question`
- Body: `Please explain a duplicate charge on this demonstration account.`

Click **Route ticket**. Point to predicted queue, calibrated confidence, top-three queues, request ID,
model version, latency, and any low-confidence warning.

Say: “Only subject and body enter the shared preprocessing and model pipeline. Post-submission
fields are rejected.”

### 39–50 seconds — submit correction feedback

Choose whether the prediction is correct. For a correction demonstration, select another valid
queue, add `Synthetic reviewer correction`, and submit once.

Say: “Feedback is a delayed label linked to the immutable model version. A second feedback event for
the same request is rejected instead of silently overwriting history.”

### 50–60 seconds — show the privacy-safe prediction log

Run a metadata-only query:

```bash
docker compose exec -T postgres psql -U ticket_router -d ticket_router -c \
  "SELECT request_id, model_version, predicted_queue, confidence, low_confidence, created_at FROM prediction_events ORDER BY created_at DESC LIMIT 1;"
```

Say: “The schema has no raw subject or body columns. It stores model lineage, safe aggregates, and a
one-way fingerprint; normal logging also omits ticket text.”

### 60–73 seconds — plant and detect drift

Run:

```bash
uv run python -m ticket_router.monitoring.simulate_drift
```

Open the generated HTML report and point to longer input lengths, shifted predicted-label
distribution, and reduced confidence.

Say: “Evidently receives explicit derived numeric and categorical features, never raw text. Drift
and delayed-label quality are kept separate.”

### 73–82 seconds — show monitoring outcome

Show the report status and concise JSON/Streamlit monitoring summary.

Say: “Alert status combines multiple signals and can be `insufficient_data`; one small drifted batch
cannot trigger retraining on its own.”

### 82–90 seconds — explain retraining and promotion

Show this command without running a long flow:

```bash
uv run python -m ticket_router.orchestration retraining
```

Say: “Prefect can build a parent-linked dataset from enough human-approved labels, train, gate, and
register a new candidate. It stops there. Champion promotion remains a separate human-approved
command.”

## Claims to avoid

- Do not call local endpoints a public deployment or quote benchmark values from memory.
- Do not imply the synthetic CC BY-NC dataset represents organic production traffic.
- Do not claim drift proves accuracy regression when delayed labels are unavailable.
- Do not run final test evaluation again for the demo.
- Do not show raw `.env`, ticket datasets, MLflow input artifacts, or database fingerprints.
