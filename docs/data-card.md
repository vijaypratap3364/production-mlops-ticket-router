# Data card: customer-support ticket routing dataset

## Source and license

This project retrieves
[`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)
from Hugging Face at the immutable revision recorded in `configs/base.yaml`. The dataset is
attributed to Tobi Bueck / Softoft and has DOI
[`10.57967/hf/6184`](https://doi.org/10.57967/hf/6184). Its dataset card identifies the license as
CC BY-NC 4.0. The dataset is not redistributed in this repository; its noncommercial terms remain
separate from the project's eventual source-code license.

## Synthetic nature

The tickets are synthetic. They may contain repetitive templates, artificial wording, duplicated
versions, simplified queue boundaries, and generation artifacts that differ materially from organic
support traffic. Results on this dataset must not be presented as evidence of real-world production
performance.

## Intended use

The data supports a local, educational demonstration of reproducible text classification, leakage
prevention, experiment tracking, model serving, feedback, and monitoring. The predictive inputs are
limited to the ticket `subject`, `body`, and a marked `text` combination derived only from those two
fields. The selected target is `queue`.

## Inappropriate uses

- Commercial use that is incompatible with the dataset's CC BY-NC 4.0 terms.
- Safety-critical or fully autonomous routing without human review and real-domain validation.
- Evaluation claims about real customers, languages, organizations, fairness, or production quality.
- Training with agent answers, assigned priority, tags, ticket type, resolution, or other
  post-submission fields.
- Publishing or logging individual ticket text, even though the current source is synthetic.

## Processing and validation

1. Download the configured full Hugging Face revision and verify every source-file SHA-256 hash.
2. Normalize source columns, retain English rows, normalize null markers, and require `queue` plus
   usable `subject` or `body` text.
3. Derive `text` only from marked `subject` and `body` fields and create source-aware record IDs.
4. Validate the normalized artifact with a strict Pandera contract covering schema, ID format and
   uniqueness, English language, nonempty target/text, source-to-combined-text consistency, and
   configured technical length bounds.
5. Compute normalized SHA-256 text-group hashes and conservative template hashes. Exact groups
   remain intact across train, validation, and test.
6. Rank queue labels by descending observed count with a case-insensitive label tie-break. Select the
   configured top ten only after minimum-count and split-feasibility checks.
7. Create stateless `model_text` with NFKC and whitespace normalization plus email, URL, and
   phone-like masking. No stemming, stop-word removal, or learned vocabulary is applied.
8. Exclude contradictory exact-text groups and allocate remaining groups within each queue using
   seed 42 and 70/15/15 targets.

The generated `artifacts/reports/selected_classes.json` is the authoritative label mapping. Labels
are never manually embedded in source code.

## Measured Stage 3 profile

The reproducible Stage 3 run over normalized SHA-256
`abac81257c3e21b462217ba02455b11e4bc7427b13e0adc5c0f6a3cc946751e7` validated 28,190
English records and removed no additional records. All ten observed queues exceeded the configured
minimum of 100 records and the calculated stratified-split feasibility minimum of seven records.

| Class ID | Queue | Records | Proportion |
|---:|---|---:|---:|
| 0 | Technical Support | 8,135 | 28.8578% |
| 1 | Product Support | 5,288 | 18.7584% |
| 2 | Customer Service | 4,257 | 15.1011% |
| 3 | IT Support | 3,326 | 11.7985% |
| 4 | Billing and Payments | 2,892 | 10.2590% |
| 5 | Returns and Exchanges | 1,400 | 4.9663% |
| 6 | Service Outages and Maintenance | 1,102 | 3.9092% |
| 7 | Sales and Pre-Sales | 839 | 2.9762% |
| 8 | Human Resources | 552 | 1.9581% |
| 9 | General Inquiry | 399 | 1.4154% |

The largest-to-smallest class ratio is 20.3885. Exact normalized-text analysis found 4,495
duplicate groups containing 8,990 records, equivalent to 4,495 records beyond one representative
per group. No same-text/different-queue contradictions were found. Six records met the configured
near-empty threshold, and the conservative variable-token template rule found no additional
non-exact template groups of at least five records. These are data-quality measurements, not model
performance metrics.

## Measured Stage 4 split

No contradictory exact-text group was present, so all 28,190 selected records remained usable. The
grouped-stratified allocator produced:

| Split | Records | Actual percentage | Target percentage |
|---|---:|---:|---:|
| Train | 19,729 | 69.9858% | 70% |
| Validation | 4,232 | 15.0124% | 15% |
| Test | 4,229 | 15.0018% | 15% |

Every selected queue appears in every split. The largest absolute class-proportion deviation was
0.01331 percentage points, for `Service Outages and Maintenance` in validation, well inside the
configured 1-percentage-point tolerance. Record-ID and normalized-text-hash intersections between
splits are empty. These are preparation integrity measurements, not model-performance results.

The test split is sealed from routine training loaders. It may be opened only through the explicit
final-evaluation path after candidate selection and configuration are frozen.

## Leakage exclusions

The general feature contract accepts `subject`, `body`, derived `text`, and derived `model_text`.
Prepared modeling utilities expose exactly `model_text` and return `queue` separately as the target.
The target `queue` and upstream `answer`, response, priority, type, tag, assigned-agent, and
resolution fields are rejected by the modeling boundary. EDA may audit column names and aggregate
target counts but never passes those fields into a predictor.

## Known limitations

- The pinned repository contains multiple CSV variants that may overlap.
- Exact and template duplication can inflate naive random-split results.
- Provided language labels are trusted rather than independently detected.
- Queue frequency is imbalanced and represents this synthetic source, not a universal taxonomy.
- Aggregate token frequencies may reflect generator templates rather than organic user language.
- Near-duplicate detection is deliberately conservative and is not semantic similarity detection.
- The final test split remains sealed. Stage 5 reports validation-only baseline performance; no
  final-test performance result exists yet.

## Privacy considerations

Downloaded data, normalized Parquet, and generated reports remain ignored local artifacts. Reports
contain aggregate statistics, hashes, and class labels but no ticket examples or record identifiers.
Common-token tables use document-frequency aggregation, suppress low-frequency tokens, exclude URLs
and email-like strings, and contain no surrounding text. Production prediction logs will not store
raw ticket content by default.

## Reproducibility

From the repository root:

```text
uv sync --locked --all-groups
uv run python -m ticket_router.data.download
uv run python -m ticket_router.data.normalize
uv run python -m ticket_router.data.analyze
uv run python -m ticket_router.data.prepare
uv run python -m ticket_router.modeling.train_baseline
```

The analysis command verifies normalized-data lineage and regenerates:

```text
artifacts/reports/selected_classes.json
artifacts/reports/duplicate_analysis.json
artifacts/reports/eda_report.json
artifacts/reports/eda_report.html
artifacts/reports/split_summary.json
data/processed/train.parquet
data/processed/validation.parquet
data/processed/test.parquet
data/processed/split_manifest.json
data/reference/training_reference.parquet
```

The exact measured profile and selected labels must be read from those generated artifacts. They are
not committed because reports derive from an uncommitted dataset.
