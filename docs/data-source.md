# Dataset source and reproducible ingestion

## Source and attribution

This project uses the public Hugging Face dataset
[`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets),
created by Tobi Bueck / Softoft and identified by DOI
[`10.57967/hf/6184`](https://doi.org/10.57967/hf/6184).

The configured immutable revision is recorded in `configs/base.yaml`. Ingestion rejects mutable
revision names such as `main`, records the resolved Hub revision when available, and hashes every
downloaded source file. The repository currently contains multiple CSV variants; Stage 2 treats all
CSV files in the pinned snapshot as raw source files and records each one independently in the raw
manifest. Later dataset-definition work must evaluate version overlap and duplicates before splitting.

## License notice

The upstream dataset card marks the data as
[`CC BY-NC 4.0`](https://creativecommons.org/licenses/by-nc/4.0/). This permits attributed
noncommercial use subject to the license terms. The dataset is not redistributed in this GitHub
repository. Downloaded CSV files, normalized Parquet data, and generated manifests remain ignored
local artifacts. The project's MIT source-code license does not relicense the dataset.

Users are responsible for confirming that their use is noncommercial and otherwise compatible with
the upstream license.

## Selected normalized fields

The normalized Parquet artifact contains:

- `ticket_record_id`: deterministic SHA-256 identifier incorporating immutable source lineage;
- `source_row_id`, `source_file`, and `source_row_number`: traceability to the raw snapshot;
- `language`: canonical selected language (`en`);
- `subject` and `body`: separately preserved predictive source fields;
- `text`: field-marked combination derived only from `subject` and `body`;
- `queue`: target label, retained for later dataset definition and training but never a predictor.

Rows are retained only when their language is English, `queue` is present, and at least one of
`subject` or `body` contains usable text. Common explicit null markers such as blank strings, `null`,
`none`, `nan`, `n/a`, and `<na>` are normalized to null.

Rows containing more CSV fields than the declared header are structurally malformed: their values
cannot be assigned to columns reliably, so normalization drops them before language or content
filtering. The normalization manifest records this count separately as
`rows_dropped_malformed`.

## Leakage-prone fields

Raw files are immutable, but the normalized Parquet artifact deliberately excludes post-submission
or agent-generated columns such as:

- `answer` and response fields;
- `priority`;
- ticket `type`;
- tags and tag columns;
- agent-generated or assigned fields;
- resolution fields.

The target `queue` remains in normalized data but is explicitly forbidden by the modeling feature
contract. Modeling adapters must call `validate_model_feature_frame` or
`validate_model_feature_columns`; only `subject`, `body`, and `text` are accepted.

## Reproduction commands

From the repository root, install the locked environment and download the configured snapshot:

```text
uv sync --locked --all-groups
uv run python -m ticket_router.data.download
uv run python -m ticket_router.data.normalize
```

Equivalent Make aliases are available:

```text
make download-data
make normalize-data
```

The downloader reuses an existing verified cache without contacting the network. It refuses to
replace a partial, hash-mismatched, or differently configured snapshot. After reviewing the reason,
explicit replacement is available with:

```text
uv run python -m ticket_router.data.download --force
uv run python -m ticket_router.data.normalize --force
```

Generated local files:

```text
data/raw/source/...
data/raw/data_manifest.json
data/interim/normalized_tickets.parquet
data/interim/normalization_manifest.json
```

The raw manifest records source/revision/license, UTC retrieval time, row and column metadata, file
paths, file sizes and SHA-256 hashes, configuration hash, and Git code identity when available. The
normalization manifest links the raw-manifest/input hashes to the output Parquet hash and records all
row-filter counts. These artifacts are reproducible but intentionally not committed.

## Dataset limitations

- The dataset is described as synthetic and may contain repetitive templates, unrealistic wording,
  inconsistent labels, or artifacts from its generation process.
- Multiple upstream CSV versions may overlap. Raw row counts are not automatically equivalent to
  unique real-world tickets.
- Offline performance on this data may overstate performance on organic support tickets.
- Language labels can be noisy; Stage 2 uses the provided language field and does not claim
  independent language detection.
- Queue names and frequencies describe the source dataset, not a universal production taxonomy.
- Stage 3 adds exact/template duplicate analysis, deterministic class selection, and aggregate EDA.
  Leakage-safe grouped train/validation/test splitting remains a later stage.
