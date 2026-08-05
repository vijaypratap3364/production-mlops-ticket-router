"""Reproducible, privacy-safe EDA report construction and rendering."""

from __future__ import annotations

import html
import json
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from uuid import uuid4

import polars as pl
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # type: ignore[import-untyped]

from ticket_router.data.class_selection import ClassSelectionReport
from ticket_router.data.duplicates import DuplicateAnalysis
from ticket_router.data.manifests import NormalizationManifest, RawDataManifest, atomic_write_json
from ticket_router.data.validation import ValidationSummary
from ticket_router.features.contracts import find_forbidden_feature_columns

TOKEN_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z'-]{2,24}\b")
EMAIL_PATTERN = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+", flags=re.IGNORECASE)
TOKEN_EXCLUSIONS = ENGLISH_STOP_WORDS | {
    "body",
    "subject",
    "hello",
    "dear",
    "regards",
    "thank",
    "thanks",
}


def build_eda_report(
    *,
    frame: pl.DataFrame,
    selected_frame: pl.DataFrame,
    validation: ValidationSummary,
    duplicates: DuplicateAnalysis,
    class_selection: ClassSelectionReport,
    normalization_manifest: NormalizationManifest,
    raw_manifest: RawDataManifest,
    common_tokens_per_class: int,
    token_min_document_frequency: int,
) -> dict[str, object]:
    """Build aggregate-only EDA content with complete source lineage."""
    missing_values = {
        column: int(count) for column, count in frame.null_count().to_dicts()[0].items()
    }
    leakage_columns = find_forbidden_feature_columns(raw_manifest.column_names)
    upstream_removed = (
        normalization_manifest.rows_dropped_malformed
        + normalization_manifest.rows_dropped_missing_queue
        + normalization_manifest.rows_dropped_missing_text
    )

    return {
        "report_version": 1,
        "source": {
            "repository": raw_manifest.source_repository,
            "requested_revision": raw_manifest.requested_revision,
            "resolved_revision": raw_manifest.resolved_revision,
            "dataset_license": raw_manifest.dataset_license,
            "normalized_file_sha256": normalization_manifest.output_file_sha256,
            "normalization_configuration_hash": normalization_manifest.configuration_hash,
            "analysis_configuration_hash": class_selection.configuration_hash,
        },
        "record_counts": {
            "raw_rows": raw_manifest.row_count,
            "normalization_rows_read": normalization_manifest.rows_read,
            "normalized_english_records": frame.height,
            "validated_records": validation.valid_records,
            "selected_class_records": selected_frame.height,
            "records_removed_during_stage3_validation": validation.records_removed,
            "records_removed_during_normalization": upstream_removed,
            "normalization_drops": {
                "malformed": normalization_manifest.rows_dropped_malformed,
                "missing_queue": normalization_manifest.rows_dropped_missing_queue,
                "missing_text": normalization_manifest.rows_dropped_missing_text,
            },
        },
        "missing_values": missing_values,
        "duplicates": duplicates.to_dict(),
        "class_distribution": {
            "original_counts": class_selection.original_class_counts,
            "selected_counts": class_selection.selected_class_counts,
            "selected_proportions": class_selection.class_proportions,
            "imbalance_ratio_largest_to_smallest": class_selection.imbalance_ratio,
        },
        "length_distributions": {
            "subject_characters": _length_distribution(frame, "subject"),
            "body_characters": _length_distribution(frame, "body"),
            "combined_text_characters": _length_distribution(frame, "text"),
            "combined_text_words": _word_count_distribution(frame),
        },
        "common_non_sensitive_tokens_by_class": _common_tokens_by_class(
            selected_frame,
            top_n=common_tokens_per_class,
            minimum_document_frequency=token_min_document_frequency,
        ),
        "leakage_audit": {
            "raw_source_columns": list(raw_manifest.column_names),
            "potential_leakage_columns_found": list(leakage_columns),
            "allowed_model_inputs": ["subject", "body", "text"],
            "note": (
                "Token statistics are aggregate document frequencies only; no ticket examples "
                "or record identifiers are included."
            ),
        },
    }


def write_eda_artifacts(
    report: Mapping[str, object],
    *,
    json_path: Path,
    html_path: Path,
) -> None:
    """Write matching deterministic JSON and standalone HTML reports."""
    atomic_write_json(json_path, dict(report))
    _atomic_write_text(html_path, render_eda_html(report))


def render_eda_html(report: Mapping[str, object]) -> str:
    """Render a standalone aggregate report without external assets or scripts."""
    record_counts = _mapping(report["record_counts"])
    normalization_drops = _mapping(record_counts["normalization_drops"])
    missing_values = _mapping(report["missing_values"])
    duplicates = _mapping(report["duplicates"])
    class_distribution = _mapping(report["class_distribution"])
    selected_counts = _mapping(class_distribution["selected_counts"])
    selected_proportions = _mapping(class_distribution["selected_proportions"])
    lengths = _mapping(report["length_distributions"])
    tokens = _mapping(report["common_non_sensitive_tokens_by_class"])
    leakage = _mapping(report["leakage_audit"])

    class_rows = [
        (
            label,
            selected_counts[label],
            f"{_as_float(selected_proportions[label]):.4%}",
        )
        for label in selected_counts
    ]
    length_rows = [(name, *_mapping(values).values()) for name, values in lengths.items()]
    token_rows = [
        (label, ", ".join(str(item["token"]) for item in _list_of_mappings(items)))
        for label, items in tokens.items()
    ]

    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ticket Router Data Analysis</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1100px; color: #172033; }}
    h1, h2 {{ color: #173b57; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
    th, td {{ border: 1px solid #d8e0e8; padding: .55rem; text-align: left; }}
    th {{ background: #eef4f8; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; }}
    .card {{ background: #f5f8fa; border-radius: .5rem; padding: 1rem; }}
    .card strong {{ display: block; font-size: 1.35rem; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>Customer-Support Ticket Data Analysis</h1>
  <p>Reproducible aggregate report. It contains no raw ticket examples or record identifiers.</p>
  <h2>Record flow</h2>
  <div class="grid">{record_cards}</div>
  <h2>Missing values</h2>
  {missing_values_table}
  <h2>Records removed before analysis</h2>
  {normalization_drops_table}
  <h2>Duplicate and quality signals</h2>
  {duplicate_table}
  <h2>Selected class distribution</h2>
  <p>Largest-to-smallest imbalance ratio: <strong>{imbalance_ratio}</strong></p>
  {class_table}
  <h2>Length distributions</h2>
  {length_table}
  <h2>Frequent aggregate tokens by class</h2>
  {token_table}
  <h2>Leakage audit</h2>
  <p>Potential leakage columns present upstream: <code>{leakage_columns}</code></p>
</body>
</html>
""".format(
        record_cards="".join(
            f'<div class="card"><span>{_escape(key)}</span><strong>{_escape(value)}</strong></div>'
            for key, value in record_counts.items()
            if not isinstance(value, Mapping)
        ),
        duplicate_table=_html_table(
            ("Metric", "Value"),
            ((key, value) for key, value in duplicates.items()),
        ),
        missing_values_table=_html_table(
            ("Column", "Missing records"),
            ((key, value) for key, value in missing_values.items()),
        ),
        normalization_drops_table=_html_table(
            ("Reason", "Records removed"),
            ((key, value) for key, value in normalization_drops.items()),
        ),
        imbalance_ratio=_escape(class_distribution["imbalance_ratio_largest_to_smallest"]),
        class_table=_html_table(("Queue", "Count", "Proportion"), class_rows),
        length_table=_html_table(
            ("Field", "Count", "Min", "Max", "Mean", "P50", "P90", "P95", "P99"),
            length_rows,
        ),
        token_table=_html_table(("Queue", "Top aggregate tokens"), token_rows),
        leakage_columns=", ".join(
            str(value) for value in _list(leakage["potential_leakage_columns_found"])
        ),
    )


def _length_distribution(frame: pl.DataFrame, column: str) -> dict[str, int | float]:
    values = frame.select(pl.col(column).str.len_chars().drop_nulls().alias("length"))["length"]
    return _numeric_distribution(values)


def _word_count_distribution(frame: pl.DataFrame) -> dict[str, int | float]:
    values = frame.select(
        pl.col("text")
        .map_elements(lambda value: len(TOKEN_PATTERN.findall(value)), return_dtype=pl.Int64)
        .alias("word_count")
    )["word_count"]
    return _numeric_distribution(values)


def _numeric_distribution(values: pl.Series) -> dict[str, int | float]:
    if values.is_empty():
        return {key: 0 for key in ("count", "min", "max", "mean", "p50", "p90", "p95", "p99")}
    return {
        "count": len(values),
        "min": _as_int(values.min()),
        "max": _as_int(values.max()),
        "mean": _as_float(values.mean()),
        "p50": _as_float(values.quantile(0.50, interpolation="linear")),
        "p90": _as_float(values.quantile(0.90, interpolation="linear")),
        "p95": _as_float(values.quantile(0.95, interpolation="linear")),
        "p99": _as_float(values.quantile(0.99, interpolation="linear")),
    }


def _common_tokens_by_class(
    frame: pl.DataFrame,
    *,
    top_n: int,
    minimum_document_frequency: int,
) -> dict[str, list[dict[str, int | float | str]]]:
    counters: dict[str, Counter[str]] = {}
    class_sizes: Counter[str] = Counter()
    for queue, text in frame.select("queue", "text").iter_rows():
        label = str(queue)
        class_sizes[label] += 1
        counter = counters.setdefault(label, Counter())
        counter.update(_privacy_filtered_tokens(str(text)))

    result: dict[str, list[dict[str, int | float | str]]] = {}
    for label in sorted(counters, key=str.casefold):
        eligible = [
            (token, count)
            for token, count in counters[label].items()
            if count >= minimum_document_frequency
        ]
        eligible.sort(key=lambda item: (-item[1], item[0]))
        result[label] = [
            {
                "token": token,
                "document_count": count,
                "document_rate": count / class_sizes[label],
            }
            for token, count in eligible[:top_n]
        ]
    return result


def _privacy_filtered_tokens(value: str) -> set[str]:
    redacted = EMAIL_PATTERN.sub(" ", URL_PATTERN.sub(" ", value))
    return {
        token
        for token in (match.casefold() for match in TOKEN_PATTERN.findall(redacted))
        if token not in TOKEN_EXCLUSIONS
    }


def _html_table(headers: tuple[str, ...], rows: Iterable[Iterable[object]]) -> str:
    header_html = "".join(f"<th>{_escape(value)}</th>" for value in headers)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{_escape(value)}</td>" for value in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>"


def _escape(value: object) -> str:
    if isinstance(value, (dict, list, tuple)):
        return html.escape(json.dumps(value, sort_keys=True))
    return html.escape(str(value))


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("Expected report mapping")
    return {str(key): item for key, item in value.items()}


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError("Expected report list")
    return value


def _list_of_mappings(value: object) -> list[Mapping[str, object]]:
    return [_mapping(item) for item in _list(value)]


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Expected numeric report value, received {type(value).__name__}")


def _as_int(value: object) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    raise TypeError(f"Expected numeric report value, received {type(value).__name__}")


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary_path.write_text(value, encoding="utf-8", newline="\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
