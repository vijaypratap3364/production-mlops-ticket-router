"""Privacy-safe exact and template duplicate analysis."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from hashlib import sha256

import polars as pl

NORMALIZED_TEXT_HASH_COLUMN = "normalized_text_hash"
TEMPLATE_TEXT_HASH_COLUMN = "template_text_hash"
WHITESPACE_PATTERN = re.compile(r"\s+")
URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+", flags=re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
VARIABLE_TOKEN_PATTERN = re.compile(r"\b(?=\w*[0-9])\w+\b")
WORD_PATTERN = re.compile(r"[A-Za-z]{2,}")
FIELD_MARKERS = frozenset({"subject", "body"})


@dataclass(frozen=True)
class DuplicateAnalysis:
    """Aggregate duplicate results containing no ticket text."""

    total_records: int
    duplicate_record_id_count: int
    unique_normalized_text_hashes: int
    exact_duplicate_groups: int
    exact_duplicate_rows: int
    exact_duplicate_redundant_records: int
    exact_duplicate_rate: float
    contradictory_duplicate_groups: int
    contradictory_duplicate_rows: int
    near_empty_records: int
    near_empty_rate: float
    possible_template_duplicate_groups: int
    possible_template_duplicate_rows: int
    possible_template_duplicate_rate: float
    largest_template_group_sizes: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible aggregate values."""
        return asdict(self)


def canonicalize_text(value: str) -> str:
    """Normalize Unicode, case, and whitespace without changing word content."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def normalized_text_hash(value: str) -> str:
    """Create a stable SHA-256 group key for exact normalized text."""
    return sha256(canonicalize_text(value).encode("utf-8")).hexdigest()


def template_text_hash(value: str) -> str:
    """Create a conservative template key with variable-looking tokens masked."""
    normalized = canonicalize_text(value)
    normalized = URL_PATTERN.sub("<url>", normalized)
    normalized = EMAIL_PATTERN.sub("<email>", normalized)
    normalized = VARIABLE_TOKEN_PATTERN.sub("<variable>", normalized)
    return sha256(normalized.encode("utf-8")).hexdigest()


def add_duplicate_hashes(frame: pl.DataFrame) -> pl.DataFrame:
    """Add deterministic duplicate-group hashes without exposing source text."""
    return frame.with_columns(
        pl.col("text")
        .map_elements(normalized_text_hash, return_dtype=pl.String)
        .alias(NORMALIZED_TEXT_HASH_COLUMN),
        pl.col("text")
        .map_elements(template_text_hash, return_dtype=pl.String)
        .alias(TEMPLATE_TEXT_HASH_COLUMN),
    )


def analyze_duplicates(
    frame: pl.DataFrame,
    *,
    near_empty_word_threshold: int,
    template_min_group_size: int,
) -> DuplicateAnalysis:
    """Measure duplicate, contradiction, near-empty, and template risk."""
    enriched = add_duplicate_hashes(frame)
    exact_groups = enriched.group_by(NORMALIZED_TEXT_HASH_COLUMN).agg(
        pl.len().alias("record_count"),
        pl.col("queue").n_unique().alias("queue_count"),
    )
    duplicate_groups = exact_groups.filter(pl.col("record_count") > 1)
    contradictory_groups = exact_groups.filter(pl.col("queue_count") > 1)

    template_groups = enriched.group_by(TEMPLATE_TEXT_HASH_COLUMN).agg(
        pl.len().alias("record_count"),
        pl.col(NORMALIZED_TEXT_HASH_COLUMN).n_unique().alias("distinct_text_count"),
    )
    possible_templates = template_groups.filter(
        (pl.col("record_count") >= template_min_group_size) & (pl.col("distinct_text_count") > 1)
    )

    semantic_word_count = pl.col("text").map_elements(
        _semantic_word_count,
        return_dtype=pl.Int64,
    )
    near_empty_records = enriched.select(
        (semantic_word_count <= near_empty_word_threshold).sum().alias("count")
    ).item()
    total_records = enriched.height
    exact_duplicate_rows = _sum_column(duplicate_groups, "record_count")
    contradictory_rows = _sum_column(contradictory_groups, "record_count")
    template_rows = _sum_column(possible_templates, "record_count")
    largest_template_groups = tuple(
        possible_templates.sort("record_count", descending=True).head(10)["record_count"].to_list()
    )

    return DuplicateAnalysis(
        total_records=total_records,
        duplicate_record_id_count=total_records - enriched["ticket_record_id"].n_unique(),
        unique_normalized_text_hashes=exact_groups.height,
        exact_duplicate_groups=duplicate_groups.height,
        exact_duplicate_rows=exact_duplicate_rows,
        exact_duplicate_redundant_records=exact_duplicate_rows - duplicate_groups.height,
        exact_duplicate_rate=_rate(exact_duplicate_rows, total_records),
        contradictory_duplicate_groups=contradictory_groups.height,
        contradictory_duplicate_rows=contradictory_rows,
        near_empty_records=int(near_empty_records),
        near_empty_rate=_rate(int(near_empty_records), total_records),
        possible_template_duplicate_groups=possible_templates.height,
        possible_template_duplicate_rows=template_rows,
        possible_template_duplicate_rate=_rate(template_rows, total_records),
        largest_template_group_sizes=largest_template_groups,
    )


def contradictory_duplicate_groups(frame: pl.DataFrame) -> pl.DataFrame:
    """Return hash-level contradictory-label evidence without raw ticket text."""
    enriched = add_duplicate_hashes(frame)
    return (
        enriched.group_by(NORMALIZED_TEXT_HASH_COLUMN)
        .agg(
            pl.len().alias("record_count"),
            pl.col("queue").n_unique().alias("queue_count"),
            pl.col("queue").unique().sort().alias("queues"),
        )
        .filter(pl.col("queue_count") > 1)
        .sort(NORMALIZED_TEXT_HASH_COLUMN)
    )


def _semantic_word_count(value: str) -> int:
    words = (word.casefold() for word in WORD_PATTERN.findall(value))
    return sum(word not in FIELD_MARKERS for word in words)


def _sum_column(frame: pl.DataFrame, column: str) -> int:
    if frame.is_empty():
        return 0
    return int(frame[column].sum())


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
