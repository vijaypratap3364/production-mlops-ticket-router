"""Typed lineage manifest for deterministic prepared data splits."""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from ticket_router.config import SplitRatios, SplittingSettings, TextPreprocessingSettings
from ticket_router.data.manifests import GitVersion, atomic_write_json


class OutputFileManifest(BaseModel):
    """Integrity metadata for one generated Parquet artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(ge=0)


class SplitManifest(BaseModel):
    """Complete preprocessing, duplicate, split, and output lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_version: int = 1
    preparation_timestamp_utc: str
    data_source_hashes: dict[str, str]
    configuration_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    preprocessing: TextPreprocessingSettings
    splitting: SplittingSettings
    split_ratios: SplitRatios
    random_seed: int = Field(ge=0)
    duplicate_group_column: str
    selected_input_rows: int = Field(ge=0)
    contradictory_group_count: int = Field(ge=0)
    contradictory_rows_excluded: int = Field(ge=0)
    final_row_count: int = Field(ge=0)
    split_counts: dict[str, int]
    split_percentages: dict[str, float]
    per_class_counts: dict[str, dict[str, int]]
    maximum_class_proportion_deviation: dict[str, float]
    most_deviant_class: dict[str, str]
    label_mapping: dict[str, int]
    model_feature_columns: tuple[str, ...]
    target_column: str
    output_files: dict[str, OutputFileManifest]
    preparation_code_version: GitVersion | None

    @classmethod
    def read(cls, path: Path) -> Self:
        """Load and validate a split manifest."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def write(self, path: Path) -> None:
        """Write the manifest atomically."""
        atomic_write_json(path, self.model_dump(mode="json"))
