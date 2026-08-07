"""Build the privacy-safe monitoring reference from final training data."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from ticket_router.api.model_loader import load_champion
from ticket_router.api.service import LoadedChampion
from ticket_router.config import Settings
from ticket_router.data.manifests import GitVersion, atomic_write_json, get_git_version
from ticket_router.hashing import sha256_file, sha256_json
from ticket_router.logging_config import configure_logging, get_logger
from ticket_router.monitoring.config import (
    DEFAULT_MONITORING_CONFIG_PATH,
    MonitoringConfig,
)
from ticket_router.monitoring.features import FEATURE_DEFINITIONS, derive_text_monitoring_features
from ticket_router.registry.service import ModelRegistryService


class ReferencePredictor(Protocol):
    classes_: object

    def predict(self, values: Sequence[str]) -> object: ...

    def predict_proba(self, values: Sequence[str]) -> object: ...


class MonitoringReferenceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_version: int = 1
    creation_timestamp_utc: str
    champion_model_name: str
    champion_model_version: str
    champion_alias: str
    row_count: int = Field(gt=0)
    columns: tuple[str, ...]
    feature_definitions: dict[str, str]
    source_file_sha256: dict[str, str]
    split_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    monitoring_configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confidence_warning_threshold: float = Field(ge=0.0, le=1.0)
    champion_baseline_macro_f1: float = Field(ge=0.0, le=1.0)
    code_version: GitVersion | None

    @classmethod
    def read(cls, path: Path) -> MonitoringReferenceManifest:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def build_monitoring_reference(
    *,
    processed_dir: Path,
    output_path: Path,
    manifest_path: Path,
    split_manifest_path: Path,
    champion: LoadedChampion,
    confidence_warning_threshold: float,
    champion_baseline_macro_f1: float,
    monitoring_config: MonitoringConfig,
    project_root: Path,
    clock: datetime | None = None,
) -> MonitoringReferenceManifest:
    """Score train+validation and retain only aggregate monitoring values."""
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "monitoring reference already exists; remove both generated files before rebuilding"
        )
    source_paths = [
        processed_dir / f"{split}.parquet" for split in monitoring_config.reference.source_splits
    ]
    for source_path in source_paths:
        if not source_path.is_file():
            raise FileNotFoundError(f"final training split is unavailable: {source_path}")
    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"split manifest is unavailable: {split_manifest_path}")
    frames = [
        pl.read_parquet(path).select("subject", "body", "model_text") for path in source_paths
    ]
    training = pl.concat(frames, how="vertical")
    rows = _reference_rows(
        training,
        champion=champion,
        warning_threshold=confidence_warning_threshold,
        batch_size=monitoring_config.reference.inference_batch_size,
    )
    reference = pl.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference.write_parquet(output_path)
    created_at = clock or datetime.now(UTC)
    manifest = MonitoringReferenceManifest(
        creation_timestamp_utc=created_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        champion_model_name=champion.model_name,
        champion_model_version=champion.model_version,
        champion_alias=champion.alias,
        row_count=reference.height,
        columns=tuple(reference.columns),
        feature_definitions=FEATURE_DEFINITIONS,
        source_file_sha256={path.as_posix(): sha256_file(path) for path in source_paths},
        split_manifest_sha256=sha256_file(split_manifest_path),
        reference_data_sha256=sha256_file(output_path),
        monitoring_configuration_sha256=sha256_json(monitoring_config.model_dump(mode="json")),
        confidence_warning_threshold=confidence_warning_threshold,
        champion_baseline_macro_f1=champion_baseline_macro_f1,
        code_version=get_git_version(project_root),
    )
    atomic_write_json(manifest_path, manifest.model_dump(mode="json"))
    return manifest


def _reference_rows(
    training: pl.DataFrame,
    *,
    champion: LoadedChampion,
    warning_threshold: float,
    batch_size: int,
) -> list[dict[str, str | int | float | bool]]:
    model: ReferencePredictor = champion.model
    labels = tuple(str(value) for value in cast(NDArray[np.object_], np.asarray(model.classes_)))
    records = training.to_dicts()
    result: list[dict[str, str | int | float | bool]] = []
    for offset in range(0, len(records), batch_size):
        batch = records[offset : offset + batch_size]
        model_texts = [str(row["model_text"]) for row in batch]
        predictions = np.asarray(model.predict(model_texts), dtype=object)
        probabilities = np.asarray(model.predict_proba(model_texts), dtype=np.float64)
        if predictions.shape != (len(batch),) or probabilities.shape != (len(batch), len(labels)):
            raise ValueError("champion returned an invalid monitoring-reference prediction shape")
        for index, row in enumerate(batch):
            predicted_queue = str(predictions[index])
            if predicted_queue not in labels:
                raise ValueError("champion returned an unknown queue while building reference")
            confidence = float(probabilities[index, labels.index(predicted_queue)])
            feature_values = derive_text_monitoring_features(
                subject=str(row["subject"] or ""),
                body=str(row["body"] or ""),
                model_text=str(row["model_text"]),
            ).to_dict()
            result.append(
                {
                    **feature_values,
                    "predicted_queue": predicted_queue,
                    "prediction_confidence": confidence,
                    "low_confidence": confidence < warning_threshold,
                    "model_version": champion.model_version,
                }
            )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--monitoring-config", type=Path, default=DEFAULT_MONITORING_CONFIG_PATH)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--split-manifest", type=Path, default=Path("data/processed/split_manifest.json")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = Settings.load(args.config)
    config = MonitoringConfig.load(args.monitoring_config)
    configure_logging(settings.log_level)
    try:
        champion = load_champion(settings)
        registry = ModelRegistryService()
        tags = registry.model_version_tags(
            name=champion.model_name,
            version=champion.model_version,
        )
        if "test_macro_f1" not in tags:
            raise ValueError(
                "champion registry version lacks the Stage 7 test_macro_f1 lineage tag"
            )
        manifest = build_monitoring_reference(
            processed_dir=args.processed_dir,
            output_path=config.reference.output_path,
            manifest_path=config.reference.manifest_path,
            split_manifest_path=args.split_manifest,
            champion=champion,
            confidence_warning_threshold=settings.api_confidence_warning_threshold,
            champion_baseline_macro_f1=float(tags["test_macro_f1"]),
            monitoring_config=config,
            project_root=Path.cwd(),
        )
    except (ConnectionError, FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        get_logger(__name__).error("monitoring_reference_failed", error=str(exc))
        return 1
    print(json.dumps(manifest.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
