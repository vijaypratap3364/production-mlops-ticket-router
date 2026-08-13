"""Attest a release commit against the immutable champion without test reevaluation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow.models import get_model_info

from ticket_router.config import Settings
from ticket_router.data.manifests import atomic_write_json, get_git_version
from ticket_router.data.split_manifest import SplitManifest
from ticket_router.hashing import sha256_file, sha256_json
from ticket_router.logging_config import configure_logging, get_logger
from ticket_router.registry.config import DEFAULT_FINAL_MODEL_CONFIG_PATH, FinalModelConfig
from ticket_router.registry.contracts import (
    prediction_contract_passes,
    signature_matches_text_api,
)
from ticket_router.registry.service import ModelRegistryService

RELEASE_PATTERN = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+$")
REQUIRED_LINEAGE_TAGS = (
    "combined_training_data_sha256",
    "configuration_sha256",
    "split_manifest_sha256",
    "test_data_sha256",
)


class ReleaseAttestationError(RuntimeError):
    """Raised when the champion cannot be tied safely to a release commit."""


def attest_release(
    *,
    release: str,
    settings: Settings,
    final_config: FinalModelConfig,
    split_manifest_path: Path,
    output_path: Path,
    project_root: Path,
    registry: ModelRegistryService | None = None,
    model_info_loader: Callable[[str], Any] = get_model_info,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, object]:
    """Validate champion lineage/contract and record the clean release source commit."""
    if RELEASE_PATTERN.fullmatch(release) is None:
        raise ReleaseAttestationError("release must use the form vMAJOR.MINOR.PATCH")
    source = get_git_version(project_root.resolve())
    if source is None:
        raise ReleaseAttestationError("release source Git commit is unavailable")
    if source.dirty:
        raise ReleaseAttestationError("release attestation requires a clean Git worktree")

    split_manifest_path = split_manifest_path.resolve()
    manifest = SplitManifest.read(split_manifest_path)
    split_manifest_hash = sha256_file(split_manifest_path)
    _verify_split_outputs(manifest, project_root.resolve())

    mlflow.set_tracking_uri(settings.effective_mlflow_tracking_uri)
    mlflow.set_registry_uri(settings.effective_mlflow_tracking_uri)
    service = registry or ModelRegistryService()
    name = final_config.registry.model_name
    champion = service.resolve_alias(name=name, alias=final_config.registry.champion_alias)
    if champion is None:
        raise ReleaseAttestationError("champion alias is unavailable")
    tags = service.model_version_tags(name=name, version=champion.version)
    missing = [key for key in REQUIRED_LINEAGE_TAGS if not tags.get(key)]
    if missing:
        raise ReleaseAttestationError(
            "champion is missing required lineage tags: " + ", ".join(missing)
        )
    if tags["split_manifest_sha256"] != split_manifest_hash:
        raise ReleaseAttestationError("champion split-manifest hash does not match local lineage")
    if tags["test_data_sha256"] != manifest.output_files["test"].sha256:
        raise ReleaseAttestationError("champion test-data hash does not match local lineage")
    if tags.get("promotion_gates_passed") != "true":
        raise ReleaseAttestationError("champion does not record successful promotion gates")

    labels = tuple(
        label for label, _ in sorted(manifest.label_mapping.items(), key=lambda item: item[1])
    )
    model = service.load_version(name=name, version=champion.version)
    contract_passed = prediction_contract_passes(model, labels=labels)
    signature = model_info_loader(f"models:/{name}/{champion.version}").signature
    signature_passed = signature_matches_text_api(
        signature,
        required_input_dtype=final_config.promotion.required_input_dtype,
        required_output_dtype=final_config.promotion.required_output_dtype,
    )
    if not contract_passed or not signature_passed:
        raise ReleaseAttestationError("champion prediction or signature contract failed")

    timestamp = (clock or (lambda: datetime.now(UTC)))()
    if timestamp.tzinfo is None:
        raise ValueError("release attestation timestamp must be timezone-aware")
    payload: dict[str, object] = {
        "schema_version": 1,
        "release": release,
        "attested_at_utc": timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "release_git_commit": source.commit,
        "release_git_dirty": source.dirty,
        "registered_model_name": name,
        "champion_alias": final_config.registry.champion_alias,
        "champion_version": champion.version,
        "champion_run_id": champion.run_id,
        "historical_training_git_commit": tags.get("git_commit", "unavailable"),
        "split_manifest_sha256": split_manifest_hash,
        "combined_training_data_sha256": tags["combined_training_data_sha256"],
        "test_data_sha256": tags["test_data_sha256"],
        "configuration_sha256": tags["configuration_sha256"],
        "prediction_contract_passed": contract_passed,
        "signature_compatible": signature_passed,
        "promotion_gates_passed": True,
        "alias_changed": False,
        "test_reevaluated": False,
    }
    attestation_hash = sha256_json(payload)
    document = {**payload, "attestation_payload_sha256": attestation_hash}
    atomic_write_json(output_path, document)
    service.set_model_version_tags(
        name=name,
        version=champion.version,
        tags={
            "release": release,
            "release_git_commit": source.commit,
            "release_attested_at_utc": str(payload["attested_at_utc"]),
            "release_attestation_sha256": attestation_hash,
            "release_contract_passed": "true",
            "release_test_reevaluated": "false",
        },
    )
    return document


def _verify_split_outputs(manifest: SplitManifest, project_root: Path) -> None:
    for output in manifest.output_files.values():
        path = Path(output.path)
        resolved = path if path.is_absolute() else project_root / path
        if not resolved.is_file() or sha256_file(resolved) != output.sha256:
            raise ReleaseAttestationError(f"split output fails lineage verification: {resolved}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--final-config", type=Path, default=DEFAULT_FINAL_MODEL_CONFIG_PATH)
    parser.add_argument(
        "--split-manifest", type=Path, default=Path("data/processed/split_manifest.json")
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = Settings.load(args.config)
    configure_logging(settings.log_level)
    output = args.output or Path("artifacts/reports/releases") / f"{args.release}.json"
    try:
        result = attest_release(
            release=args.release,
            settings=settings,
            final_config=FinalModelConfig.load(args.final_config),
            split_manifest_path=args.split_manifest,
            output_path=output,
            project_root=Path.cwd(),
        )
    except (FileNotFoundError, OSError, ReleaseAttestationError, RuntimeError, ValueError) as exc:
        get_logger(__name__).error("release_attestation_failed", error=str(exc))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
