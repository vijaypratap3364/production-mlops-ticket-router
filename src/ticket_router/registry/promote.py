"""Explicit human-triggered candidate-to-champion MLflow alias promotion."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from mlflow.models import get_model_info

from ticket_router.config import Settings
from ticket_router.data.manifests import atomic_write_json, get_git_version
from ticket_router.data.split_manifest import SplitManifest
from ticket_router.logging_config import configure_logging, get_logger
from ticket_router.modeling.mlflow_tracking import configure_experiment_tracking
from ticket_router.registry.config import (
    DEFAULT_FINAL_MODEL_CONFIG_PATH,
    FinalModelConfig,
)
from ticket_router.registry.contracts import (
    prediction_contract_passes,
    signature_matches_text_api,
)
from ticket_router.registry.gates import PromotionEvidence, evaluate_promotion_gates
from ticket_router.registry.service import ModelRegistryService, RegisteredVersion


class PromotionError(RuntimeError):
    """Raised when explicit alias promotion is unsafe or unauthorized."""


def promote_candidate(
    *,
    settings: Settings,
    final_config: FinalModelConfig,
    split_manifest_path: Path,
    audit_directory: Path,
    project_root: Path,
    approved: bool,
) -> dict[str, object]:
    """Recheck live evidence and move champion only with explicit approval."""
    if not approved:
        raise PromotionError("Promotion requires the explicit --approve flag.")
    tracking = configure_experiment_tracking(
        settings=settings,
        experiment_config=final_config,
        project_root=project_root.resolve(),
    )
    registry = ModelRegistryService()
    candidate = registry.resolve_alias(
        name=final_config.registry.model_name,
        alias=final_config.registry.candidate_alias,
    )
    if candidate is None:
        raise PromotionError("No candidate alias exists; run final registration first.")
    champion = registry.resolve_alias(
        name=final_config.registry.model_name,
        alias=final_config.registry.champion_alias,
    )
    candidate_tags = registry.model_version_tags(name=candidate.name, version=candidate.version)
    split_manifest = SplitManifest.read(split_manifest_path.resolve())
    labels = tuple(
        label for label, _ in sorted(split_manifest.label_mapping.items(), key=lambda item: item[1])
    )
    model_load_succeeded = False
    contract_passed = False
    try:
        model = registry.load_alias(
            name=candidate.name, alias=final_config.registry.candidate_alias
        )
        model_load_succeeded = True
        contract_passed = prediction_contract_passes(model, labels=labels)
    except (OSError, RuntimeError, ValueError):
        pass
    signature = get_model_info(
        f"models:/{candidate.name}@{final_config.registry.candidate_alias}"
    ).signature
    signature_compatible = signature_matches_text_api(
        signature,
        required_input_dtype=final_config.promotion.required_input_dtype,
        required_output_dtype=final_config.promotion.required_output_dtype,
    )
    evidence = _evidence_from_tags(
        candidate_tags,
        model_load_succeeded=model_load_succeeded,
        prediction_contract_passed=contract_passed,
        signature_compatible=signature_compatible,
    )
    champion_macro_f1 = _champion_macro_f1(registry, champion)
    decision = evaluate_promotion_gates(
        evidence,
        settings=final_config.promotion,
        champion_macro_f1=champion_macro_f1,
    )
    timestamp = datetime.now(UTC)
    git_version = get_git_version(project_root.resolve())
    audit: dict[str, object] = {
        "timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
        "registered_model_name": candidate.name,
        "candidate_version": candidate.version,
        "champion_version_before": champion.version if champion is not None else None,
        "tracking_uri": tracking.resolved_uri,
        "git_commit": git_version.commit if git_version else None,
        "git_dirty": git_version.dirty if git_version else None,
        "explicit_approval": True,
        "gate_results": decision.to_dict(),
        "promoted": False,
    }
    audit_directory = audit_directory.resolve()
    audit_directory.mkdir(parents=True, exist_ok=True)
    audit_path = audit_directory / (
        f"promotion-{timestamp.strftime('%Y%m%dT%H%M%SZ')}-v{candidate.version}.json"
    )
    if not decision.allowed:
        atomic_write_json(audit_path, audit)
        raise PromotionError(f"Candidate failed promotion gates; audit: {audit_path}")
    promoted = registry.promote_candidate(
        name=candidate.name,
        candidate_alias=final_config.registry.candidate_alias,
        champion_alias=final_config.registry.champion_alias,
    )
    registry.set_model_version_tags(
        name=promoted.name,
        version=promoted.version,
        tags={
            "champion_alias": final_config.registry.champion_alias,
            "champion_promoted_at_utc": str(audit["timestamp_utc"]),
        },
    )
    audit["promoted"] = True
    audit["champion_version_after"] = promoted.version
    atomic_write_json(audit_path, audit)
    return {**audit, "audit_path": str(audit_path)}


def _evidence_from_tags(
    tags: dict[str, str],
    *,
    model_load_succeeded: bool,
    prediction_contract_passed: bool,
    signature_compatible: bool,
) -> PromotionEvidence:
    required = (
        "test_macro_f1",
        "minimum_per_class_recall",
        "inference_milliseconds_per_record",
    )
    missing = [key for key in required if key not in tags]
    if missing:
        raise PromotionError("Candidate registry lineage tags are missing: " + ", ".join(missing))
    return PromotionEvidence(
        macro_f1=float(tags["test_macro_f1"]),
        minimum_per_class_recall=float(tags["minimum_per_class_recall"]),
        inference_milliseconds_per_record=float(tags["inference_milliseconds_per_record"]),
        model_load_succeeded=model_load_succeeded,
        prediction_contract_passed=prediction_contract_passed,
        signature_compatible=signature_compatible,
    )


def _champion_macro_f1(
    registry: ModelRegistryService,
    champion: RegisteredVersion | None,
) -> float | None:
    if champion is None:
        return None
    tags = registry.model_version_tags(name=champion.name, version=champion.version)
    if "test_macro_f1" not in tags:
        raise PromotionError("Existing champion is missing its test_macro_f1 lineage tag.")
    return float(tags["test_macro_f1"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--final-config", type=Path, default=DEFAULT_FINAL_MODEL_CONFIG_PATH)
    parser.add_argument(
        "--split-manifest", type=Path, default=Path("data/processed/split_manifest.json")
    )
    parser.add_argument(
        "--audit-directory", type=Path, default=Path("artifacts/reports/promotions")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = Settings.load(args.config)
    final_config = FinalModelConfig.load(args.final_config)
    configure_logging(settings.log_level)
    logger = get_logger(__name__)
    try:
        result = promote_candidate(
            settings=settings,
            final_config=final_config,
            split_manifest_path=args.split_manifest,
            audit_directory=args.audit_directory,
            project_root=Path.cwd(),
            approved=args.approve,
        )
    except (ConnectionError, FileNotFoundError, PromotionError, ValueError) as exc:
        logger.error("candidate_promotion_failed", error=str(exc))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
