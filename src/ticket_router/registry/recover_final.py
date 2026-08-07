"""Resume post-evaluation registration without reopening the held-out test split."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import mlflow
import mlflow.sklearn
import polars as pl
from mlflow.entities import Run
from mlflow.models import get_model_info

from ticket_router.config import Settings
from ticket_router.data.manifests import atomic_write_json
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


class FinalRecoveryError(RuntimeError):
    """Raised when saved final-evaluation evidence is incomplete or ambiguous."""


def recover_final_registration(
    *,
    settings: Settings,
    final_config: FinalModelConfig,
    reports_dir: Path,
    model_artifacts_dir: Path,
    project_root: Path,
) -> dict[str, object]:
    """Register and gate a logged final model using saved evidence only."""
    reports_dir = reports_dir.resolve()
    audit_path = reports_dir / "test_access_audit.json"
    audit = _read_json(audit_path)
    if audit.get("status") != "authorized_and_opened":
        raise FinalRecoveryError("Recovery requires an incomplete authorized test-access audit.")
    evaluation_run_id = str(audit["evaluation_run_id"])
    artifact_directory = model_artifacts_dir.resolve() / evaluation_run_id / "final_model"
    metrics = _numeric_mapping(_read_json(artifact_directory / "metrics.json"))
    latency = _read_json(artifact_directory / "latency_distribution.json")
    lineage = _read_json(artifact_directory / "training_metadata.json")
    per_class_frame = pl.read_csv(artifact_directory / "per_class_metrics.csv")
    labels = tuple(str(value) for value in per_class_frame["class"].to_list())
    minimum_recall = float(cast(float, per_class_frame["recall"].min()))
    tracking = configure_experiment_tracking(
        settings=settings,
        experiment_config=final_config,
        project_root=project_root.resolve(),
    )
    client = mlflow.MlflowClient()
    run = _find_final_run(client, final_config, evaluation_run_id)
    logged_models = [
        model
        for model in client.search_logged_models(
            experiment_ids=[run.info.experiment_id],
            filter_string=f"source_run_id = '{run.info.run_id}'",
        )
        if str(model.status) == "READY"
    ]
    if len(logged_models) != 1:
        raise FinalRecoveryError(
            f"Expected one ready logged model for final run; found {len(logged_models)}."
        )
    model_uri = str(logged_models[0].model_uri)
    model_load_succeeded = False
    prediction_contract_passed = False
    try:
        loaded = mlflow.sklearn.load_model(model_uri)
        model_load_succeeded = True
        prediction_contract_passed = prediction_contract_passes(loaded, labels=labels)
    except (OSError, RuntimeError, ValueError):
        pass
    signature = get_model_info(model_uri).signature
    signature_compatible = signature_matches_text_api(
        signature,
        required_input_dtype=final_config.promotion.required_input_dtype,
        required_output_dtype=final_config.promotion.required_output_dtype,
    )
    evidence = PromotionEvidence(
        macro_f1=metrics["macro_f1"],
        minimum_per_class_recall=minimum_recall,
        inference_milliseconds_per_record=float(
            cast(float, latency["median_milliseconds_per_record"])
        ),
        model_load_succeeded=model_load_succeeded,
        prediction_contract_passed=prediction_contract_passed,
        signature_compatible=signature_compatible,
    )
    registry = ModelRegistryService()
    champion = registry.resolve_alias(
        name=final_config.registry.model_name,
        alias=final_config.registry.champion_alias,
    )
    champion_macro_f1 = _champion_macro_f1(registry, champion)
    registered = registry.register_candidate(
        name=final_config.registry.model_name,
        model_uri=model_uri,
        run_id=run.info.run_id,
        candidate_alias=final_config.registry.candidate_alias,
        tags=_registry_tags(lineage, evidence, artifact_directory),
    )
    decision = evaluate_promotion_gates(
        evidence,
        settings=final_config.promotion,
        champion_macro_f1=champion_macro_f1,
    )
    registry.set_model_version_tags(
        name=registered.name,
        version=registered.version,
        tags={
            "promotion_gates_passed": str(decision.allowed).lower(),
            "candidate_alias": final_config.registry.candidate_alias,
        },
    )
    report_directory = reports_dir / evaluation_run_id
    report_directory.mkdir(parents=True, exist_ok=True)
    gate_report = {
        "registered_model_name": registered.name,
        "candidate_version": registered.version,
        "candidate_alias": final_config.registry.candidate_alias,
        "champion_version_before_evaluation": champion.version if champion else None,
        "promotion": decision.to_dict(),
        "human_promotion_required": True,
        "promotion_command": "uv run python -m ticket_router.registry.promote --approve",
        "recovered_without_test_access": True,
    }
    atomic_write_json(report_directory / "promotion_gate_results.json", gate_report)
    summary = {
        **lineage,
        "mlflow_run_id": run.info.run_id,
        "model_uri": model_uri,
        "registered_model_name": registered.name,
        "registered_model_version": registered.version,
        "candidate_alias": final_config.registry.candidate_alias,
        "champion_alias": None,
        "metrics": metrics,
        "minimum_per_class_recall": minimum_recall,
        "promotion_allowed": decision.allowed,
        "recovered_without_test_access": True,
    }
    atomic_write_json(report_directory / "final_evaluation_summary.json", summary)
    client.set_tag(run.info.run_id, "registered_model_version", registered.version)
    client.set_tag(run.info.run_id, "promotion_gates_passed", str(decision.allowed).lower())
    client.set_terminated(run.info.run_id, status="FINISHED")
    completed_audit = {
        **audit,
        "status": "completed_after_post_evaluation_recovery",
        "mlflow_run_id": run.info.run_id,
        "registered_model_version": registered.version,
        "test_macro_f1": metrics["macro_f1"],
        "recovery_reloaded_test_data": False,
    }
    atomic_write_json(audit_path, completed_audit)
    return {**summary, "promotion_gates": decision.to_dict(), "tracking_uri": tracking.resolved_uri}


def _find_final_run(
    client: mlflow.MlflowClient,
    config: FinalModelConfig,
    evaluation_run_id: str,
) -> Run:
    experiment = client.get_experiment_by_name(config.experiment_name)
    if experiment is None:
        raise FinalRecoveryError("Final-evaluation MLflow experiment does not exist.")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{evaluation_run_id}'",
    )
    if len(runs) != 1:
        raise FinalRecoveryError(f"Expected one matching final run; found {len(runs)}.")
    return runs[0]


def _registry_tags(
    lineage: dict[str, object],
    evidence: PromotionEvidence,
    artifact_directory: Path,
) -> dict[str, str]:
    return {
        "stage": "stage7_final_evaluation",
        "test_macro_f1": str(evidence.macro_f1),
        "minimum_per_class_recall": str(evidence.minimum_per_class_recall),
        "inference_milliseconds_per_record": str(evidence.inference_milliseconds_per_record),
        "model_load_succeeded": str(evidence.model_load_succeeded).lower(),
        "prediction_contract_passed": str(evidence.prediction_contract_passed).lower(),
        "signature_compatible": str(evidence.signature_compatible).lower(),
        "split_manifest_sha256": str(lineage["split_manifest_sha256"]),
        "combined_training_data_sha256": str(lineage["combined_training_data_sha256"]),
        "test_data_sha256": str(lineage["test_data_sha256"]),
        "configuration_sha256": str(lineage["final_configuration_sha256"]),
        "git_commit": str(lineage["git_commit"] or "unavailable"),
        "evaluation_artifact_directory": artifact_directory.as_posix(),
        "post_evaluation_recovery": "true",
    }


def _champion_macro_f1(
    registry: ModelRegistryService,
    champion: RegisteredVersion | None,
) -> float | None:
    if champion is None:
        return None
    tags = registry.model_version_tags(name=champion.name, version=champion.version)
    if "test_macro_f1" not in tags:
        raise FinalRecoveryError("Existing champion lacks a test_macro_f1 tag.")
    return float(tags["test_macro_f1"])


def _read_json(path: Path) -> dict[str, object]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalRecoveryError(f"Expected a JSON object: {path}")
    return cast(dict[str, object], value)


def _numeric_mapping(values: dict[str, object]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in values.items():
        if not isinstance(value, (int, float)):
            raise FinalRecoveryError(f"Expected numeric metric {key!r}.")
        result[key] = float(value)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--final-config", type=Path, default=DEFAULT_FINAL_MODEL_CONFIG_PATH)
    parser.add_argument(
        "--reports-dir", type=Path, default=Path("artifacts/reports/final_evaluation")
    )
    parser.add_argument("--model-artifacts-dir", type=Path, default=Path("artifacts/models/final"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = Settings.load(args.config)
    final_config = FinalModelConfig.load(args.final_config)
    configure_logging(settings.log_level)
    logger = get_logger(__name__)
    try:
        result = recover_final_registration(
            settings=settings,
            final_config=final_config,
            reports_dir=args.reports_dir,
            model_artifacts_dir=args.model_artifacts_dir,
            project_root=Path.cwd(),
        )
    except (ConnectionError, FileNotFoundError, FinalRecoveryError, ValueError) as exc:
        logger.error("final_registration_recovery_failed", error=str(exc))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
