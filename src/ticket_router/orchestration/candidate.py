"""Candidate workflow operations with idempotent registry writes."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from mlflow.models import get_model_info

from ticket_router.data.manifests import atomic_write_json
from ticket_router.modeling.train_candidates import CandidateExperimentResult, CandidateRunRecord
from ticket_router.orchestration.contracts import CandidateWorkflowResult
from ticket_router.registry.config import FinalModelConfig
from ticket_router.registry.contracts import prediction_contract_passes, signature_matches_text_api
from ticket_router.registry.gates import PromotionEvidence, evaluate_promotion_gates
from ticket_router.registry.service import ModelRegistryService


def register_and_gate_candidate(
    *,
    experiment: CandidateExperimentResult,
    final_config: FinalModelConfig,
    workflow_run_id: str,
    dataset_manifest_sha256: str,
    orchestration_configuration_sha256: str,
    summary_path: Path,
    registry: ModelRegistryService | None = None,
) -> CandidateWorkflowResult:
    """Register the selected validation candidate and record gates without promotion."""
    selected = _selected_record(experiment)
    service = registry or ModelRegistryService()
    registered = service.find_version_by_run_id(
        name=final_config.registry.model_name,
        run_id=selected.mlflow_run.run_id,
    )
    tags = {
        "stage": "stage11_orchestrated_candidate",
        "validation_macro_f1": str(selected.evaluation.metrics["macro_f1"]),
        "minimum_per_class_recall": str(_minimum_recall(selected)),
        "inference_milliseconds_per_record": str(_latency(selected)),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "orchestration_configuration_sha256": orchestration_configuration_sha256,
        "test_evaluated": "false",
        "human_promotion_required": "true",
    }
    if registered is None:
        registered = service.register_candidate(
            name=final_config.registry.model_name,
            model_uri=selected.mlflow_run.model_uri,
            run_id=selected.mlflow_run.run_id,
            candidate_alias=final_config.registry.candidate_alias,
            tags=tags,
        )
    else:
        service.assign_alias(
            name=registered.name,
            alias=final_config.registry.candidate_alias,
            version=registered.version,
        )
        service.set_model_version_tags(name=registered.name, version=registered.version, tags=tags)

    loaded = service.load_version(name=registered.name, version=registered.version)
    labels = tuple(str(row["class"]) for row in selected.evaluation.per_class_metrics)
    model_info = get_model_info(selected.mlflow_run.model_uri)
    evidence = PromotionEvidence(
        macro_f1=selected.evaluation.metrics["macro_f1"],
        minimum_per_class_recall=_minimum_recall(selected),
        inference_milliseconds_per_record=_latency(selected),
        model_load_succeeded=True,
        prediction_contract_passed=prediction_contract_passes(loaded, labels=labels),
        signature_compatible=signature_matches_text_api(
            model_info.signature,
            required_input_dtype=final_config.promotion.required_input_dtype,
            required_output_dtype=final_config.promotion.required_output_dtype,
        ),
    )
    champion = service.resolve_alias(
        name=final_config.registry.model_name,
        alias=final_config.registry.champion_alias,
    )
    champion_metric = (
        _champion_metric(service, champion.name, champion.version) if champion else None
    )
    decision = evaluate_promotion_gates(
        evidence,
        settings=final_config.promotion,
        champion_macro_f1=champion_metric,
    )
    service.set_model_version_tags(
        name=registered.name,
        version=registered.version,
        tags={
            "promotion_gates_passed": str(decision.allowed).lower(),
            "champion_promotion_performed": "false",
        },
    )
    result = CandidateWorkflowResult(
        workflow_run_id=workflow_run_id,
        selected_candidate=experiment.selected_candidate,
        mlflow_run_id=selected.mlflow_run.run_id,
        registered_model_name=registered.name,
        candidate_model_version=registered.version,
        promotion_gates_passed=decision.allowed,
        champion_promotion_performed=False,
        summary_path=summary_path.as_posix(),
    )
    atomic_write_json(
        summary_path,
        {
            **result.to_dict(),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "orchestration_configuration_sha256": orchestration_configuration_sha256,
            "promotion_gate_results": decision.to_dict(),
            "evidence_basis": "validation-only temporal candidate evidence",
            "test_evaluated": False,
            "human_promotion_required": True,
            "promotion_command": "uv run python -m ticket_router.registry.promote --approve",
        },
    )
    return result


def read_candidate_workflow_result(
    path: Path,
    *,
    expected_dataset_manifest_sha256: str,
    expected_orchestration_configuration_sha256: str,
) -> CandidateWorkflowResult:
    """Load an idempotent completed result from its machine-readable summary."""
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset_manifest_sha256") != expected_dataset_manifest_sha256:
        raise ValueError("completed candidate workflow belongs to a different dataset manifest")
    if (
        payload.get("orchestration_configuration_sha256")
        != expected_orchestration_configuration_sha256
    ):
        raise ValueError("completed candidate workflow belongs to different orchestration settings")
    return CandidateWorkflowResult(
        **{name: payload[name] for name in CandidateWorkflowResult.__dataclass_fields__}
    )


def _selected_record(experiment: CandidateExperimentResult) -> CandidateRunRecord:
    return next(
        record
        for record in experiment.candidates
        if record.spec.name == experiment.selected_candidate
    )


def _minimum_recall(record: CandidateRunRecord) -> float:
    return min(float(row["recall"]) for row in record.evaluation.per_class_metrics)


def _latency(record: CandidateRunRecord) -> float:
    return float(
        cast(float, record.artifacts.inference_benchmark["median_milliseconds_per_record"])
    )


def _champion_metric(service: ModelRegistryService, name: str, version: str) -> float:
    tags = service.model_version_tags(name=name, version=version)
    value = tags.get("validation_macro_f1") or tags.get("test_macro_f1")
    if value is None:
        raise ValueError("existing champion lacks a comparable macro-F1 lineage tag")
    return float(value)
