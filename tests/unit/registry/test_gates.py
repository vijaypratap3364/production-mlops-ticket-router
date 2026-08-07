"""Tests for absolute and champion-relative promotion gates."""

from ticket_router.registry.config import FinalModelConfig
from ticket_router.registry.gates import PromotionEvidence, evaluate_promotion_gates


def _passing_evidence() -> PromotionEvidence:
    return PromotionEvidence(
        macro_f1=0.68,
        minimum_per_class_recall=0.52,
        inference_milliseconds_per_record=0.2,
        model_load_succeeded=True,
        prediction_contract_passed=True,
        signature_compatible=True,
    )


def test_initial_promotion_passes_only_when_every_absolute_gate_passes() -> None:
    settings = FinalModelConfig.load().promotion
    decision = evaluate_promotion_gates(
        _passing_evidence(),
        settings=settings,
        champion_macro_f1=None,
    )

    assert decision.allowed
    assert decision.first_champion
    assert all(check.passed for check in decision.checks)


def test_failed_promotion_reports_quality_and_contract_failures() -> None:
    settings = FinalModelConfig.load().promotion
    evidence = PromotionEvidence(
        macro_f1=0.55,
        minimum_per_class_recall=0.20,
        inference_milliseconds_per_record=2.0,
        model_load_succeeded=False,
        prediction_contract_passed=False,
        signature_compatible=False,
    )
    decision = evaluate_promotion_gates(
        evidence,
        settings=settings,
        champion_macro_f1=0.70,
    )

    assert not decision.allowed
    assert not decision.first_champion
    assert {check.name for check in decision.checks if not check.passed} == {
        "minimum_macro_f1",
        "champion_macro_f1_regression",
        "minimum_per_class_recall",
        "inference_latency",
        "model_load",
        "prediction_contract",
        "model_signature",
    }
