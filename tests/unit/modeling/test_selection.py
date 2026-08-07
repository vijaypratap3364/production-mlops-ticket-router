"""Tests for deterministic candidate ranking and operational guardrails."""

from __future__ import annotations

from ticket_router.modeling.experiment_config import CandidateExperimentSettings
from ticket_router.modeling.selection import CandidateEvidence, rank_candidates


def test_candidate_ranking_uses_macro_f1_after_guardrails(
    small_experiment_config: CandidateExperimentSettings,
) -> None:
    lower = _evidence("lower", macro_f1=0.60, recall=0.60)
    stronger = _evidence("stronger", macro_f1=0.65, recall=0.60)

    ranking = rank_candidates(
        (lower, stronger),
        settings=small_experiment_config.selection,
    )

    assert ranking.selected_candidate == "stronger"
    assert all(decision.eligible for decision in ranking.decisions)


def test_candidate_guardrails_reject_class_collapse_and_latency(
    small_experiment_config: CandidateExperimentSettings,
) -> None:
    safe = _evidence("safe", macro_f1=0.60, recall=0.60)
    collapsed = _evidence(
        "collapsed",
        macro_f1=0.75,
        recall=0.10,
        latency=20.0,
    )

    ranking = rank_candidates(
        (collapsed, safe),
        settings=small_experiment_config.selection,
    )
    decision = next(item for item in ranking.decisions if item.candidate.name == "collapsed")

    assert ranking.selected_candidate == "safe"
    assert not decision.eligible
    assert any("per-class recall" in reason for reason in decision.rejection_reasons)
    assert any("latency" in reason for reason in decision.rejection_reasons)


def _evidence(
    name: str,
    *,
    macro_f1: float,
    recall: float,
    latency: float = 0.1,
) -> CandidateEvidence:
    return CandidateEvidence(
        name=name,
        metrics={"macro_f1": macro_f1, "weighted_f1": macro_f1},
        per_class_metrics=(
            {
                "class": "Queue",
                "precision": 0.60,
                "recall": recall,
                "f1": 0.60,
                "support": 10,
            },
        ),
        cv_macro_f1_mean=macro_f1,
        cv_macro_f1_standard_deviation=0.01,
        inference_milliseconds_per_record=latency,
        serialized_model_size_bytes=1024,
    )
