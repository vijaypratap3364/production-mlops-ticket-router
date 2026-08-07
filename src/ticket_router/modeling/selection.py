"""Deterministic candidate guardrails and macro-F1 ranking."""

from __future__ import annotations

from dataclasses import dataclass

from ticket_router.modeling.experiment_config import CandidateSelectionSettings


class CandidateSelectionError(RuntimeError):
    """Raised when no candidate satisfies the configured safeguards."""


@dataclass(frozen=True)
class CandidateEvidence:
    """Validation and cross-validation evidence used by selection."""

    name: str
    metrics: dict[str, float]
    per_class_metrics: tuple[dict[str, float | int | str], ...]
    cv_macro_f1_mean: float
    cv_macro_f1_standard_deviation: float
    inference_milliseconds_per_record: float
    serialized_model_size_bytes: int


@dataclass(frozen=True)
class CandidateDecision:
    """Guardrail result for one candidate."""

    candidate: CandidateEvidence
    eligible: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class CandidateRanking:
    """Full deterministic ranking plus selected eligible candidate."""

    selected_candidate: str
    decisions: tuple[CandidateDecision, ...]


def rank_candidates(
    candidates: tuple[CandidateEvidence, ...],
    *,
    settings: CandidateSelectionSettings,
) -> CandidateRanking:
    """Apply every guardrail, then rank eligible candidates by validation macro F1."""
    if not candidates:
        raise CandidateSelectionError("Candidate selection requires at least one candidate.")
    decisions = tuple(_evaluate(candidate, settings) for candidate in candidates)
    ordered = tuple(
        sorted(
            decisions,
            key=lambda decision: (
                not decision.eligible,
                -decision.candidate.metrics["macro_f1"],
                -decision.candidate.metrics["weighted_f1"],
                decision.candidate.name,
            ),
        )
    )
    eligible = [decision for decision in ordered if decision.eligible]
    if not eligible:
        summary = "; ".join(
            f"{decision.candidate.name}: {', '.join(decision.rejection_reasons)}"
            for decision in ordered
        )
        raise CandidateSelectionError("No candidate passed all selection guardrails. " + summary)
    return CandidateRanking(
        selected_candidate=eligible[0].candidate.name,
        decisions=ordered,
    )


def _evaluate(
    candidate: CandidateEvidence,
    settings: CandidateSelectionSettings,
) -> CandidateDecision:
    reasons: list[str] = []
    collapsed = [
        str(row["class"])
        for row in candidate.per_class_metrics
        if float(row["recall"]) < settings.minimum_per_class_recall
    ]
    if collapsed:
        reasons.append(
            "per-class recall below "
            f"{settings.minimum_per_class_recall:.3f}: {', '.join(collapsed)}"
        )
    major_collapsed = [
        str(row["class"])
        for row in candidate.per_class_metrics
        if int(row["support"]) >= settings.major_class_minimum_support
        and float(row["recall"]) < settings.major_class_minimum_recall
    ]
    if major_collapsed:
        reasons.append(
            "major-class recall below "
            f"{settings.major_class_minimum_recall:.3f}: {', '.join(major_collapsed)}"
        )
    if (
        candidate.inference_milliseconds_per_record
        > settings.maximum_inference_milliseconds_per_record
    ):
        reasons.append(
            "inference latency exceeds "
            f"{settings.maximum_inference_milliseconds_per_record:.3f} ms/record"
        )
    maximum_bytes = settings.maximum_serialized_model_size_mb * 1024 * 1024
    if candidate.serialized_model_size_bytes > maximum_bytes:
        reasons.append(
            f"serialized model exceeds {settings.maximum_serialized_model_size_mb:.1f} MiB"
        )
    if candidate.cv_macro_f1_standard_deviation > settings.maximum_cv_macro_f1_standard_deviation:
        reasons.append(
            "cross-validation macro-F1 standard deviation exceeds "
            f"{settings.maximum_cv_macro_f1_standard_deviation:.3f}"
        )
    cv_drop = candidate.cv_macro_f1_mean - candidate.metrics["macro_f1"]
    if cv_drop > settings.maximum_cv_to_validation_macro_f1_drop:
        reasons.append(
            "validation macro F1 is below cross-validation mean by more than "
            f"{settings.maximum_cv_to_validation_macro_f1_drop:.3f}"
        )
    return CandidateDecision(
        candidate=candidate,
        eligible=not reasons,
        rejection_reasons=tuple(reasons),
    )
