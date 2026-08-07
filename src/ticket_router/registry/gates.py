"""Pure, auditable candidate-versus-champion promotion gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ticket_router.registry.config import PromotionSettings


@dataclass(frozen=True)
class PromotionEvidence:
    """Quality and operational evidence attached to a registered model version."""

    macro_f1: float
    minimum_per_class_recall: float
    inference_milliseconds_per_record: float
    model_load_succeeded: bool
    prediction_contract_passed: bool
    signature_compatible: bool


@dataclass(frozen=True)
class GateResult:
    """One explicit promotion predicate and its measured outcome."""

    name: str
    passed: bool
    actual: str
    requirement: str


@dataclass(frozen=True)
class PromotionDecision:
    """Complete gate record; aliases move only when every result passes."""

    allowed: bool
    first_champion: bool
    checks: tuple[GateResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "first_champion": self.first_champion,
            "checks": [asdict(check) for check in self.checks],
        }


def evaluate_promotion_gates(
    candidate: PromotionEvidence,
    *,
    settings: PromotionSettings,
    champion_macro_f1: float | None,
) -> PromotionDecision:
    """Apply absolute gates and the relative champion regression tolerance."""
    checks = [
        GateResult(
            name="minimum_macro_f1",
            passed=candidate.macro_f1 >= settings.minimum_macro_f1,
            actual=f"{candidate.macro_f1:.6f}",
            requirement=f">= {settings.minimum_macro_f1:.6f}",
        ),
        GateResult(
            name="champion_macro_f1_regression",
            passed=(
                champion_macro_f1 is None
                or candidate.macro_f1 >= champion_macro_f1 - settings.maximum_macro_f1_regression
            ),
            actual=(
                "no existing champion"
                if champion_macro_f1 is None
                else f"candidate={candidate.macro_f1:.6f}, champion={champion_macro_f1:.6f}"
            ),
            requirement=(
                "absolute gates apply to initial champion"
                if champion_macro_f1 is None
                else f"regression <= {settings.maximum_macro_f1_regression:.6f}"
            ),
        ),
        GateResult(
            name="minimum_per_class_recall",
            passed=candidate.minimum_per_class_recall >= settings.minimum_per_class_recall,
            actual=f"{candidate.minimum_per_class_recall:.6f}",
            requirement=f">= {settings.minimum_per_class_recall:.6f}",
        ),
        GateResult(
            name="inference_latency",
            passed=(
                candidate.inference_milliseconds_per_record
                <= settings.maximum_inference_milliseconds_per_record
            ),
            actual=f"{candidate.inference_milliseconds_per_record:.6f} ms/record",
            requirement=(f"<= {settings.maximum_inference_milliseconds_per_record:.6f} ms/record"),
        ),
        GateResult(
            name="model_load",
            passed=candidate.model_load_succeeded or not settings.require_model_load,
            actual=str(candidate.model_load_succeeded).lower(),
            requirement=str(settings.require_model_load).lower(),
        ),
        GateResult(
            name="prediction_contract",
            passed=(
                candidate.prediction_contract_passed or not settings.require_prediction_contract
            ),
            actual=str(candidate.prediction_contract_passed).lower(),
            requirement=str(settings.require_prediction_contract).lower(),
        ),
        GateResult(
            name="model_signature",
            passed=(candidate.signature_compatible or not settings.require_signature_compatibility),
            actual=str(candidate.signature_compatible).lower(),
            requirement=str(settings.require_signature_compatibility).lower(),
        ),
    ]
    return PromotionDecision(
        allowed=all(check.passed for check in checks),
        first_champion=champion_macro_f1 is None,
        checks=tuple(checks),
    )
