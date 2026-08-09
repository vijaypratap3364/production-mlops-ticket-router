"""Framework-neutral workflow decisions and machine-readable results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class RetrainingDecision:
    """Auditable decision made before any training work begins."""

    should_retrain: bool
    trigger: Literal["automatic", "manual", "none"]
    reasons: tuple[str, ...]
    feedback_count: int
    consecutive_critical_windows: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateWorkflowResult:
    """Serializable outcome of candidate training, registration, and gates."""

    workflow_run_id: str
    selected_candidate: str
    mlflow_run_id: str
    registered_model_name: str
    candidate_model_version: str
    promotion_gates_passed: bool
    champion_promotion_performed: bool
    summary_path: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ConditionalRetrainingResult:
    """Serializable terminal state for a conditional retraining evaluation."""

    run_id: str
    status: Literal["not_triggered", "insufficient_feedback", "candidate_registered", "failed"]
    decision: RetrainingDecision
    dataset_manifest_path: str | None
    candidate: CandidateWorkflowResult | None
    champion_promotion_performed: bool = False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["decision"] = self.decision.to_dict()
        payload["candidate"] = self.candidate.to_dict() if self.candidate else None
        return payload
