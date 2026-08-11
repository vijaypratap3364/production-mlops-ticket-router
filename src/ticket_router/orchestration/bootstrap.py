"""Explicit, idempotent fresh-machine bootstrap for the local Compose stack."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mlflow

from ticket_router.config import Settings
from ticket_router.registry.service import ModelRegistryService


@dataclass(frozen=True)
class BootstrapStep:
    name: str
    module: str
    arguments: tuple[str, ...] = ()


@dataclass(frozen=True)
class BootstrapResult:
    status: Literal["already_ready", "candidate_promoted", "recovered", "completed"]
    executed_steps: tuple[str, ...]


StepRunner = Callable[[BootstrapStep], None]
DEFAULT_FINAL_ACCESS_AUDIT = Path("artifacts/reports/final_evaluation/test_access_audit.json")


class BootstrapError(RuntimeError):
    """Raised when persisted state makes a safe bootstrap continuation ambiguous."""


def bootstrap_steps(*, promote_champion: bool) -> tuple[BootstrapStep, ...]:
    """Return the auditable commands; promotion is absent unless explicitly requested."""
    steps = [
        BootstrapStep("download pinned dataset", "ticket_router.data.download"),
        BootstrapStep("normalize English tickets", "ticket_router.data.normalize"),
        BootstrapStep("validate and analyze data", "ticket_router.data.analyze"),
        BootstrapStep("prepare leakage-safe splits", "ticket_router.data.prepare"),
        BootstrapStep("train validation candidates", "ticket_router.modeling.train_candidates"),
        BootstrapStep(
            "evaluate and register final candidate", "ticket_router.registry.evaluate_final"
        ),
    ]
    if promote_champion:
        steps.append(
            BootstrapStep(
                "explicitly promote gated candidate",
                "ticket_router.registry.promote",
                ("--approve",),
            )
        )
    return tuple(steps)


def execute_bootstrap(
    *,
    settings: Settings,
    promote_champion: bool,
    registry: ModelRegistryService | None = None,
    runner: StepRunner | None = None,
    final_access_audit: Path = DEFAULT_FINAL_ACCESS_AUDIT,
) -> BootstrapResult:
    """Reuse a ready registry or execute the full local, zero-cost bootstrap sequence."""
    mlflow.set_tracking_uri(settings.effective_mlflow_tracking_uri)
    service = registry or ModelRegistryService()
    model_name = settings.effective_registered_model_name
    champion_alias = settings.mlflow_model_alias
    champion = service.resolve_alias(name=model_name, alias=champion_alias)
    if champion is not None:
        return BootstrapResult(status="already_ready", executed_steps=())

    run_step = runner or _run_step
    candidate = service.resolve_alias(name=model_name, alias="candidate")
    if candidate is not None:
        if not promote_champion:
            return BootstrapResult(status="completed", executed_steps=())
        promotion = BootstrapStep(
            "explicitly promote gated candidate",
            "ticket_router.registry.promote",
            ("--approve",),
        )
        run_step(promotion)
        return BootstrapResult(
            status="candidate_promoted",
            executed_steps=(promotion.name,),
        )

    audit_status = _final_access_audit_status(final_access_audit)
    if audit_status == "authorized_and_opened":
        recovery = BootstrapStep(
            "recover interrupted final registration",
            "ticket_router.registry.recover_final",
        )
        run_step(recovery)
        recovery_steps = [recovery.name]
        if promote_champion:
            promotion = BootstrapStep(
                "explicitly promote gated candidate",
                "ticket_router.registry.promote",
                ("--approve",),
            )
            run_step(promotion)
            recovery_steps.append(promotion.name)
        return BootstrapResult(status="recovered", executed_steps=tuple(recovery_steps))
    if audit_status is not None:
        raise BootstrapError(
            "A final-test access audit exists without a candidate or champion alias "
            f"(status={audit_status!r}). Refusing to retrain or reopen the held-out test set."
        )

    executed: list[str] = []
    for step in bootstrap_steps(promote_champion=promote_champion):
        run_step(step)
        executed.append(step.name)
    return BootstrapResult(status="completed", executed_steps=tuple(executed))


def _final_access_audit_status(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BootstrapError(f"Final-test access audit is unreadable: {path}") from exc
    if not isinstance(value, dict) or not isinstance(status := value.get("status"), str):
        raise BootstrapError(f"Final-test access audit has no valid status: {path}")
    return status


def _run_step(step: BootstrapStep) -> None:
    print(f"Bootstrap step: {step.name}", flush=True)
    subprocess.run(
        [sys.executable, "-m", step.module, *step.arguments],
        check=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--promote-champion",
        action="store_true",
        help="Run the separate gated --approve command after candidate registration.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = execute_bootstrap(
            settings=Settings.load(),
            promote_champion=args.promote_champion,
        )
    except BootstrapError as exc:
        print(f"Bootstrap failed safely: {exc}", file=sys.stderr)
        return 1
    print(f"Bootstrap status: {result.status}")
    for step in result.executed_steps:
        print(f"Completed: {step}")
    if not args.promote_champion and result.status != "already_ready":
        print("Champion promotion was not requested; the API may remain not-ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
