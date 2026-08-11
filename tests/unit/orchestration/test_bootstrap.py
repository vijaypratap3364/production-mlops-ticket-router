"""Compose bootstrap ordering, idempotency, and promotion-boundary tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ticket_router.config import Settings
from ticket_router.orchestration.bootstrap import (
    BootstrapError,
    BootstrapStep,
    bootstrap_steps,
    execute_bootstrap,
)
from ticket_router.registry.service import ModelRegistryService, RegisteredVersion


class AliasRegistry(ModelRegistryService):
    def __init__(self, aliases: dict[str, RegisteredVersion]) -> None:
        self.aliases = aliases

    def resolve_alias(self, *, name: str, alias: str) -> RegisteredVersion | None:
        return self.aliases.get(alias)


def _version(version: str) -> RegisteredVersion:
    return RegisteredVersion(
        name="ticket-router",
        version=version,
        run_id=f"run-{version}",
        source=f"models:/{version}",
    )


def _settings() -> Settings:
    return Settings.load(env_file=None)


def test_bootstrap_never_promotes_without_explicit_flag() -> None:
    safe_steps = bootstrap_steps(promote_champion=False)
    approved_steps = bootstrap_steps(promote_champion=True)

    assert all(step.module != "ticket_router.registry.promote" for step in safe_steps)
    assert approved_steps[-1].module == "ticket_router.registry.promote"
    assert approved_steps[-1].arguments == ("--approve",)


def test_fresh_bootstrap_runs_ordered_pipeline(tmp_path: Path) -> None:
    executed: list[BootstrapStep] = []

    result = execute_bootstrap(
        settings=_settings(),
        promote_champion=True,
        registry=AliasRegistry({}),
        runner=executed.append,
        final_access_audit=tmp_path / "missing-audit.json",
    )

    assert result.status == "completed"
    assert [step.module for step in executed] == [
        "ticket_router.data.download",
        "ticket_router.data.normalize",
        "ticket_router.data.analyze",
        "ticket_router.data.prepare",
        "ticket_router.modeling.train_candidates",
        "ticket_router.registry.evaluate_final",
        "ticket_router.registry.promote",
    ]


def test_bootstrap_reuses_existing_champion() -> None:
    executed: list[BootstrapStep] = []

    result = execute_bootstrap(
        settings=_settings(),
        promote_champion=True,
        registry=AliasRegistry({"champion": _version("7")}),
        runner=executed.append,
    )

    assert result.status == "already_ready"
    assert executed == []


def test_existing_candidate_requires_explicit_promotion() -> None:
    executed: list[BootstrapStep] = []

    result = execute_bootstrap(
        settings=_settings(),
        promote_champion=True,
        registry=AliasRegistry({"candidate": _version("8")}),
        runner=executed.append,
    )

    assert result.status == "candidate_promoted"
    assert [step.module for step in executed] == ["ticket_router.registry.promote"]


def test_interrupted_final_registration_recovers_without_retraining(tmp_path: Path) -> None:
    audit = tmp_path / "test_access_audit.json"
    audit.write_text(json.dumps({"status": "authorized_and_opened"}), encoding="utf-8")
    executed: list[BootstrapStep] = []

    result = execute_bootstrap(
        settings=_settings(),
        promote_champion=True,
        registry=AliasRegistry({}),
        runner=executed.append,
        final_access_audit=audit,
    )

    assert result.status == "recovered"
    assert [step.module for step in executed] == [
        "ticket_router.registry.recover_final",
        "ticket_router.registry.promote",
    ]


def test_completed_final_audit_without_registry_fails_safely(tmp_path: Path) -> None:
    audit = tmp_path / "test_access_audit.json"
    audit.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    executed: list[BootstrapStep] = []

    with pytest.raises(BootstrapError, match="Refusing to retrain"):
        execute_bootstrap(
            settings=_settings(),
            promote_champion=True,
            registry=AliasRegistry({}),
            runner=executed.append,
            final_access_audit=audit,
        )

    assert executed == []
