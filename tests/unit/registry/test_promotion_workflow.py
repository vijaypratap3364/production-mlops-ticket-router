"""Promotion behavior with registry, signature, and audit boundaries mocked locally."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from ticket_router.config import Settings
from ticket_router.data.split_manifest import SplitManifest
from ticket_router.registry import promote
from ticket_router.registry.config import FinalModelConfig
from ticket_router.registry.promote import PromotionError
from ticket_router.registry.service import ModelRegistryService, RegisteredVersion


class FixturePredictor:
    def predict(self, values: Sequence[str]) -> object:
        return np.asarray(["Billing"] * len(values), dtype=object)


class PromotionRegistry:
    def __init__(self, *, passing: bool) -> None:
        self.candidate = RegisteredVersion("ticket-router", "3", "run-3", "models:/3")
        self.passing = passing
        self.promoted = False
        self.tag_updates: list[dict[str, str]] = []

    def resolve_alias(self, *, name: str, alias: str) -> RegisteredVersion | None:
        del name
        return self.candidate if alias == "candidate" else None

    def model_version_tags(self, *, name: str, version: str) -> dict[str, str]:
        del name, version
        return {
            "test_macro_f1": "0.72" if self.passing else "0.10",
            "minimum_per_class_recall": "0.55" if self.passing else "0.10",
            "inference_milliseconds_per_record": "0.20" if self.passing else "20.0",
        }

    def load_alias(self, *, name: str, alias: str) -> FixturePredictor:
        del name, alias
        if not self.passing:
            raise RuntimeError("fixture load failure")
        return FixturePredictor()

    def promote_candidate(
        self,
        *,
        name: str,
        candidate_alias: str,
        champion_alias: str,
    ) -> RegisteredVersion:
        del name, candidate_alias, champion_alias
        self.promoted = True
        return self.candidate

    def set_model_version_tags(
        self,
        *,
        name: str,
        version: str,
        tags: dict[str, str],
    ) -> None:
        del name, version
        self.tag_updates.append(tags)


def _patch_promotion_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    registry: PromotionRegistry,
) -> None:
    manifest = cast(
        SplitManifest,
        SimpleNamespace(label_mapping={"Billing": 0, "Technical": 1}),
    )
    monkeypatch.setattr(
        promote,
        "configure_experiment_tracking",
        lambda **_: SimpleNamespace(resolved_uri="file:///fixture-mlruns"),
    )
    monkeypatch.setattr(
        promote,
        "ModelRegistryService",
        lambda: cast(ModelRegistryService, registry),
    )
    monkeypatch.setattr(
        promote,
        "SplitManifest",
        SimpleNamespace(read=lambda _: manifest),
    )
    monkeypatch.setattr(promote, "get_model_info", lambda _: SimpleNamespace(signature=object()))
    monkeypatch.setattr(promote, "get_git_version", lambda _: None)
    monkeypatch.setattr(promote, "signature_matches_text_api", lambda *_, **__: registry.passing)


def test_approved_candidate_moves_champion_and_writes_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = PromotionRegistry(passing=True)
    _patch_promotion_dependencies(monkeypatch, registry)

    result = promote.promote_candidate(
        settings=Settings.load(env_file=None),
        final_config=FinalModelConfig.load(),
        split_manifest_path=tmp_path / "split_manifest.json",
        audit_directory=tmp_path / "promotions",
        project_root=tmp_path,
        approved=True,
    )

    audit_path = Path(cast(str, result["audit_path"]))
    persisted = json.loads(audit_path.read_text(encoding="utf-8"))
    assert registry.promoted is True
    assert result["champion_version_after"] == "3"
    assert persisted["promoted"] is True
    assert persisted["gate_results"]["allowed"] is True
    assert registry.tag_updates[0]["champion_alias"] == "champion"


def test_failed_candidate_records_gate_evidence_without_moving_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = PromotionRegistry(passing=False)
    _patch_promotion_dependencies(monkeypatch, registry)

    with pytest.raises(PromotionError, match="failed promotion gates"):
        promote.promote_candidate(
            settings=Settings.load(env_file=None),
            final_config=FinalModelConfig.load(),
            split_manifest_path=tmp_path / "split_manifest.json",
            audit_directory=tmp_path / "promotions",
            project_root=tmp_path,
            approved=True,
        )

    audit_paths = tuple((tmp_path / "promotions").glob("promotion-*.json"))
    persisted = json.loads(audit_paths[0].read_text(encoding="utf-8"))
    assert registry.promoted is False
    assert persisted["promoted"] is False
    assert persisted["gate_results"]["allowed"] is False


def test_promotion_rejects_missing_lineage_and_invalid_existing_champion() -> None:
    with pytest.raises(PromotionError, match="minimum_per_class_recall"):
        promote._evidence_from_tags(
            {"test_macro_f1": "0.7"},
            model_load_succeeded=True,
            prediction_contract_passed=True,
            signature_compatible=True,
        )

    registry = PromotionRegistry(passing=True)
    champion = RegisteredVersion("ticket-router", "2", "run-2", "models:/2")
    registry.model_version_tags = lambda **_: {}  # type: ignore[method-assign]
    with pytest.raises(PromotionError, match="missing its test_macro_f1"):
        promote._champion_macro_f1(cast(ModelRegistryService, registry), champion)
