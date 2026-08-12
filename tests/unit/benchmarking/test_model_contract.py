"""Registered champion API, probability, label, and serialization contract tests."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast

import numpy as np

from ticket_router.api.service import LoadedChampion, ProbabilisticPredictor
from ticket_router.benchmarking.contract import validate_champion_contract
from ticket_router.config import Settings
from ticket_router.data.split_manifest import SplitManifest


class SerializableFixtureModel:
    classes_ = np.asarray(["Billing", "Returns", "Technical"], dtype=object)

    def predict_proba(self, values: Sequence[str]) -> object:
        rows = []
        for value in values:
            if "invoice" in value.casefold():
                rows.append([0.8, 0.1, 0.1])
            else:
                rows.append([0.1, 0.1, 0.8])
        return np.asarray(rows, dtype=np.float64)

    def predict(self, values: Sequence[str]) -> object:
        probabilities = np.asarray(self.predict_proba(values), dtype=np.float64)
        return self.classes_[probabilities.argmax(axis=1)]


def test_champion_contract_verifies_schema_labels_probabilities_top_k_and_roundtrip() -> None:
    settings = Settings.load(env_file=None)
    champion = LoadedChampion(
        model=cast(ProbabilisticPredictor, SerializableFixtureModel()),
        model_name="fixture-router",
        model_version="9",
        alias="champion",
        loaded_at=datetime(2026, 8, 12, tzinfo=UTC),
        labels=("Billing", "Returns", "Technical"),
        input_contract={"predictive_fields": ["subject", "body"]},
    )
    manifest = cast(
        SplitManifest,
        SimpleNamespace(label_mapping={"Billing": 0, "Returns": 1, "Technical": 2}),
    )

    result = validate_champion_contract(
        champion=champion,
        settings=settings,
        split_manifest=manifest,
    )

    assert result.passed is True
    assert all(result.checks.values())
    assert result.probability_dimensions == (2, 3)
    assert result.top_k_size == 3
    assert result.deterministic_fixed_sample is True
    assert result.serialized_roundtrip_size_bytes > 0


def test_champion_contract_rejects_mismatched_label_mapping() -> None:
    settings = Settings.load(env_file=None)
    champion = LoadedChampion(
        model=cast(ProbabilisticPredictor, SerializableFixtureModel()),
        model_name="fixture-router",
        model_version="9",
        alias="champion",
        loaded_at=datetime(2026, 8, 12, tzinfo=UTC),
        labels=("Billing", "Returns", "Technical"),
        input_contract={"predictive_fields": ["subject", "body"]},
    )
    manifest = cast(
        SplitManifest,
        SimpleNamespace(label_mapping={"Billing": 0, "Returns": 1, "Other": 2}),
    )

    result = validate_champion_contract(
        champion=champion,
        settings=settings,
        split_manifest=manifest,
    )

    assert result.passed is False
    assert result.checks["label_mapping_matches_model"] is False
