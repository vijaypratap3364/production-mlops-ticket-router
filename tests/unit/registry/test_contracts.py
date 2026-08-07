"""Tests for the future API prediction and MLflow signature contract."""

from collections.abc import Sequence

import numpy as np
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]

from ticket_router.modeling.final_model import build_final_pipeline
from ticket_router.modeling.mlflow_tracking import create_model_signature
from ticket_router.registry.config import FinalModelConfig
from ticket_router.registry.contracts import (
    prediction_contract_passes,
    signature_matches_text_api,
)


def _fitted_final_pipeline() -> tuple[Pipeline, tuple[str, ...]]:
    config = FinalModelConfig.load()
    labels = ("Billing", "Technical")
    texts = [
        "invoice payment billing",
        "billing invoice refund",
        "payment receipt billing",
        "network outage technical",
        "technical router failure",
        "network connection support",
    ]
    targets = [labels[0]] * 3 + [labels[1]] * 3
    pipeline = build_final_pipeline(config.selected_candidate, random_seed=42)
    pipeline.fit(texts, targets)
    return pipeline, labels


def test_signature_matches_api_string_tensor_contract() -> None:
    model, _ = _fitted_final_pipeline()
    signature, _ = create_model_signature(model)

    assert signature_matches_text_api(
        signature,
        required_input_dtype="str",
        required_output_dtype="str",
    )
    assert not signature_matches_text_api(
        None,
        required_input_dtype="str",
        required_output_dtype="str",
    )


def test_prediction_contract_accepts_only_known_label_vector() -> None:
    model, labels = _fitted_final_pipeline()

    assert prediction_contract_passes(model, labels=labels)

    class InvalidPredictor:
        def predict(self, values: Sequence[str]) -> object:
            return np.asarray(["unknown"] * len(values), dtype=object)

    assert not prediction_contract_passes(InvalidPredictor(), labels=labels)
