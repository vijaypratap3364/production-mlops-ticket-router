"""Model signature and prediction-contract checks shared by registry gates and serving."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Protocol, cast

import numpy as np
from mlflow.models import ModelSignature
from numpy.typing import NDArray


class Predictor(Protocol):
    """Minimum interface required by the future prediction service."""

    def predict(self, values: Sequence[str]) -> object: ...


def signature_matches_text_api(
    signature: ModelSignature | None,
    *,
    required_input_dtype: str,
    required_output_dtype: str,
) -> bool:
    """Require one variable-length string tensor in and one string tensor out."""
    if signature is None:
        return False
    rendered = signature.to_dict()
    inputs = _signature_fields(rendered.get("inputs"))
    outputs = _signature_fields(rendered.get("outputs"))
    return _is_string_vector(inputs, required_input_dtype) and _is_string_vector(
        outputs, required_output_dtype
    )


def prediction_contract_passes(
    model: Predictor,
    *,
    labels: tuple[str, ...],
) -> bool:
    """Run privacy-safe smoke inputs and verify output shape and label membership."""
    examples = (
        "Example billing question for a local invoice.",
        "Example technical request about a network outage.",
        "Example return request for a product.",
    )
    try:
        observed = np.asarray(model.predict(examples), dtype=object)
    except (TypeError, ValueError, RuntimeError):
        return False
    if observed.shape != (len(examples),):
        return False
    allowed = set(labels)
    return all(str(value) in allowed for value in cast(NDArray[np.object_], observed))


def _signature_fields(value: object) -> list[dict[str, object]]:
    if not isinstance(value, str):
        return []
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _is_string_vector(fields: list[dict[str, object]], required_dtype: str) -> bool:
    if len(fields) != 1 or fields[0].get("type") != "tensor":
        return False
    tensor_spec = fields[0].get("tensor-spec")
    if not isinstance(tensor_spec, dict):
        return False
    return tensor_spec.get("dtype") == required_dtype and tensor_spec.get("shape") == [-1]
