"""Resolve the champion alias once and pin its immutable registry version."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import cast

import mlflow
import numpy as np

from ticket_router.api.errors import ModelUnavailableError
from ticket_router.api.service import LoadedChampion, ProbabilisticPredictor
from ticket_router.config import Settings
from ticket_router.data.normalize import combine_ticket_text
from ticket_router.features.text import preprocess_model_text
from ticket_router.registry.service import ModelRegistryService

ChampionLoader = Callable[[Settings], LoadedChampion]


def load_champion(settings: Settings) -> LoadedChampion:
    """Resolve champion, then load the resolved numeric version to avoid alias races."""
    mlflow.set_tracking_uri(settings.effective_mlflow_tracking_uri)
    name = settings.effective_registered_model_name
    alias = settings.mlflow_model_alias
    registry = ModelRegistryService()
    resolved = registry.resolve_alias(name=name, alias=alias)
    if resolved is None:
        raise ModelUnavailableError("configured champion alias does not exist")
    loaded = registry.load_version(name=name, version=resolved.version)
    if not callable(getattr(loaded, "predict_proba", None)):
        raise ModelUnavailableError("champion does not provide calibrated probabilities")
    labels = tuple(str(label) for label in getattr(loaded, "classes_", ()))
    if not labels or len(set(labels)) != len(labels):
        raise ModelUnavailableError("champion label contract is unavailable or invalid")
    predictor = cast(ProbabilisticPredictor, loaded)
    smoke_text = preprocess_model_text(
        combine_ticket_text("Synthetic readiness check", "Local model contract verification."),
        settings.project_config.preprocessing,
    )
    try:
        smoke_prediction = np.asarray(predictor.predict((smoke_text,)), dtype=object)
        smoke_probabilities = np.asarray(predictor.predict_proba((smoke_text,)), dtype=np.float64)
    except Exception as exc:
        raise ModelUnavailableError("champion prediction contract check failed") from exc
    if (
        smoke_prediction.shape != (1,)
        or str(smoke_prediction[0]) not in labels
        or smoke_probabilities.shape != (1, len(labels))
        or not np.isfinite(smoke_probabilities).all()
        or (smoke_probabilities < 0.0).any()
        or (smoke_probabilities > 1.0).any()
        or not np.allclose(smoke_probabilities.sum(axis=1), 1.0, atol=1e-7)
    ):
        raise ModelUnavailableError("champion prediction contract check failed")
    contract: dict[str, object] = {
        "predictive_fields": ["subject", "body"],
        "derived_model_field": "model_text",
        "subject_max_length": settings.api_settings.maximum_subject_characters,
        "body_max_length": settings.api_settings.maximum_body_characters,
        "minimum_usable_characters": settings.api_settings.minimum_usable_characters,
        "probabilities": "calibrated",
    }
    return LoadedChampion(
        model=predictor,
        model_name=name,
        model_version=resolved.version,
        alias=alias,
        loaded_at=datetime.now(UTC),
        labels=labels,
        input_contract=contract,
    )
