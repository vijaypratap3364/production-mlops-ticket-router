"""Registered champion contract verification without retraining or alias mutation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path

import joblib  # type: ignore[import-untyped]
import numpy as np

from ticket_router.api.metrics import APIMetrics
from ticket_router.api.model_loader import load_champion
from ticket_router.api.schemas import TicketRequest
from ticket_router.api.service import LoadedChampion, PredictionService
from ticket_router.benchmarking.config import (
    DEFAULT_BENCHMARK_CONFIG_PATH,
    BenchmarkConfig,
)
from ticket_router.benchmarking.contracts import ModelContractResult
from ticket_router.config import Settings
from ticket_router.data.split_manifest import SplitManifest
from ticket_router.db.repositories import InMemoryPredictionFeedbackRepository
from ticket_router.features.text import combine_ticket_text, preprocess_model_text

FIXED_CONTRACT_TICKETS = (
    TicketRequest(subject="Invoice question", body="Please explain this synthetic billing charge."),
    TicketRequest(subject="Network outage", body="The synthetic local service cannot connect."),
)


def validate_champion_contract(
    *,
    champion: LoadedChampion,
    settings: Settings,
    split_manifest: SplitManifest,
    samples: Sequence[TicketRequest] = FIXED_CONTRACT_TICKETS,
) -> ModelContractResult:
    """Verify the exact API/model/label/serialization contract for one immutable version."""
    if not samples:
        raise ValueError("at least one contract sample is required")
    schema = TicketRequest.model_json_schema()
    schema_fields = set(schema.get("properties", {}))
    labels = champion.labels
    label_mapping = split_manifest.label_mapping
    label_ids = sorted(label_mapping.values())
    texts = [
        preprocess_model_text(
            combine_ticket_text(ticket.subject or None, ticket.body or None),
            settings.project_config.preprocessing,
        )
        for ticket in samples
    ]
    predictions_first = np.asarray(champion.model.predict(texts), dtype=object)
    probabilities_first = np.asarray(champion.model.predict_proba(texts), dtype=np.float64)
    predictions_second = np.asarray(champion.model.predict(texts), dtype=object)
    probabilities_second = np.asarray(champion.model.predict_proba(texts), dtype=np.float64)
    deterministic = bool(
        np.array_equal(predictions_first, predictions_second)
        and np.array_equal(probabilities_first, probabilities_second)
    )

    buffer = BytesIO()
    joblib.dump(champion.model, buffer)
    serialized_size = buffer.tell()
    buffer.seek(0)
    reloaded = joblib.load(buffer)
    reloaded_predictions = np.asarray(reloaded.predict(texts), dtype=object)
    reloaded_probabilities = np.asarray(reloaded.predict_proba(texts), dtype=np.float64)

    store = InMemoryPredictionFeedbackRepository()
    service = PredictionService(
        champion=champion,
        api_settings=settings.api_settings,
        preprocessing=settings.project_config.preprocessing,
        store=store,
        metrics=APIMetrics(),
    )
    responses = service.predict_many(samples)
    expected_top_k = min(settings.api_settings.default_top_k, len(labels))
    top_k_valid = all(
        len(response.top_k) == expected_top_k
        and response.top_k[0].queue == response.predicted_queue
        and [item.confidence for item in response.top_k]
        == sorted((item.confidence for item in response.top_k), reverse=True)
        for response in responses
    )
    checks = {
        "accepted_input_schema": schema_fields == {"subject", "body", "metadata"},
        "predictive_input_contract": champion.input_contract.get("predictive_fields")
        == ["subject", "body"],
        "output_shape": predictions_first.shape == (len(samples),),
        "output_labels": set(str(value) for value in predictions_first).issubset(set(labels)),
        "probability_dimensions": probabilities_first.shape == (len(samples), len(labels)),
        "probabilities_finite_and_normalized": bool(
            np.isfinite(probabilities_first).all()
            and (probabilities_first >= 0.0).all()
            and (probabilities_first <= 1.0).all()
            and np.allclose(probabilities_first.sum(axis=1), 1.0, atol=1e-7)
        ),
        "label_mapping_matches_model": set(label_mapping) == set(labels)
        and label_ids == list(range(len(labels))),
        "top_k_contract": top_k_valid,
        "serializable_roundtrip": bool(
            np.array_equal(predictions_first, reloaded_predictions)
            and np.array_equal(probabilities_first, reloaded_probabilities)
        ),
        "deterministic_fixed_sample": deterministic,
    }
    return ModelContractResult(
        passed=all(checks.values()),
        checks=checks,
        labels=labels,
        label_mapping=label_mapping,
        probability_dimensions=(len(samples), len(labels)),
        top_k_size=expected_top_k,
        deterministic_fixed_sample=deterministic,
        serialized_roundtrip_size_bytes=serialized_size,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=DEFAULT_BENCHMARK_CONFIG_PATH,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = Settings.load(args.config)
    benchmark_config = BenchmarkConfig.load(args.benchmark_config)
    result = validate_champion_contract(
        champion=load_champion(settings),
        settings=settings,
        split_manifest=SplitManifest.read(benchmark_config.split_manifest_path),
    )
    print(json.dumps(result.model_dump(mode="json"), indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
