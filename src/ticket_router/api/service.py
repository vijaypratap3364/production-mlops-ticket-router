"""Framework-light prediction and feedback application service."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Protocol, cast
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from ticket_router.api.errors import (
    DatabaseUnavailableError,
    DuplicateFeedbackError,
    InvalidFeedbackLabelError,
    PredictionFailureError,
    RequestConstraintError,
    UnknownPredictionError,
)
from ticket_router.api.metrics import APIMetrics
from ticket_router.api.schemas import (
    ClassProbability,
    FeedbackRequest,
    FeedbackResponse,
    PredictionResponse,
    TicketRequest,
)
from ticket_router.config import APISettings, TextPreprocessingSettings
from ticket_router.data.normalize import combine_ticket_text
from ticket_router.db.contracts import (
    FeedbackEvent,
    PredictionEvent,
    PredictionFeedbackRepository,
)
from ticket_router.db.exceptions import (
    FeedbackAlreadyExistsError,
    PersistenceUnavailableError,
    PredictionNotFoundError,
)
from ticket_router.db.privacy import text_fingerprint
from ticket_router.features.text import preprocess_model_text
from ticket_router.logging_config import get_logger
from ticket_router.monitoring.features import derive_text_monitoring_features

LOW_CONFIDENCE_WARNING = "Prediction confidence is below the configured review threshold."


class ProbabilisticPredictor(Protocol):
    """Serving contract required from a champion model."""

    classes_: object

    def predict(self, values: Sequence[str]) -> object: ...

    def predict_proba(self, values: Sequence[str]) -> object: ...


@dataclass(frozen=True)
class LoadedChampion:
    model: ProbabilisticPredictor
    model_name: str
    model_version: str
    alias: str
    loaded_at: datetime
    labels: tuple[str, ...]
    input_contract: dict[str, object]
    training_data_hash: str | None = None
    macro_f1: float | None = None
    model_size_bytes: int | None = None
    created_at: datetime | None = None


class PredictionService:
    """Use the loaded champion without fitting or mutating it."""

    def __init__(
        self,
        *,
        champion: LoadedChampion,
        api_settings: APISettings,
        preprocessing: TextPreprocessingSettings,
        store: PredictionFeedbackRepository,
        metrics: APIMetrics,
        store_redacted_text: bool = False,
        input_hmac_secret: str | None = None,
    ) -> None:
        self.champion = champion
        self.api_settings = api_settings
        self._preprocessing = preprocessing
        self._store = store
        self._metrics = metrics
        self._store_redacted_text = store_redacted_text
        self._input_hmac_secret = input_hmac_secret

    def predict_one(self, ticket: TicketRequest, *, request_id: str) -> PredictionResponse:
        return self.predict_many((ticket,), request_ids=(request_id,))[0]

    def predict_many(
        self,
        tickets: Sequence[TicketRequest],
        *,
        request_ids: tuple[str, ...] | None = None,
    ) -> list[PredictionResponse]:
        if not tickets:
            raise RequestConstraintError("prediction batch cannot be empty")
        if len(tickets) > self.api_settings.maximum_batch_size:
            raise RequestConstraintError("prediction batch exceeds configured maximum")
        ids = request_ids or tuple(str(uuid4()) for _ in tickets)
        if len(ids) != len(tickets):
            raise ValueError("request IDs must align with prediction inputs")
        texts = [self._model_text(ticket) for ticket in tickets]
        started = perf_counter()
        try:
            predictions = np.asarray(self.champion.model.predict(texts), dtype=object)
            probabilities = np.asarray(self.champion.model.predict_proba(texts), dtype=np.float64)
        except Exception as exc:
            raise PredictionFailureError("champion inference failed") from exc
        elapsed = perf_counter() - started
        self._metrics.prediction_latency.observe(elapsed)
        self._validate_model_outputs(predictions, probabilities, expected_rows=len(tickets))
        timestamp = datetime.now(UTC)
        responses = [
            self._response(
                request_id=request_id,
                predicted_queue=str(predictions[index]),
                probabilities=probabilities[index],
                timestamp=timestamp,
            )
            for index, request_id in enumerate(ids)
        ]
        events = tuple(
            self._prediction_event(
                response=response,
                ticket=ticket,
                model_text=text,
                latency_ms=elapsed * 1000.0 / len(tickets),
            )
            for response, ticket, text in zip(responses, tickets, texts, strict=True)
        )
        try:
            self._store.save_predictions(events)
        except Exception as exc:
            self._metrics.persistence_failures.labels(operation="prediction").inc()
            get_logger(__name__).error(
                "prediction_persistence_failed",
                error_type=type(exc).__name__,
                event_count=len(events),
            )
        self._metrics.batch_size.observe(len(tickets))
        self._metrics.predictions.inc(len(responses))
        for response in responses:
            self._metrics.predicted_labels.labels(queue=response.predicted_queue).inc()
            if response.warning is not None:
                self._metrics.low_confidence.inc()
        return responses

    def record_feedback(self, feedback: FeedbackRequest) -> FeedbackResponse:
        corrected_queue = feedback.corrected_queue.strip()
        if corrected_queue not in set(self.champion.labels):
            raise InvalidFeedbackLabelError(corrected_queue)
        recorded_at = datetime.now(UTC)
        event = FeedbackEvent(
            feedback_id=str(uuid4()),
            request_id=feedback.request_id,
            corrected_queue=corrected_queue,
            accepted=feedback.accepted,
            comment=feedback.comment,
            source=feedback.source,
            created_at=recorded_at,
        )
        try:
            stored = self._store.save_feedback(event)
        except PredictionNotFoundError as exc:
            raise UnknownPredictionError(feedback.request_id) from exc
        except FeedbackAlreadyExistsError as exc:
            raise DuplicateFeedbackError(feedback.request_id) from exc
        except PersistenceUnavailableError as exc:
            self._metrics.persistence_failures.labels(operation="feedback").inc()
            raise DatabaseUnavailableError("feedback persistence failed") from exc
        return FeedbackResponse(
            request_id=stored.request_id,
            feedback_id=stored.feedback_id,
            corrected_queue=stored.corrected_queue,
            recorded_at=stored.created_at,
        )

    def _prediction_event(
        self,
        *,
        response: PredictionResponse,
        ticket: TicketRequest,
        model_text: str,
        latency_ms: float,
    ) -> PredictionEvent:
        digest, algorithm = text_fingerprint(
            model_text,
            hmac_secret=self._input_hmac_secret,
        )
        top_k: tuple[dict[str, str | float], ...] = tuple(
            {"queue": item.queue, "confidence": item.confidence} for item in response.top_k
        )
        metadata = ticket.metadata.model_dump(exclude_none=True) if ticket.metadata else {}
        monitoring = derive_text_monitoring_features(
            subject=ticket.subject,
            body=ticket.body,
            model_text=model_text,
        )
        return PredictionEvent(
            request_id=response.request_id,
            created_at=response.prediction_timestamp,
            predicted_queue=response.predicted_queue,
            confidence=response.confidence,
            top_k=top_k,
            model_name=response.model_name,
            model_version=response.model_version,
            subject_length=monitoring.subject_length,
            body_length=monitoring.body_length,
            word_count=monitoring.word_count,
            combined_length=monitoring.combined_length,
            uppercase_ratio=monitoring.uppercase_ratio,
            digit_ratio=monitoring.digit_ratio,
            punctuation_ratio=monitoring.punctuation_ratio,
            url_count=monitoring.url_count,
            email_marker_count=monitoring.email_marker_count,
            language_indicator=None,
            low_confidence=response.warning is not None,
            latency_ms=latency_ms,
            redacted_text=model_text if self._store_redacted_text else None,
            text_hash=digest,
            text_hash_algorithm=algorithm,
            request_metadata=metadata,
        )

    def _model_text(self, ticket: TicketRequest) -> str:
        limits = self.api_settings
        if len(ticket.subject) > limits.maximum_subject_characters:
            raise RequestConstraintError("subject exceeds configured maximum")
        if len(ticket.body) > limits.maximum_body_characters:
            raise RequestConstraintError("body exceeds configured maximum")
        usable_length = len(ticket.subject.strip()) + len(ticket.body.strip())
        if usable_length < limits.minimum_usable_characters:
            raise RequestConstraintError("ticket does not contain enough usable text")
        combined = combine_ticket_text(ticket.subject or None, ticket.body or None)
        return preprocess_model_text(combined, self._preprocessing)

    def _response(
        self,
        *,
        request_id: str,
        predicted_queue: str,
        probabilities: NDArray[np.float64],
        timestamp: datetime,
    ) -> PredictionResponse:
        ranked = sorted(
            zip(self.champion.labels, probabilities, strict=True),
            key=lambda item: (-float(item[1]), item[0]),
        )
        top_k = [
            ClassProbability(queue=label, confidence=float(probability))
            for label, probability in ranked[: self.api_settings.default_top_k]
        ]
        if predicted_queue not in self.champion.labels or predicted_queue != top_k[0].queue:
            raise PredictionFailureError("champion output does not match probability ranking")
        confidence = top_k[0].confidence
        warning = (
            LOW_CONFIDENCE_WARNING
            if confidence < self.api_settings.confidence_warning_threshold
            else None
        )
        return PredictionResponse(
            request_id=request_id,
            predicted_queue=predicted_queue,
            confidence=confidence,
            top_k=top_k,
            model_name=self.champion.model_name,
            model_version=self.champion.model_version,
            prediction_timestamp=timestamp,
            warning=warning,
        )

    def _validate_model_outputs(
        self,
        predictions: NDArray[np.object_],
        probabilities: NDArray[np.float64],
        *,
        expected_rows: int,
    ) -> None:
        if predictions.shape != (expected_rows,):
            raise PredictionFailureError("champion returned an invalid prediction shape")
        expected_probability_shape = (expected_rows, len(self.champion.labels))
        if probabilities.shape != expected_probability_shape:
            raise PredictionFailureError("champion returned an invalid probability shape")
        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0.0).any()
            or (probabilities > 1.0).any()
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-7)
        ):
            raise PredictionFailureError("champion returned invalid probabilities")
        observed = set(cast(list[str], predictions.tolist()))
        if not observed.issubset(set(self.champion.labels)):
            raise PredictionFailureError("champion returned an unknown queue")
