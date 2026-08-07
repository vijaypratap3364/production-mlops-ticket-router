"""SQLAlchemy 2 and in-memory implementations of persistence repositories."""

from __future__ import annotations

from datetime import UTC, datetime
from threading import Lock
from typing import cast
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from ticket_router.db.contracts import (
    FeedbackEvent,
    MonitoringRun,
    PredictionEvent,
    RetrainingRun,
    StoredFeedback,
)
from ticket_router.db.exceptions import (
    FeedbackAlreadyExistsError,
    PersistenceUnavailableError,
    PredictionNotFoundError,
)
from ticket_router.db.models import (
    FeedbackEventModel,
    MonitoringRunModel,
    PredictionEventModel,
    RetrainingRunModel,
)
from ticket_router.monitoring.contracts import CurrentPrediction, LabeledPrediction

APPROVED_REQUEST_METADATA_FIELDS = frozenset({"client_name", "correlation_id"})


class InMemoryPredictionFeedbackRepository:
    """Thread-safe non-durable adapter for database-free local API tests."""

    def __init__(self) -> None:
        self._predictions: dict[str, PredictionEvent] = {}
        self._feedback_by_request: dict[str, StoredFeedback] = {}
        self._available = True
        self._lock = Lock()

    def save_predictions(self, events: tuple[PredictionEvent, ...]) -> None:
        with self._lock:
            self._require_available()
            for event in events:
                self._predictions.setdefault(event.request_id, event)

    def get_prediction(self, request_id: str) -> PredictionEvent | None:
        with self._lock:
            self._require_available()
            return self._predictions.get(request_id)

    def save_feedback(self, event: FeedbackEvent) -> StoredFeedback:
        with self._lock:
            self._require_available()
            prediction = self._predictions.get(event.request_id)
            if prediction is None:
                raise PredictionNotFoundError(event.request_id)
            if event.request_id in self._feedback_by_request:
                raise FeedbackAlreadyExistsError(event.request_id)
            stored = StoredFeedback(
                feedback_id=event.feedback_id,
                request_id=event.request_id,
                corrected_queue=event.corrected_queue,
                accepted=event.accepted,
                comment=event.comment,
                source=event.source,
                model_version=prediction.model_version,
                created_at=event.created_at,
            )
            self._feedback_by_request[event.request_id] = stored
            return stored

    def get_feedback_for_request(self, request_id: str) -> StoredFeedback | None:
        with self._lock:
            self._require_available()
            return self._feedback_by_request.get(request_id)

    def close(self) -> None:
        with self._lock:
            self._available = False

    def _require_available(self) -> None:
        if not self._available:
            raise PersistenceUnavailableError("persistence adapter is closed")


class SQLAlchemyPredictionFeedbackRepository:
    """Transaction-scoped prediction and one-feedback-per-request persistence."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def save_predictions(self, events: tuple[PredictionEvent, ...]) -> None:
        try:
            with self._session_factory.begin() as session:
                for event in events:
                    request_uuid = _parse_uuid(event.request_id)
                    if session.get(PredictionEventModel, request_uuid) is None:
                        session.add(_prediction_model(event, request_uuid=request_uuid))
        except (SQLAlchemyError, ValueError) as exc:
            raise PersistenceUnavailableError("prediction persistence failed") from exc

    def get_prediction(self, request_id: str) -> PredictionEvent | None:
        try:
            request_uuid = _parse_uuid(request_id)
        except ValueError:
            return None
        try:
            with self._session_factory() as session:
                model = session.get(PredictionEventModel, request_uuid)
                return _prediction_record(model) if model is not None else None
        except SQLAlchemyError as exc:
            raise PersistenceUnavailableError("prediction lookup failed") from exc

    def save_feedback(self, event: FeedbackEvent) -> StoredFeedback:
        try:
            request_uuid = _parse_uuid(event.request_id)
            feedback_uuid = _parse_uuid(event.feedback_id)
        except ValueError as exc:
            raise PredictionNotFoundError(event.request_id) from exc
        try:
            with self._session_factory.begin() as session:
                prediction = session.get(PredictionEventModel, request_uuid)
                if prediction is None:
                    raise PredictionNotFoundError(event.request_id)
                existing = session.scalar(
                    select(FeedbackEventModel).where(FeedbackEventModel.request_id == request_uuid)
                )
                if existing is not None:
                    raise FeedbackAlreadyExistsError(event.request_id)
                model = FeedbackEventModel(
                    feedback_id=feedback_uuid,
                    request_id=request_uuid,
                    created_at=_require_utc(event.created_at),
                    corrected_queue=event.corrected_queue,
                    accepted=event.accepted,
                    comment=event.comment,
                    source=event.source,
                    model_version=prediction.model_version,
                )
                session.add(model)
            return _stored_feedback(model)
        except (FeedbackAlreadyExistsError, PredictionNotFoundError):
            raise
        except IntegrityError as exc:
            raise FeedbackAlreadyExistsError(event.request_id) from exc
        except SQLAlchemyError as exc:
            raise PersistenceUnavailableError("feedback persistence failed") from exc

    def get_feedback_for_request(self, request_id: str) -> StoredFeedback | None:
        try:
            request_uuid = _parse_uuid(request_id)
        except ValueError:
            return None
        try:
            with self._session_factory() as session:
                model = session.scalar(
                    select(FeedbackEventModel).where(FeedbackEventModel.request_id == request_uuid)
                )
                return _stored_feedback(model) if model is not None else None
        except SQLAlchemyError as exc:
            raise PersistenceUnavailableError("feedback lookup failed") from exc

    def close(self) -> None:
        """The application lifespan owns and disposes the shared engine."""


class SQLAlchemyMonitoringRunRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def save(self, run: MonitoringRun) -> None:
        model = MonitoringRunModel(
            run_id=_parse_uuid(run.run_id),
            started_at=_require_utc(run.started_at),
            completed_at=_optional_utc(run.completed_at),
            reference_period_start=_require_utc(run.reference_period_start),
            reference_period_end=_require_utc(run.reference_period_end),
            current_period_start=_require_utc(run.current_period_start),
            current_period_end=_require_utc(run.current_period_end),
            drift_status=run.drift_status,
            report_paths=list(run.report_paths),
            summary=run.summary,
        )
        try:
            with self._session_factory.begin() as session:
                session.merge(model)
        except SQLAlchemyError as exc:
            raise PersistenceUnavailableError("monitoring-run persistence failed") from exc

    def get(self, run_id: str) -> MonitoringRun | None:
        try:
            with self._session_factory() as session:
                model = session.get(MonitoringRunModel, _parse_uuid(run_id))
                return _monitoring_record(model) if model is not None else None
        except SQLAlchemyError as exc:
            raise PersistenceUnavailableError("monitoring-run lookup failed") from exc


class SQLAlchemyMonitoringDataRepository:
    """Read bounded privacy-safe prediction windows and their delayed labels."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def load_predictions(
        self,
        *,
        start: datetime,
        end: datetime,
        model_version: str | None = None,
    ) -> tuple[CurrentPrediction, ...]:
        statement = (
            select(PredictionEventModel)
            .where(PredictionEventModel.created_at >= _require_utc(start))
            .where(PredictionEventModel.created_at < _require_utc(end))
            .where(PredictionEventModel.combined_length > 0)
            .order_by(PredictionEventModel.created_at, PredictionEventModel.request_id)
        )
        if model_version is not None:
            statement = statement.where(PredictionEventModel.model_version == model_version)
        try:
            with self._session_factory() as session:
                models = session.scalars(statement).all()
        except SQLAlchemyError as exc:
            raise PersistenceUnavailableError("monitoring prediction query failed") from exc
        return tuple(_current_prediction(model) for model in models)

    def load_labeled_predictions(
        self,
        *,
        start: datetime,
        end: datetime,
        model_version: str | None = None,
    ) -> tuple[LabeledPrediction, ...]:
        statement = (
            select(PredictionEventModel, FeedbackEventModel)
            .join(
                FeedbackEventModel, FeedbackEventModel.request_id == PredictionEventModel.request_id
            )
            .where(PredictionEventModel.created_at >= _require_utc(start))
            .where(PredictionEventModel.created_at < _require_utc(end))
            .where(PredictionEventModel.combined_length > 0)
            .order_by(PredictionEventModel.created_at, PredictionEventModel.request_id)
        )
        if model_version is not None:
            statement = statement.where(PredictionEventModel.model_version == model_version)
        try:
            with self._session_factory() as session:
                rows = session.execute(statement).all()
        except SQLAlchemyError as exc:
            raise PersistenceUnavailableError("monitoring feedback query failed") from exc
        return tuple(
            LabeledPrediction(
                predicted_queue=prediction.predicted_queue,
                corrected_queue=feedback.corrected_queue,
                confidence=prediction.confidence,
                accepted=feedback.accepted,
                model_version=prediction.model_version,
            )
            for prediction, feedback in rows
        )


class SQLAlchemyRetrainingRunRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def save(self, run: RetrainingRun) -> None:
        model = RetrainingRunModel(
            run_id=_parse_uuid(run.run_id),
            trigger_reason=run.trigger_reason,
            source_data_period_start=_require_utc(run.source_data_period_start),
            source_data_period_end=_require_utc(run.source_data_period_end),
            status=run.status,
            mlflow_run_id=run.mlflow_run_id,
            candidate_model_version=run.candidate_model_version,
            gate_results=run.gate_results,
            started_at=_require_utc(run.started_at),
            completed_at=_optional_utc(run.completed_at),
        )
        try:
            with self._session_factory.begin() as session:
                session.merge(model)
        except SQLAlchemyError as exc:
            raise PersistenceUnavailableError("retraining-run persistence failed") from exc

    def get(self, run_id: str) -> RetrainingRun | None:
        try:
            with self._session_factory() as session:
                model = session.get(RetrainingRunModel, _parse_uuid(run_id))
                return _retraining_record(model) if model is not None else None
        except SQLAlchemyError as exc:
            raise PersistenceUnavailableError("retraining-run lookup failed") from exc


def _prediction_model(event: PredictionEvent, *, request_uuid: UUID) -> PredictionEventModel:
    if not set(event.request_metadata).issubset(APPROVED_REQUEST_METADATA_FIELDS):
        raise ValueError("request metadata contains an unapproved field")
    return PredictionEventModel(
        request_id=request_uuid,
        created_at=_require_utc(event.created_at),
        model_name=event.model_name,
        model_version=event.model_version,
        predicted_queue=event.predicted_queue,
        confidence=event.confidence,
        top_k=list(event.top_k),
        subject_length=event.subject_length,
        body_length=event.body_length,
        word_count=event.word_count,
        combined_length=event.combined_length,
        uppercase_ratio=event.uppercase_ratio,
        digit_ratio=event.digit_ratio,
        punctuation_ratio=event.punctuation_ratio,
        url_count=event.url_count,
        email_marker_count=event.email_marker_count,
        language_indicator=event.language_indicator,
        low_confidence=event.low_confidence,
        latency_ms=event.latency_ms,
        redacted_text=event.redacted_text,
        text_hash=event.text_hash,
        text_hash_algorithm=event.text_hash_algorithm,
        request_metadata=event.request_metadata,
    )


def _prediction_record(model: PredictionEventModel) -> PredictionEvent:
    top_k = cast(
        tuple[dict[str, str | float], ...],
        tuple(
            {"queue": str(item["queue"]), "confidence": float(item["confidence"])}
            for item in model.top_k
        ),
    )
    return PredictionEvent(
        request_id=str(model.request_id),
        created_at=_as_utc(model.created_at),
        model_name=model.model_name,
        model_version=model.model_version,
        predicted_queue=model.predicted_queue,
        confidence=model.confidence,
        top_k=top_k,
        subject_length=model.subject_length,
        body_length=model.body_length,
        word_count=model.word_count,
        combined_length=model.combined_length,
        uppercase_ratio=model.uppercase_ratio,
        digit_ratio=model.digit_ratio,
        punctuation_ratio=model.punctuation_ratio,
        url_count=model.url_count,
        email_marker_count=model.email_marker_count,
        language_indicator=model.language_indicator,
        low_confidence=model.low_confidence,
        latency_ms=model.latency_ms,
        redacted_text=model.redacted_text,
        text_hash=model.text_hash,
        text_hash_algorithm=model.text_hash_algorithm,
        request_metadata={key: str(value) for key, value in model.request_metadata.items()},
    )


def _stored_feedback(model: FeedbackEventModel) -> StoredFeedback:
    return StoredFeedback(
        feedback_id=str(model.feedback_id),
        request_id=str(model.request_id),
        corrected_queue=model.corrected_queue,
        accepted=model.accepted,
        comment=model.comment,
        source=model.source,
        model_version=model.model_version,
        created_at=_as_utc(model.created_at),
    )


def _current_prediction(model: PredictionEventModel) -> CurrentPrediction:
    return CurrentPrediction(
        request_id=str(model.request_id),
        created_at=_as_utc(model.created_at),
        model_version=model.model_version,
        predicted_queue=model.predicted_queue,
        prediction_confidence=model.confidence,
        low_confidence=model.low_confidence,
        subject_length=model.subject_length,
        body_length=model.body_length,
        combined_length=model.combined_length,
        word_count=model.word_count,
        uppercase_ratio=model.uppercase_ratio,
        digit_ratio=model.digit_ratio,
        punctuation_ratio=model.punctuation_ratio,
        url_count=model.url_count,
        email_marker_count=model.email_marker_count,
    )


def _monitoring_record(model: MonitoringRunModel) -> MonitoringRun:
    return MonitoringRun(
        run_id=str(model.run_id),
        started_at=_as_utc(model.started_at),
        completed_at=_optional_as_utc(model.completed_at),
        reference_period_start=_as_utc(model.reference_period_start),
        reference_period_end=_as_utc(model.reference_period_end),
        current_period_start=_as_utc(model.current_period_start),
        current_period_end=_as_utc(model.current_period_end),
        drift_status=model.drift_status,
        report_paths=tuple(model.report_paths),
        summary=cast(dict[str, object], model.summary),
    )


def _retraining_record(model: RetrainingRunModel) -> RetrainingRun:
    return RetrainingRun(
        run_id=str(model.run_id),
        trigger_reason=model.trigger_reason,
        source_data_period_start=_as_utc(model.source_data_period_start),
        source_data_period_end=_as_utc(model.source_data_period_end),
        status=model.status,
        mlflow_run_id=model.mlflow_run_id,
        candidate_model_version=model.candidate_model_version,
        gate_results=cast(dict[str, object], model.gate_results),
        started_at=_as_utc(model.started_at),
        completed_at=_optional_as_utc(model.completed_at),
    )


def _parse_uuid(value: str) -> UUID:
    return UUID(value)


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("database timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _optional_utc(value: datetime | None) -> datetime | None:
    return _require_utc(value) if value is not None else None


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _optional_as_utc(value: datetime | None) -> datetime | None:
    return _as_utc(value) if value is not None else None
