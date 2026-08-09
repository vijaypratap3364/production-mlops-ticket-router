"""Repository behavior, constraints, and privacy tests on a disposable database."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from ticket_router.db.contracts import (
    FeedbackEvent,
    MonitoringRun,
    PredictionEvent,
    RetrainingRun,
)
from ticket_router.db.exceptions import (
    FeedbackAlreadyExistsError,
    PersistenceUnavailableError,
    PredictionNotFoundError,
)
from ticket_router.db.models import FeedbackEventModel
from ticket_router.db.repositories import (
    SQLAlchemyMonitoringDataRepository,
    SQLAlchemyMonitoringRunRepository,
    SQLAlchemyPredictionFeedbackRepository,
    SQLAlchemyRetrainingRunRepository,
)


def _prediction_event(*, request_id: str | None = None) -> PredictionEvent:
    return PredictionEvent(
        request_id=request_id or str(uuid4()),
        created_at=datetime(2026, 8, 7, 12, tzinfo=UTC),
        model_name="fixture-router",
        model_version="7",
        predicted_queue="Billing",
        confidence=0.8,
        top_k=(
            {"queue": "Billing", "confidence": 0.8},
            {"queue": "Technical", "confidence": 0.2},
        ),
        subject_length=7,
        body_length=12,
        word_count=5,
        combined_length=19,
        uppercase_ratio=0.1,
        digit_ratio=0.0,
        punctuation_ratio=0.05,
        url_count=0,
        email_marker_count=0,
        language_indicator=None,
        low_confidence=False,
        latency_ms=1.25,
        redacted_text=None,
        text_hash="a" * 64,
        text_hash_algorithm="hmac-sha256",
        request_metadata={"client_name": "unit-test"},
    )


def test_prediction_round_trip_is_idempotent_and_contains_no_raw_columns(
    sqlite_engine: Engine,
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    repository = SQLAlchemyPredictionFeedbackRepository(sqlite_session_factory)
    event = _prediction_event()

    repository.save_predictions((event,))
    repository.save_predictions((event,))
    stored = repository.get_prediction(event.request_id)
    columns = {column["name"] for column in inspect(sqlite_engine).get_columns("prediction_events")}

    assert stored == event
    assert "subject" not in columns
    assert "body" not in columns
    assert stored is not None
    assert stored.redacted_text is None
    assert stored.text_hash == "a" * 64


def test_feedback_requires_prediction_and_rejects_duplicates(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    repository = SQLAlchemyPredictionFeedbackRepository(sqlite_session_factory)
    prediction = _prediction_event()
    feedback = FeedbackEvent(
        feedback_id=str(uuid4()),
        request_id=prediction.request_id,
        corrected_queue="Technical",
        accepted=False,
        comment="Reviewed synthetic fixture",
        source="reviewer",
        created_at=datetime(2026, 8, 7, 13, tzinfo=UTC),
    )

    with pytest.raises(PredictionNotFoundError):
        repository.save_feedback(feedback)

    repository.save_predictions((prediction,))
    stored = repository.save_feedback(feedback)

    assert stored.model_version == prediction.model_version
    assert repository.get_feedback_for_request(prediction.request_id) == stored
    with pytest.raises(FeedbackAlreadyExistsError):
        repository.save_feedback(
            FeedbackEvent(
                feedback_id=str(uuid4()),
                request_id=prediction.request_id,
                corrected_queue="Billing",
                accepted=True,
                comment=None,
                source="user",
                created_at=datetime(2026, 8, 7, 14, tzinfo=UTC),
            )
        )


def test_database_foreign_key_rejects_orphan_feedback(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    with pytest.raises(IntegrityError), sqlite_session_factory.begin() as session:
        session.add(
            FeedbackEventModel(
                feedback_id=uuid4(),
                request_id=uuid4(),
                created_at=datetime.now(UTC),
                corrected_queue="Billing",
                accepted=None,
                comment=None,
                source="user",
                model_version="7",
            )
        )


def test_unapproved_request_metadata_is_rejected(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    repository = SQLAlchemyPredictionFeedbackRepository(sqlite_session_factory)
    event = _prediction_event()
    invalid = PredictionEvent(
        **{
            **event.__dict__,
            "request_metadata": {"authorization": "must-not-store"},
        }
    )

    with pytest.raises(PersistenceUnavailableError):
        repository.save_predictions((invalid,))


def test_monitoring_and_retraining_run_repositories(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    monitoring_repository = SQLAlchemyMonitoringRunRepository(sqlite_session_factory)
    retraining_repository = SQLAlchemyRetrainingRunRepository(sqlite_session_factory)
    started = datetime(2026, 8, 7, 10, tzinfo=UTC)
    monitoring = MonitoringRun(
        run_id=str(uuid4()),
        started_at=started,
        completed_at=started + timedelta(minutes=2),
        reference_period_start=started - timedelta(days=14),
        reference_period_end=started - timedelta(days=7),
        current_period_start=started - timedelta(days=7),
        current_period_end=started,
        drift_status="no_drift",
        report_paths=("artifacts/reports/drift.json",),
        summary={"records": 100},
    )
    retraining = RetrainingRun(
        run_id=str(uuid4()),
        trigger_reason="reviewed_drift",
        source_data_period_start=started - timedelta(days=30),
        source_data_period_end=started,
        status="candidate_registered",
        mlflow_run_id="a" * 32,
        candidate_model_version="8",
        gate_results={"passed": True},
        started_at=started,
        completed_at=started + timedelta(minutes=10),
    )

    monitoring_repository.save(monitoring)
    older_monitoring = MonitoringRun(
        **{
            **monitoring.__dict__,
            "run_id": str(uuid4()),
            "started_at": started - timedelta(days=1),
            "completed_at": started - timedelta(days=1) + timedelta(minutes=2),
            "drift_status": "healthy",
        }
    )
    monitoring_repository.save(older_monitoring)
    retraining_repository.save(retraining)
    older_retraining = RetrainingRun(
        **{
            **retraining.__dict__,
            "run_id": str(uuid4()),
            "started_at": started - timedelta(days=1),
            "completed_at": started - timedelta(days=1) + timedelta(minutes=10),
        }
    )
    retraining_repository.save(older_retraining)

    assert monitoring_repository.get(monitoring.run_id) == monitoring
    assert retraining_repository.get(retraining.run_id) == retraining
    assert monitoring_repository.get(str(uuid4())) is None
    assert retraining_repository.get(str(uuid4())) is None
    assert monitoring_repository.list_recent(limit=1) == (monitoring,)
    assert monitoring_repository.list_recent(limit=2) == (monitoring, older_monitoring)
    assert retraining_repository.list_recent(limit=1) == (retraining,)
    assert retraining_repository.list_recent(limit=2) == (retraining, older_retraining)
    with pytest.raises(ValueError, match="positive"):
        monitoring_repository.list_recent(limit=0)
    with pytest.raises(ValueError, match="positive"):
        retraining_repository.list_recent(limit=0)
    assert isinstance(UUID(monitoring.run_id), UUID)


def test_monitoring_data_queries_filter_time_model_version_and_join_feedback(
    sqlite_session_factory: sessionmaker[Session],
) -> None:
    predictions = SQLAlchemyPredictionFeedbackRepository(sqlite_session_factory)
    monitoring = SQLAlchemyMonitoringDataRepository(sqlite_session_factory)
    version_seven = _prediction_event()
    version_eight = PredictionEvent(
        **{
            **_prediction_event().__dict__,
            "request_id": str(uuid4()),
            "model_version": "8",
            "created_at": datetime(2026, 8, 7, 13, tzinfo=UTC),
        }
    )
    predictions.save_predictions((version_seven, version_eight))
    predictions.save_feedback(
        FeedbackEvent(
            feedback_id=str(uuid4()),
            request_id=version_seven.request_id,
            corrected_queue="Technical",
            accepted=False,
            comment=None,
            source="reviewer",
            created_at=datetime(2026, 8, 7, 14, tzinfo=UTC),
        )
    )

    current = monitoring.load_predictions(
        start=datetime(2026, 8, 7, 11, tzinfo=UTC),
        end=datetime(2026, 8, 7, 14, tzinfo=UTC),
        model_version="7",
    )
    labeled = monitoring.load_labeled_predictions(
        start=datetime(2026, 8, 7, 11, tzinfo=UTC),
        end=datetime(2026, 8, 7, 14, tzinfo=UTC),
        model_version="7",
    )

    assert len(current) == 1
    assert current[0].model_version == "7"
    assert current[0].combined_length == 19
    assert len(labeled) == 1
    assert labeled[0].corrected_queue == "Technical"
    assert labeled[0].model_version == "7"
