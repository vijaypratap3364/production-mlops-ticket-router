"""SQLAlchemy 2 ORM mappings for operational and MLOps metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ticket_router.db.base import Base

JSON_DOCUMENT = JSON().with_variant(JSONB(), "postgresql")


class PredictionEventModel(Base):
    __tablename__ = "prediction_events"
    __table_args__ = (
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="confidence_range"),
        CheckConstraint("subject_length >= 0", name="subject_length_nonnegative"),
        CheckConstraint("body_length >= 0", name="body_length_nonnegative"),
        CheckConstraint("word_count >= 0", name="word_count_nonnegative"),
        CheckConstraint("latency_ms >= 0", name="latency_nonnegative"),
    )

    request_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    model_name: Mapped[str] = mapped_column(String(200))
    model_version: Mapped[str] = mapped_column(String(100), index=True)
    predicted_queue: Mapped[str] = mapped_column(String(200), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    top_k: Mapped[list[dict[str, Any]]] = mapped_column(JSON_DOCUMENT)
    subject_length: Mapped[int] = mapped_column(Integer)
    body_length: Mapped[int] = mapped_column(Integer)
    word_count: Mapped[int] = mapped_column(Integer)
    language_indicator: Mapped[str | None] = mapped_column(String(16), nullable=True)
    low_confidence: Mapped[bool] = mapped_column(Boolean, index=True)
    latency_ms: Mapped[float] = mapped_column(Float)
    redacted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    text_hash: Mapped[str] = mapped_column(String(64), index=True)
    text_hash_algorithm: Mapped[str] = mapped_column(String(32))
    request_metadata: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT)


class FeedbackEventModel(Base):
    __tablename__ = "feedback_events"
    __table_args__ = (UniqueConstraint("request_id"),)

    feedback_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    request_id: Mapped[UUID] = mapped_column(
        Uuid,
        ForeignKey("prediction_events.request_id", ondelete="RESTRICT"),
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    corrected_queue: Mapped[str] = mapped_column(String(200))
    accepted: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    comment: Mapped[str | None] = mapped_column(String(500), nullable=True)
    source: Mapped[str] = mapped_column(String(32))
    model_version: Mapped[str] = mapped_column(String(100), index=True)


class MonitoringRunModel(Base):
    __tablename__ = "monitoring_runs"
    __table_args__ = (
        CheckConstraint(
            "reference_period_end >= reference_period_start",
            name="reference_period_order",
        ),
        CheckConstraint(
            "current_period_end >= current_period_start",
            name="current_period_order",
        ),
    )

    run_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reference_period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    reference_period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    current_period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    current_period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    drift_status: Mapped[str] = mapped_column(String(50), index=True)
    report_paths: Mapped[list[str]] = mapped_column(JSON_DOCUMENT)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT)


class RetrainingRunModel(Base):
    __tablename__ = "retraining_runs"
    __table_args__ = (
        CheckConstraint(
            "source_data_period_end >= source_data_period_start",
            name="source_data_period_order",
        ),
    )

    run_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    trigger_reason: Mapped[str] = mapped_column(String(500))
    source_data_period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    source_data_period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(50), index=True)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    candidate_model_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    gate_results: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
