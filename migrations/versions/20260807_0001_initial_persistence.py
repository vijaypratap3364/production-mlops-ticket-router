"""Create privacy-safe prediction, feedback, monitoring, and retraining tables."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260807_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSON_DOCUMENT = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "prediction_events",
        sa.Column("request_id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_name", sa.String(length=200), nullable=False),
        sa.Column("model_version", sa.String(length=100), nullable=False),
        sa.Column("predicted_queue", sa.String(length=200), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("top_k", JSON_DOCUMENT, nullable=False),
        sa.Column("subject_length", sa.Integer(), nullable=False),
        sa.Column("body_length", sa.Integer(), nullable=False),
        sa.Column("word_count", sa.Integer(), nullable=False),
        sa.Column("language_indicator", sa.String(length=16), nullable=True),
        sa.Column("low_confidence", sa.Boolean(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("redacted_text", sa.Text(), nullable=True),
        sa.Column("text_hash", sa.String(length=64), nullable=False),
        sa.Column("text_hash_algorithm", sa.String(length=32), nullable=False),
        sa.Column("request_metadata", JSON_DOCUMENT, nullable=False),
        sa.CheckConstraint(
            "body_length >= 0",
            name=op.f("ck_prediction_events_body_length_nonnegative"),
        ),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name=op.f("ck_prediction_events_confidence_range"),
        ),
        sa.CheckConstraint(
            "latency_ms >= 0",
            name=op.f("ck_prediction_events_latency_nonnegative"),
        ),
        sa.CheckConstraint(
            "subject_length >= 0",
            name=op.f("ck_prediction_events_subject_length_nonnegative"),
        ),
        sa.CheckConstraint(
            "word_count >= 0",
            name=op.f("ck_prediction_events_word_count_nonnegative"),
        ),
        sa.PrimaryKeyConstraint("request_id", name=op.f("pk_prediction_events")),
    )
    op.create_index(op.f("ix_prediction_events_created_at"), "prediction_events", ["created_at"])
    op.create_index(
        op.f("ix_prediction_events_low_confidence"),
        "prediction_events",
        ["low_confidence"],
    )
    op.create_index(
        op.f("ix_prediction_events_model_version"),
        "prediction_events",
        ["model_version"],
    )
    op.create_index(
        op.f("ix_prediction_events_predicted_queue"),
        "prediction_events",
        ["predicted_queue"],
    )
    op.create_index(op.f("ix_prediction_events_text_hash"), "prediction_events", ["text_hash"])

    op.create_table(
        "feedback_events",
        sa.Column("feedback_id", sa.Uuid(), nullable=False),
        sa.Column("request_id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("corrected_queue", sa.String(length=200), nullable=False),
        sa.Column("accepted", sa.Boolean(), nullable=True),
        sa.Column("comment", sa.String(length=500), nullable=True),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("model_version", sa.String(length=100), nullable=False),
        sa.ForeignKeyConstraint(
            ["request_id"],
            ["prediction_events.request_id"],
            name=op.f("fk_feedback_events_request_id_prediction_events"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("feedback_id", name=op.f("pk_feedback_events")),
        sa.UniqueConstraint("request_id", name=op.f("uq_feedback_events_request_id")),
    )
    op.create_index(op.f("ix_feedback_events_created_at"), "feedback_events", ["created_at"])
    op.create_index(
        op.f("ix_feedback_events_model_version"),
        "feedback_events",
        ["model_version"],
    )
    op.create_index(op.f("ix_feedback_events_request_id"), "feedback_events", ["request_id"])

    op.create_table(
        "monitoring_runs",
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reference_period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reference_period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("current_period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("current_period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("drift_status", sa.String(length=50), nullable=False),
        sa.Column("report_paths", JSON_DOCUMENT, nullable=False),
        sa.Column("summary", JSON_DOCUMENT, nullable=False),
        sa.CheckConstraint(
            "current_period_end >= current_period_start",
            name=op.f("ck_monitoring_runs_current_period_order"),
        ),
        sa.CheckConstraint(
            "reference_period_end >= reference_period_start",
            name=op.f("ck_monitoring_runs_reference_period_order"),
        ),
        sa.PrimaryKeyConstraint("run_id", name=op.f("pk_monitoring_runs")),
    )
    op.create_index(op.f("ix_monitoring_runs_drift_status"), "monitoring_runs", ["drift_status"])
    op.create_index(op.f("ix_monitoring_runs_started_at"), "monitoring_runs", ["started_at"])

    op.create_table(
        "retraining_runs",
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("trigger_reason", sa.String(length=500), nullable=False),
        sa.Column("source_data_period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source_data_period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("mlflow_run_id", sa.String(length=100), nullable=True),
        sa.Column("candidate_model_version", sa.String(length=100), nullable=True),
        sa.Column("gate_results", JSON_DOCUMENT, nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "source_data_period_end >= source_data_period_start",
            name=op.f("ck_retraining_runs_source_data_period_order"),
        ),
        sa.PrimaryKeyConstraint("run_id", name=op.f("pk_retraining_runs")),
    )
    op.create_index(op.f("ix_retraining_runs_started_at"), "retraining_runs", ["started_at"])
    op.create_index(op.f("ix_retraining_runs_status"), "retraining_runs", ["status"])


def downgrade() -> None:
    op.drop_index(op.f("ix_retraining_runs_status"), table_name="retraining_runs")
    op.drop_index(op.f("ix_retraining_runs_started_at"), table_name="retraining_runs")
    op.drop_table("retraining_runs")
    op.drop_index(op.f("ix_monitoring_runs_started_at"), table_name="monitoring_runs")
    op.drop_index(op.f("ix_monitoring_runs_drift_status"), table_name="monitoring_runs")
    op.drop_table("monitoring_runs")
    op.drop_index(op.f("ix_feedback_events_request_id"), table_name="feedback_events")
    op.drop_index(op.f("ix_feedback_events_model_version"), table_name="feedback_events")
    op.drop_index(op.f("ix_feedback_events_created_at"), table_name="feedback_events")
    op.drop_table("feedback_events")
    op.drop_index(op.f("ix_prediction_events_text_hash"), table_name="prediction_events")
    op.drop_index(op.f("ix_prediction_events_predicted_queue"), table_name="prediction_events")
    op.drop_index(op.f("ix_prediction_events_model_version"), table_name="prediction_events")
    op.drop_index(op.f("ix_prediction_events_low_confidence"), table_name="prediction_events")
    op.drop_index(op.f("ix_prediction_events_created_at"), table_name="prediction_events")
    op.drop_table("prediction_events")
