"""Add privacy-safe text-distribution features to prediction events."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260807_0002"
down_revision: str | None = "20260807_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("prediction_events") as batch:
        batch.add_column(
            sa.Column("combined_length", sa.Integer(), nullable=False, server_default="0")
        )
        batch.add_column(
            sa.Column("uppercase_ratio", sa.Float(), nullable=False, server_default="0")
        )
        batch.add_column(sa.Column("digit_ratio", sa.Float(), nullable=False, server_default="0"))
        batch.add_column(
            sa.Column("punctuation_ratio", sa.Float(), nullable=False, server_default="0")
        )
        batch.add_column(sa.Column("url_count", sa.Integer(), nullable=False, server_default="0"))
        batch.add_column(
            sa.Column("email_marker_count", sa.Integer(), nullable=False, server_default="0")
        )
        batch.create_check_constraint("combined_length_nonnegative", "combined_length >= 0")
        batch.create_check_constraint(
            "uppercase_ratio_range", "uppercase_ratio >= 0 AND uppercase_ratio <= 1"
        )
        batch.create_check_constraint("digit_ratio_range", "digit_ratio >= 0 AND digit_ratio <= 1")
        batch.create_check_constraint(
            "punctuation_ratio_range", "punctuation_ratio >= 0 AND punctuation_ratio <= 1"
        )
        batch.create_check_constraint("url_count_nonnegative", "url_count >= 0")
        batch.create_check_constraint("email_marker_count_nonnegative", "email_marker_count >= 0")


def downgrade() -> None:
    with op.batch_alter_table("prediction_events") as batch:
        batch.drop_constraint("email_marker_count_nonnegative", type_="check")
        batch.drop_constraint("url_count_nonnegative", type_="check")
        batch.drop_constraint("punctuation_ratio_range", type_="check")
        batch.drop_constraint("digit_ratio_range", type_="check")
        batch.drop_constraint("uppercase_ratio_range", type_="check")
        batch.drop_constraint("combined_length_nonnegative", type_="check")
        batch.drop_column("email_marker_count")
        batch.drop_column("url_count")
        batch.drop_column("punctuation_ratio")
        batch.drop_column("digit_ratio")
        batch.drop_column("uppercase_ratio")
        batch.drop_column("combined_length")
