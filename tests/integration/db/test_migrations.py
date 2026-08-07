"""Alembic upgrade/downgrade and optional PostgreSQL execution tests."""

from __future__ import annotations

import os
from io import StringIO
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.engine import make_url

from ticket_router.db.session import EXPECTED_SCHEMA_REVISION

EXPECTED_TABLES = {
    "alembic_version",
    "prediction_events",
    "feedback_events",
    "monitoring_runs",
    "retraining_runs",
}


def _alembic_config(connection: object) -> Config:
    config = Config("alembic.ini")
    config.attributes["connection"] = connection
    return config


def _upgrade(engine: Engine) -> None:
    with engine.begin() as connection:
        command.upgrade(_alembic_config(connection), "head")


def _downgrade(engine: Engine) -> None:
    with engine.begin() as connection:
        command.downgrade(_alembic_config(connection), "base")


def test_initial_migration_upgrade_and_downgrade_on_disposable_database(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{(tmp_path / 'migration.db').as_posix()}")
    try:
        _upgrade(engine)

        assert set(inspect(engine).get_table_names()) == EXPECTED_TABLES
        with engine.connect() as connection:
            assert connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar_one() == (EXPECTED_SCHEMA_REVISION)

        _downgrade(engine)
        assert inspect(engine).get_table_names() == ["alembic_version"]
    finally:
        engine.dispose()


def test_initial_migration_compiles_for_postgresql(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg://test_user:test_password@127.0.0.1/test_ticket_router",
    )
    output = StringIO()
    config = Config("alembic.ini", output_buffer=output)

    command.upgrade(config, "head", sql=True)

    sql = output.getvalue()
    assert "CREATE TABLE prediction_events" in sql
    assert "JSONB" in sql
    assert "UUID" in sql
    assert "TIMESTAMP WITH TIME ZONE" in sql


@pytest.mark.integration
def test_initial_migration_on_disposable_postgresql() -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not configured")
    parsed = make_url(database_url)
    database_name = parsed.database or ""
    if not (database_name.startswith("test_") or database_name.endswith("_test")):
        pytest.fail("TEST_DATABASE_URL must name an explicitly disposable test database")
    engine = create_engine(database_url)
    try:
        _downgrade(engine)
        _upgrade(engine)
        assert set(inspect(engine).get_table_names()) == EXPECTED_TABLES
    finally:
        _downgrade(engine)
        engine.dispose()
