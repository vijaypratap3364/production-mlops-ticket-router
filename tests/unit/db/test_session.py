"""Database session readiness and lifecycle behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text

from ticket_router.db.session import EXPECTED_SCHEMA_REVISION, DatabaseSessionManager


def _assert_readiness(manager: DatabaseSessionManager, *, expected: bool) -> None:
    assert manager.ready is expected


def test_session_manager_connects_without_schema_requirement_and_closes(tmp_path: Path) -> None:
    manager = DatabaseSessionManager(
        f"sqlite:///{(tmp_path / 'unmigrated.db').as_posix()}",
        require_current_schema=False,
    )

    _assert_readiness(manager, expected=False)
    manager.connect()
    _assert_readiness(manager, expected=True)
    with manager.session_factory() as session:
        assert session.execute(text("SELECT 1")).scalar_one() == 1
    manager.close()
    _assert_readiness(manager, expected=False)


def test_session_manager_requires_expected_alembic_revision(tmp_path: Path) -> None:
    manager = DatabaseSessionManager(
        f"sqlite:///{(tmp_path / 'schema.db').as_posix()}",
        require_current_schema=True,
    )
    with manager.engine.begin() as connection:
        connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32))"))
        connection.execute(
            text("INSERT INTO alembic_version (version_num) VALUES (:revision)"),
            {"revision": "outdated"},
        )

    with pytest.raises(ConnectionError, match="readiness check failed"):
        manager.connect()
    _assert_readiness(manager, expected=False)

    with manager.engine.begin() as connection:
        connection.execute(
            text("UPDATE alembic_version SET version_num = :revision"),
            {"revision": EXPECTED_SCHEMA_REVISION},
        )
    manager.connect()
    _assert_readiness(manager, expected=True)
    manager.close()
