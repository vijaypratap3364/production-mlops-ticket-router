"""Disposable SQLite repository fixture for fast SQLAlchemy contract tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from ticket_router.db.base import Base


@pytest.fixture
def sqlite_engine(tmp_path: Path) -> Iterator[Engine]:
    engine = create_engine(f"sqlite:///{(tmp_path / 'repository.db').as_posix()}")

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection: object, _: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def sqlite_session_factory(sqlite_engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=sqlite_engine, class_=Session, expire_on_commit=False)
