"""SQLAlchemy engine/session lifecycle with schema-version readiness checks."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

EXPECTED_SCHEMA_REVISION = "20260807_0002"


class DatabaseSessionManager:
    """Own the bounded engine and typed SQLAlchemy 2 session factory."""

    def __init__(
        self,
        database_url: str,
        *,
        require_current_schema: bool = True,
    ) -> None:
        self.engine: Engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=5,
        )
        self.session_factory = sessionmaker(
            bind=self.engine,
            class_=Session,
            expire_on_commit=False,
        )
        self._require_current_schema = require_current_schema
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def connect(self) -> None:
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
                if self._require_current_schema:
                    revision = connection.execute(
                        text("SELECT version_num FROM alembic_version")
                    ).scalar_one_or_none()
                    if revision != EXPECTED_SCHEMA_REVISION:
                        raise RuntimeError("database schema is not at the expected revision")
        except (SQLAlchemyError, RuntimeError) as exc:
            self._ready = False
            raise ConnectionError("database readiness check failed") from exc
        self._ready = True

    def close(self) -> None:
        self._ready = False
        self.engine.dispose()
