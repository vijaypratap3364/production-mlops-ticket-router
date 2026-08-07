"""Optional SQLAlchemy connectivity probe used by API readiness."""

from __future__ import annotations

from typing import Protocol

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError


class DatabaseProbe(Protocol):
    @property
    def ready(self) -> bool: ...

    def connect(self) -> None: ...

    def close(self) -> None: ...


class SQLAlchemyDatabaseProbe:
    """Own an engine and verify connectivity without logging its potentially secret URL."""

    def __init__(self, database_url: str) -> None:
        self._engine: Engine = create_engine(database_url, pool_pre_ping=True)
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def connect(self) -> None:
        try:
            with self._engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except SQLAlchemyError as exc:
            self._ready = False
            raise ConnectionError("database connectivity check failed") from exc
        self._ready = True

    def close(self) -> None:
        self._ready = False
        self._engine.dispose()
