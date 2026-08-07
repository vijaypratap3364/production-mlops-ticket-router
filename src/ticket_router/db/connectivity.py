"""Database connectivity protocol and schema-aware SQLAlchemy implementation."""

from __future__ import annotations

from typing import Protocol

from sqlalchemy.orm import Session, sessionmaker

from ticket_router.db.session import DatabaseSessionManager


class DatabaseProbe(Protocol):
    @property
    def ready(self) -> bool: ...

    @property
    def session_factory(self) -> sessionmaker[Session]: ...

    def connect(self) -> None: ...

    def close(self) -> None: ...


class SQLAlchemyDatabaseProbe(DatabaseSessionManager):
    """Backward-compatible API name for the schema-aware session manager."""

    def __init__(self, database_url: str) -> None:
        super().__init__(database_url, require_current_schema=True)
