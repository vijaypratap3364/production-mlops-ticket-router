"""Structured, local-first application logging."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.stdlib import BoundLogger


def configure_logging(log_level: str = "INFO", *, json_logs: bool = True) -> None:
    """Configure standard-library and structlog output for an application process."""
    level = log_level.upper()
    if level not in logging.getLevelNamesMapping():
        raise ValueError(f"unknown log level: {log_level}")

    renderer: Any
    if json_logs:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
        force=True,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> BoundLogger:
    """Return a named structured logger."""
    return structlog.stdlib.get_logger(name)
