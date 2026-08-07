"""Persistence-domain exceptions independent of HTTP and SQLAlchemy details."""

from __future__ import annotations


class PersistenceError(RuntimeError):
    """Base error for stable persistence behavior."""


class PersistenceUnavailableError(PersistenceError):
    """The configured persistence adapter could not complete an operation."""


class PredictionNotFoundError(PersistenceError):
    """Feedback referenced a prediction that does not exist."""


class FeedbackAlreadyExistsError(PersistenceError):
    """One-feedback-per-prediction policy rejected a duplicate."""
