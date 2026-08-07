"""Stable application errors mapped to sanitized HTTP responses."""

from __future__ import annotations


class APIServiceError(RuntimeError):
    """Base class for expected prediction-service failures."""

    code = "service_error"
    status_code = 500
    public_message = "The request could not be completed."


class ModelUnavailableError(APIServiceError):
    code = "model_unavailable"
    status_code = 503
    public_message = "The champion model is unavailable."


class DatabaseUnavailableError(APIServiceError):
    code = "database_unavailable"
    status_code = 503
    public_message = "The prediction store is unavailable."


class UnknownPredictionError(APIServiceError):
    code = "unknown_feedback_request"
    status_code = 404
    public_message = "No prediction exists for the supplied request_id."


class InvalidFeedbackLabelError(APIServiceError):
    code = "invalid_feedback_label"
    status_code = 422
    public_message = "The corrected queue is not valid for the loaded model."


class RequestConstraintError(APIServiceError):
    code = "invalid_request"
    status_code = 422
    public_message = "The request violates an API input constraint."


class PredictionFailureError(APIServiceError):
    code = "prediction_failure"
    status_code = 500
    public_message = "Prediction failed."
