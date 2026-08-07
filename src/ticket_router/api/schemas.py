"""Versioned Pydantic contracts for the inference HTTP adapter."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

SubjectText = Annotated[str, StringConstraints(max_length=2000)]
BodyText = Annotated[str, StringConstraints(max_length=20000)]


class RequestMetadata(BaseModel):
    """Bounded operational metadata that is never passed to the model."""

    model_config = ConfigDict(extra="forbid")

    client_name: str | None = Field(default=None, min_length=1, max_length=64)
    correlation_id: str | None = Field(default=None, min_length=1, max_length=128)


class TicketRequest(BaseModel):
    """One support ticket containing only approved predictive source fields."""

    model_config = ConfigDict(extra="forbid")

    subject: SubjectText = ""
    body: BodyText = ""
    metadata: RequestMetadata | None = None

    @model_validator(mode="after")
    def require_usable_text(self) -> TicketRequest:
        if not self.subject.strip() and not self.body.strip():
            raise ValueError("subject or body must contain usable text")
        return self


class BatchPredictionRequest(BaseModel):
    """Bounded collection of prediction inputs; deployment limit is checked separately."""

    model_config = ConfigDict(extra="forbid")

    items: list[TicketRequest] = Field(min_length=1, max_length=1000)


class ClassProbability(BaseModel):
    queue: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    request_id: str
    predicted_queue: str
    confidence: float = Field(ge=0.0, le=1.0)
    top_k: list[ClassProbability]
    model_name: str
    model_version: str
    prediction_timestamp: datetime
    warning: str | None = None


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=128)
    corrected_queue: str = Field(min_length=1, max_length=200)
    accepted: bool | None = None
    comment: str | None = Field(default=None, min_length=1, max_length=500)
    source: Literal["user", "agent", "reviewer", "demo"] = "user"


class FeedbackResponse(BaseModel):
    request_id: str
    feedback_id: str
    corrected_queue: str
    recorded_at: datetime


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    ready: bool
    model_ready: bool
    database_ready: bool


class ModelMetadataResponse(BaseModel):
    model_name: str
    model_version: str
    alias: str
    load_timestamp: datetime
    input_contract: dict[str, Any]
    labels: list[str]


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str
    validation: list[dict[str, Any]] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
