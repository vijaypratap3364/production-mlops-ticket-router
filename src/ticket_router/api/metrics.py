"""Per-process Prometheus metrics for the inference adapter."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest


class APIMetrics:
    """Own a registry so tests and multiple app factories do not collide globally."""

    content_type = "text/plain; version=0.0.4; charset=utf-8"

    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        self.requests = Counter(
            "ticket_router_http_requests_total",
            "HTTP requests received.",
            ("method", "route", "status"),
            registry=self.registry,
        )
        self.predictions = Counter(
            "ticket_router_predictions_total",
            "Ticket predictions returned.",
            registry=self.registry,
        )
        self.errors = Counter(
            "ticket_router_errors_total",
            "Handled API errors.",
            ("code",),
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "ticket_router_http_request_duration_seconds",
            "HTTP request latency.",
            ("method", "route"),
            registry=self.registry,
        )
        self.prediction_latency = Histogram(
            "ticket_router_prediction_duration_seconds",
            "Model prediction latency.",
            registry=self.registry,
        )
        self.predicted_labels = Counter(
            "ticket_router_predicted_labels_total",
            "Predictions by queue label.",
            ("queue",),
            registry=self.registry,
        )
        self.low_confidence = Counter(
            "ticket_router_low_confidence_total",
            "Predictions below the configured confidence threshold.",
            registry=self.registry,
        )
        self.batch_size = Histogram(
            "ticket_router_prediction_batch_size",
            "Requested prediction batch size.",
            buckets=(1, 2, 5, 10, 25, 50, 100, 250, 500, 1000),
            registry=self.registry,
        )
        self.persistence_failures = Counter(
            "ticket_router_persistence_failures_total",
            "Best-effort or required persistence operation failures.",
            ("operation",),
            registry=self.registry,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)
