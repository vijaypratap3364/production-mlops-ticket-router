"""Realistic bounded workloads using only synthetic non-sensitive ticket text."""

from __future__ import annotations

import os
from collections import deque
from itertools import count
from random import Random
from typing import Any

from locust import HttpUser, between, task

SHORT_TICKETS = (
    {"subject": "Invoice question", "body": "Please explain this synthetic charge."},
    {"subject": "Return request", "body": "A demo parcel needs an exchange."},
    {"subject": "Network issue", "body": "The local demonstration service cannot connect."},
)
LONG_TICKETS = (
    {
        "subject": "Detailed synthetic service request",
        "body": " ".join(
            ["Non-sensitive benchmark detail about a local support and connectivity problem."] * 90
        ),
    },
    {
        "subject": "Detailed synthetic billing request",
        "body": " ".join(
            ["Non-sensitive benchmark detail about a demonstration invoice and payment question."]
            * 90
        ),
    },
)
LOAD_TEST_SEED = int(os.environ.get("TICKET_ROUTER_LOAD_TEST_SEED", "42"))
USER_SEQUENCE = count()


class TicketRoutingUser(HttpUser):
    """Bounded mix: mostly single predictions, occasional batches, sparse feedback."""

    wait_time = between(0.25, 1.0)

    def on_start(self) -> None:
        self._random = Random(LOAD_TEST_SEED + next(USER_SEQUENCE))
        self._feedback_candidates: deque[tuple[str, str]] = deque(maxlen=20)
        with self.client.get("/ready", name="GET /ready", catch_response=True) as response:
            if response.status_code != 200 or response.json().get("ready") is not True:
                response.failure("API is not ready")

    @task(12)
    def predict_single(self) -> None:
        ticket = self._random.choice((*SHORT_TICKETS, *LONG_TICKETS))
        with self.client.post(
            "/predict",
            json=ticket,
            name="POST /predict",
            catch_response=True,
        ) as response:
            payload = _validated_prediction(response)
            if payload is not None:
                self._feedback_candidates.append(
                    (str(payload["request_id"]), str(payload["predicted_queue"]))
                )

    @task(2)
    def predict_batch(self) -> None:
        batch_size = self._random.choice((2, 4, 8))
        items = [self._random.choice((*SHORT_TICKETS, *LONG_TICKETS)) for _ in range(batch_size)]
        with self.client.post(
            "/predict/batch",
            json={"items": items},
            name="POST /predict/batch",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"unexpected status {response.status_code}")
                return
            payload: Any = response.json()
            predictions = payload.get("predictions") if isinstance(payload, dict) else None
            if not isinstance(predictions, list) or len(predictions) != batch_size:
                response.failure("invalid batch prediction contract")
                return
            for prediction in predictions:
                if isinstance(prediction, dict):
                    self._feedback_candidates.append(
                        (
                            str(prediction.get("request_id", "")),
                            str(prediction.get("predicted_queue", "")),
                        )
                    )

    @task(1)
    def submit_feedback(self) -> None:
        if not self._feedback_candidates:
            self.predict_single()
            return
        request_id, predicted_queue = self._feedback_candidates.popleft()
        with self.client.post(
            "/feedback",
            json={
                "request_id": request_id,
                "corrected_queue": predicted_queue,
                "accepted": True,
                "comment": "Synthetic bounded load-test feedback",
                "source": "demo",
            },
            name="POST /feedback",
            catch_response=True,
        ) as response:
            if response.status_code != 201:
                response.failure(f"unexpected status {response.status_code}")


def _validated_prediction(response: Any) -> dict[str, object] | None:
    if response.status_code != 200:
        response.failure(f"unexpected status {response.status_code}")
        return None
    payload: Any = response.json()
    if not isinstance(payload, dict):
        response.failure("prediction response is not an object")
        return None
    required = ("request_id", "predicted_queue", "confidence", "model_version")
    if any(key not in payload for key in required):
        response.failure("prediction response is missing required fields")
        return None
    return payload
