"""Interactive single-ticket routing and delayed feedback page."""

from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]
import streamlit as st

from ticket_router.api.schemas import ModelMetadataResponse, PredictionResponse
from ticket_router.dashboard.api_client import DashboardAPIError
from ticket_router.dashboard.ui import api_client, page_intro

PREDICTION_KEY = "route_ticket_prediction"
LATENCY_KEY = "route_ticket_latency_ms"
FEEDBACK_KEY = "route_ticket_feedback_request"


def render() -> None:
    page_intro(
        "Route a ticket",
        "Send a subject and body to the champion-backed FastAPI service. Ticket text stays out of "
        "normal application logs.",
    )
    metadata = _load_metadata()
    subject_limit = _contract_int(metadata, "subject_max_length", 2000)
    body_limit = _contract_int(metadata, "body_max_length", 20000)
    with st.form("ticket_prediction_form"):
        subject = st.text_input(
            "Subject",
            max_chars=subject_limit,
            placeholder="Example: Unable to access my account",
        )
        body = st.text_area(
            "Body",
            max_chars=body_limit,
            height=180,
            placeholder="Describe the support request without including unnecessary sensitive data.",
        )
        submitted = st.form_submit_button("Route ticket", type="primary")
    if submitted:
        if not subject.strip() and not body.strip():
            st.error("Enter a subject or body before routing the ticket.")
        else:
            try:
                with st.spinner("Routing through the local API…"):
                    result = api_client().predict(subject=subject, body=body)
                st.session_state[PREDICTION_KEY] = result.prediction.model_dump(mode="json")
                st.session_state[LATENCY_KEY] = result.api_latency_ms
                st.session_state.pop(FEEDBACK_KEY, None)
            except DashboardAPIError as exc:
                st.error(str(exc))
    raw_prediction = st.session_state.get(PREDICTION_KEY)
    if isinstance(raw_prediction, dict):
        prediction = PredictionResponse.model_validate(raw_prediction)
        _render_prediction(prediction, float(st.session_state.get(LATENCY_KEY, 0.0)))
        _render_feedback(prediction, metadata)


def _load_metadata() -> ModelMetadataResponse | None:
    try:
        return api_client().model_metadata()
    except DashboardAPIError:
        return None


def _render_prediction(prediction: PredictionResponse, latency_ms: float) -> None:
    st.subheader("Routing result")
    if prediction.warning:
        st.warning(prediction.warning)
    first, second, third, fourth = st.columns(4)
    first.metric("Predicted queue", prediction.predicted_queue)
    second.metric("Confidence", f"{prediction.confidence:.1%}")
    third.metric("Model", f"{prediction.model_name} v{prediction.model_version}")
    fourth.metric("API round trip", f"{latency_ms:,.1f} ms")
    ranked = pd.DataFrame(
        [
            {"Rank": index, "Queue": item.queue, "Confidence": item.confidence}
            for index, item in enumerate(prediction.top_k[:3], start=1)
        ]
    )
    st.dataframe(
        ranked,
        hide_index=True,
        use_container_width=True,
        column_config={"Confidence": st.column_config.ProgressColumn(format="percent")},
    )
    st.caption(f"Request ID: `{prediction.request_id}`")


def _render_feedback(
    prediction: PredictionResponse,
    metadata: ModelMetadataResponse | None,
) -> None:
    st.subheader("Submit feedback")
    if st.session_state.get(FEEDBACK_KEY) == prediction.request_id:
        st.success("Feedback recorded for this prediction.")
        return
    labels = metadata.labels if metadata is not None else [item.queue for item in prediction.top_k]
    with st.form(f"feedback_form_{prediction.request_id}"):
        assessment = st.radio(
            "Was this route correct?",
            ("Prediction correct", "Choose a corrected queue"),
            horizontal=True,
        )
        accepted = assessment == "Prediction correct"
        corrected_queue = prediction.predicted_queue
        if not accepted:
            corrected_queue = st.selectbox(
                "Correct queue",
                labels,
                index=labels.index(prediction.predicted_queue)
                if prediction.predicted_queue in labels
                else 0,
            )
        comment = st.text_area(
            "Optional non-sensitive comment",
            max_chars=500,
            placeholder="Do not include credentials or unnecessary personal information.",
        )
        submit_feedback = st.form_submit_button("Submit feedback")
    if submit_feedback:
        try:
            with st.spinner("Recording feedback…"):
                api_client().submit_feedback(
                    request_id=prediction.request_id,
                    corrected_queue=corrected_queue,
                    accepted=accepted,
                    comment=comment.strip() or None,
                )
            st.session_state[FEEDBACK_KEY] = prediction.request_id
            st.success("Feedback recorded.")
        except DashboardAPIError as exc:
            st.error(str(exc))


def _contract_int(metadata: ModelMetadataResponse | None, key: str, fallback: int) -> int:
    if metadata is None:
        return fallback
    value = metadata.input_contract.get(key)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else fallback
