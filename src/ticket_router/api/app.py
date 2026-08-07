"""FastAPI application factory with champion and dependency lifespan ownership."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from structlog.contextvars import bind_contextvars, clear_contextvars

from ticket_router.api.errors import (
    APIServiceError,
    DatabaseUnavailableError,
    ModelUnavailableError,
    PredictionFailureError,
    RequestConstraintError,
)
from ticket_router.api.metrics import APIMetrics
from ticket_router.api.model_loader import ChampionLoader, load_champion
from ticket_router.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelMetadataResponse,
    PredictionResponse,
    ReadinessResponse,
    TicketRequest,
)
from ticket_router.api.service import LoadedChampion, PredictionService
from ticket_router.api.store import (
    InMemoryPredictionFeedbackStore,
    PredictionFeedbackStore,
)
from ticket_router.config import Settings
from ticket_router.db.connectivity import DatabaseProbe, SQLAlchemyDatabaseProbe
from ticket_router.logging_config import configure_logging, get_logger

StoreFactory = Callable[[], PredictionFeedbackStore]
DatabaseProbeFactory = Callable[[str], DatabaseProbe]
KNOWN_ROUTES = frozenset(
    {"/health", "/ready", "/model", "/predict", "/predict/batch", "/feedback", "/metrics"}
)


@dataclass
class RuntimeState:
    settings: Settings | None = None
    champion: LoadedChampion | None = None
    prediction_service: PredictionService | None = None
    store: PredictionFeedbackStore | None = None
    database_probe: DatabaseProbe | None = None
    model_ready: bool = False
    database_ready: bool = False

    @property
    def ready(self) -> bool:
        return self.model_ready and self.database_ready and self.prediction_service is not None


def create_app(
    *,
    settings: Settings | None = None,
    champion_loader: ChampionLoader = load_champion,
    store_factory: StoreFactory = InMemoryPredictionFeedbackStore,
    database_probe_factory: DatabaseProbeFactory = SQLAlchemyDatabaseProbe,
) -> FastAPI:
    """Create an isolated app; external resources are opened only during lifespan."""
    runtime = RuntimeState()
    metrics = APIMetrics()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> Any:
        active_settings = settings or Settings.load()
        runtime.settings = active_settings
        configure_logging(active_settings.log_level)
        logger = get_logger(__name__)
        runtime.store = store_factory()
        database_url = (
            active_settings.database_url.get_secret_value()
            if active_settings.database_url is not None
            else None
        )
        if database_url is not None:
            try:
                runtime.database_probe = database_probe_factory(database_url)
                await run_in_threadpool(runtime.database_probe.connect)
                runtime.database_ready = runtime.database_probe.ready
            except Exception as exc:
                runtime.database_ready = False
                logger.error("database_startup_failed", error_type=type(exc).__name__)
        elif active_settings.database_required:
            logger.error("database_startup_failed", error_type="MissingDatabaseUrl")
        else:
            runtime.database_ready = True
        try:
            runtime.champion = await run_in_threadpool(champion_loader, active_settings)
            runtime.model_ready = True
        except Exception as exc:
            runtime.model_ready = False
            logger.error("champion_load_failed", error_type=type(exc).__name__)
        if runtime.champion is not None and runtime.database_ready and runtime.store is not None:
            runtime.prediction_service = PredictionService(
                champion=runtime.champion,
                api_settings=active_settings.api_settings,
                preprocessing=active_settings.project_config.preprocessing,
                store=runtime.store,
                metrics=metrics,
            )
        logger.info(
            "api_startup_complete",
            model_ready=runtime.model_ready,
            database_ready=runtime.database_ready,
            model_name=(runtime.champion.model_name if runtime.champion else None),
            model_version=(runtime.champion.model_version if runtime.champion else None),
        )
        try:
            yield
        finally:
            if runtime.store is not None:
                await run_in_threadpool(runtime.store.close)
            if runtime.database_probe is not None:
                await run_in_threadpool(runtime.database_probe.close)
            runtime.prediction_service = None
            runtime.model_ready = False
            runtime.database_ready = False
            logger.info("api_shutdown_complete")

    app = FastAPI(
        title="Production MLOps Ticket Router",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.runtime = runtime
    app.state.metrics = metrics

    @app.middleware("http")
    async def request_observability(request: Request, call_next: Callable[..., Any]) -> Response:
        request_id = str(uuid4())
        request.state.request_id = request_id
        route = request.url.path if request.url.path in KNOWN_ROUTES else "unmatched"
        bind_contextvars(request_id=request_id)
        started = perf_counter()
        response: Response
        try:
            response = await call_next(request)
        finally:
            clear_contextvars()
        elapsed = perf_counter() - started
        response.headers["X-Request-ID"] = request_id
        metrics.requests.labels(
            method=request.method,
            route=route,
            status=str(response.status_code),
        ).inc()
        metrics.request_latency.labels(method=request.method, route=route).observe(elapsed)
        get_logger(__name__).info(
            "http_request_complete",
            request_id=request_id,
            method=request.method,
            route=route,
            status_code=response.status_code,
            duration_seconds=elapsed,
        )
        return response

    @app.exception_handler(APIServiceError)
    async def service_error_handler(request: Request, exc: APIServiceError) -> JSONResponse:
        metrics.errors.labels(code=exc.code).inc()
        return _error_response(
            request=request,
            status_code=exc.status_code,
            code=exc.code,
            message=exc.public_message,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        metrics.errors.labels(code="invalid_request").inc()
        validation = [
            {
                "location": [str(part) for part in error["loc"]],
                "message": str(error["msg"]),
                "type": str(error["type"]),
            }
            for error in exc.errors()
        ]
        return _error_response(
            request=request,
            status_code=422,
            code="invalid_request",
            message="Request validation failed.",
            validation=validation,
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        metrics.errors.labels(code="internal_error").inc()
        get_logger(__name__).error(
            "unhandled_api_error",
            request_id=_request_id(request),
            error_type=type(exc).__name__,
        )
        return _error_response(
            request=request,
            status_code=500,
            code="internal_error",
            message="An internal error occurred.",
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadinessResponse)
    async def ready() -> ReadinessResponse | JSONResponse:
        result = ReadinessResponse(
            ready=runtime.ready,
            model_ready=runtime.model_ready,
            database_ready=runtime.database_ready,
        )
        if not runtime.ready:
            return JSONResponse(status_code=503, content=result.model_dump(mode="json"))
        return result

    @app.get(
        "/model",
        response_model=ModelMetadataResponse,
        responses={503: {"model": ErrorResponse}},
    )
    async def model_metadata() -> ModelMetadataResponse:
        service = _require_service(runtime)
        champion = service.champion
        return ModelMetadataResponse(
            model_name=champion.model_name,
            model_version=champion.model_version,
            alias=champion.alias,
            load_timestamp=champion.loaded_at,
            input_contract=champion.input_contract,
            labels=list(champion.labels),
        )

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    async def predict(request: Request, ticket: TicketRequest) -> PredictionResponse:
        service = _require_service(runtime)
        return await run_in_threadpool(
            service.predict_one,
            ticket,
            request_id=_request_id(request),
        )

    @app.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    async def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
        service = _require_service(runtime)
        if len(payload.items) > service.api_settings.maximum_batch_size:
            raise RequestConstraintError("prediction batch exceeds configured maximum")
        predictions = await run_in_threadpool(service.predict_many, payload.items)
        return BatchPredictionResponse(predictions=predictions)

    @app.post(
        "/feedback",
        response_model=FeedbackResponse,
        status_code=201,
        responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    )
    async def feedback(payload: FeedbackRequest) -> FeedbackResponse:
        service = _require_service(runtime)
        return await run_in_threadpool(service.record_feedback, payload)

    @app.get("/metrics", response_class=Response)
    async def prometheus_metrics() -> Response:
        return Response(
            content=metrics.render(),
            headers={"Content-Type": metrics.content_type},
        )

    return app


def _require_service(runtime: RuntimeState) -> PredictionService:
    if not runtime.model_ready or runtime.champion is None:
        raise ModelUnavailableError("champion was not loaded")
    if not runtime.database_ready:
        raise DatabaseUnavailableError("database dependency is not ready")
    if runtime.prediction_service is None:
        raise PredictionFailureError("prediction service initialization failed")
    return runtime.prediction_service


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unavailable"))


def _error_response(
    *,
    request: Request,
    status_code: int,
    code: str,
    message: str,
    validation: list[dict[str, Any]] | None = None,
) -> JSONResponse:
    content: dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
            "request_id": _request_id(request),
        }
    }
    if validation is not None:
        content["error"]["validation"] = validation
    return JSONResponse(status_code=status_code, content=content)
