"""MLflow tracking adapters with a zero-cost local file-store fallback."""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import ModelSignature, infer_signature
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]

from ticket_router.config import Settings
from ticket_router.modeling.artifacts import environment_versions
from ticket_router.modeling.experiment_config import CandidateExperimentSettings


@dataclass(frozen=True)
class TrackingResolution:
    """Requested and effective MLflow tracking configuration."""

    requested_uri: str
    resolved_uri: str
    local_fallback_used: bool


@dataclass(frozen=True)
class LoggedCandidate:
    """Stable identifiers for one logged fitted candidate."""

    run_id: str
    model_uri: str


def configure_experiment_tracking(
    *,
    settings: Settings,
    experiment_config: CandidateExperimentSettings,
    project_root: Path,
    availability_check: Callable[[str], bool] | None = None,
) -> TrackingResolution:
    """Use the configured URI when reachable, otherwise an ignored local file store."""
    requested_uri = settings.effective_mlflow_tracking_uri
    resolved_uri = requested_uri
    fallback_used = False
    if requested_uri.startswith(("http://", "https://")):
        checker = availability_check or _http_tracking_available
        if not checker(requested_uri):
            if not experiment_config.allow_local_tracking_fallback:
                raise ConnectionError(f"MLflow tracking server is unavailable: {requested_uri}")
            local_directory = project_root / experiment_config.local_tracking_directory
            resolved_uri = _local_sqlite_tracking_uri(local_directory)
            fallback_used = True
    elif not requested_uri.startswith(("file:", "sqlite:")):
        local_directory = Path(requested_uri)
        if not local_directory.is_absolute():
            local_directory = project_root / local_directory
        resolved_uri = _local_sqlite_tracking_uri(local_directory)
    mlflow.set_tracking_uri(resolved_uri)
    mlflow.set_experiment(experiment_config.experiment_name)
    return TrackingResolution(
        requested_uri=requested_uri,
        resolved_uri=resolved_uri,
        local_fallback_used=fallback_used,
    )


def safe_input_example() -> NDArray[np.str_]:
    """Return synthetic model text suitable for a public MLflow artifact."""
    return np.asarray(
        [
            "Example billing question for invoice review.",
            "Example technical request about a local network issue.",
            "Example return request for an unopened product.",
        ],
        dtype=str,
    )


def create_model_signature(pipeline: Pipeline) -> tuple[ModelSignature, NDArray[np.str_]]:
    """Infer a string-tensor signature without using real ticket content."""
    input_example = safe_input_example()
    predictions = pipeline.predict(input_example)
    return infer_signature(input_example, predictions), input_example


def log_candidate_to_mlflow(
    *,
    experiment_name: str,
    run_name: str,
    pipeline: Pipeline,
    parameters: Mapping[str, object],
    metrics: Mapping[str, float],
    per_class_metrics: tuple[dict[str, float | int | str], ...],
    tags: Mapping[str, str],
    lineage: Mapping[str, object],
    cross_validation_summary: Mapping[str, object],
    local_artifact_directory: Path,
) -> LoggedCandidate:
    """Log one complete candidate run and its loadable fitted sklearn model."""
    mlflow.set_experiment(experiment_name)
    signature, input_example = create_model_signature(pipeline)
    with mlflow.start_run(run_name=run_name, tags=dict(tags)) as active_run:
        mlflow.log_params({key: _parameter_value(value) for key, value in parameters.items()})
        mlflow.log_metrics(dict(metrics))
        mlflow.log_metrics(_per_class_mlflow_metrics(per_class_metrics))
        mlflow.log_dict(dict(lineage), "lineage.json")
        versions = environment_versions()
        mlflow.log_dict(versions, "package_versions.json")
        mlflow.log_dict(dict(cross_validation_summary), "cross_validation_summary.json")
        mlflow.log_artifacts(str(local_artifact_directory), artifact_path="evaluation")
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            input_example=input_example,
            serialization_format="cloudpickle",
            metadata={"test_evaluated": False},
            pip_requirements=_model_requirements(versions),
        )
        run_id = active_run.info.run_id
    return LoggedCandidate(run_id=run_id, model_uri=f"runs:/{run_id}/model")


def _per_class_mlflow_metrics(
    rows: tuple[dict[str, float | int | str], ...],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for row in rows:
        class_slug = re.sub(r"[^a-z0-9]+", "_", str(row["class"]).casefold()).strip("_")
        metrics[f"validation_precision__{class_slug}"] = float(row["precision"])
        metrics[f"validation_recall__{class_slug}"] = float(row["recall"])
        metrics[f"validation_f1__{class_slug}"] = float(row["f1"])
    return metrics


def _parameter_value(value: object) -> str | int | float | bool:
    if value is None:
        return "none"
    if isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _model_requirements(versions: Mapping[str, str]) -> list[str]:
    packages = ("cloudpickle", "joblib", "numpy", "scikit-learn", "scipy")
    return [f"{package}=={versions[package]}" for package in packages]


def _http_tracking_available(uri: str) -> bool:
    health_url = uri.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(health_url, timeout=2.0) as response:
            status = int(response.getcode())
            return 200 <= status < 300
    except (OSError, urllib.error.URLError, ValueError):
        return False


def _local_sqlite_tracking_uri(directory: Path) -> str:
    """Create a supported zero-cost local tracking database URI."""
    resolved_directory = directory.resolve()
    resolved_directory.mkdir(parents=True, exist_ok=True)
    database_path = (resolved_directory / "mlflow.db").as_posix()
    return f"sqlite:///{database_path}"
