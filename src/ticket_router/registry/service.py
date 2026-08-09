"""MLflow Model Registry adapter using aliases rather than deprecated stages."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from typing import cast

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from ticket_router.registry.contracts import Predictor


@dataclass(frozen=True)
class RegisteredVersion:
    """Stable model-version identifiers returned by registry operations."""

    name: str
    version: str
    run_id: str
    source: str


@dataclass(frozen=True)
class ModelVersionDetails:
    """Read-only model/run metadata used by presentation adapters."""

    created_at: datetime | None
    tags: dict[str, str]
    metrics: dict[str, float]


class ModelRegistryService:
    """Small alias-aware boundary around the MLflow registry client."""

    def __init__(self, client: MlflowClient | None = None) -> None:
        self._client = client or MlflowClient()

    def register_candidate(
        self,
        *,
        name: str,
        model_uri: str,
        run_id: str,
        candidate_alias: str,
        tags: Mapping[str, str],
    ) -> RegisteredVersion:
        """Create a version, attach lineage tags, and move only the candidate alias."""
        self._ensure_registered_model(name)
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=name,
                tags=dict(tags),
                await_registration_for=60,
            )
        version = str(model_version.version)
        self._client.set_registered_model_alias(name, candidate_alias, version)
        return RegisteredVersion(
            name=name,
            version=version,
            run_id=run_id,
            source=str(model_version.source),
        )

    def find_version_by_run_id(self, *, name: str, run_id: str) -> RegisteredVersion | None:
        """Find an existing version for a run so workflow retries do not duplicate it."""
        try:
            versions = self._client.search_model_versions(f"name = '{name}'")
        except MlflowException:
            return None
        matching = [version for version in versions if str(version.run_id or "") == run_id]
        if not matching:
            return None
        version = max(matching, key=lambda item: int(item.version))
        return RegisteredVersion(
            name=name,
            version=str(version.version),
            run_id=run_id,
            source=str(version.source),
        )

    def assign_alias(self, *, name: str, alias: str, version: str) -> None:
        """Assign one non-champion alias to a known immutable model version."""
        self._client.set_registered_model_alias(name, alias, version)

    def resolve_alias(self, *, name: str, alias: str) -> RegisteredVersion | None:
        """Resolve an alias to its immutable numeric version, or return no target."""
        try:
            model_version = self._client.get_model_version_by_alias(name, alias)
        except MlflowException:
            return None
        return RegisteredVersion(
            name=name,
            version=str(model_version.version),
            run_id=str(model_version.run_id or ""),
            source=str(model_version.source),
        )

    def promote_candidate(
        self,
        *,
        name: str,
        candidate_alias: str,
        champion_alias: str,
    ) -> RegisteredVersion:
        """Move champion to the current candidate numeric version."""
        candidate = self.resolve_alias(name=name, alias=candidate_alias)
        if candidate is None:
            raise ValueError(f"candidate alias does not exist for registered model {name!r}")
        self._client.set_registered_model_alias(name, champion_alias, candidate.version)
        return candidate

    def model_version_tags(self, *, name: str, version: str) -> dict[str, str]:
        model_version = self._client.get_model_version(name, version)
        return dict(model_version.tags or {})

    def model_version_details(self, *, name: str, version: str) -> ModelVersionDetails:
        """Return registry lineage and immutable run metrics without loading the model again."""
        model_version = self._client.get_model_version(name, version)
        run = self._client.get_run(str(model_version.run_id))
        created_at = (
            datetime.fromtimestamp(model_version.creation_timestamp / 1000.0, tz=UTC)
            if model_version.creation_timestamp is not None
            else None
        )
        return ModelVersionDetails(
            created_at=created_at,
            tags=dict(model_version.tags or {}),
            metrics={key: float(value) for key, value in run.data.metrics.items()},
        )

    def set_model_version_tags(
        self,
        *,
        name: str,
        version: str,
        tags: Mapping[str, str],
    ) -> None:
        for key, value in tags.items():
            self._client.set_model_version_tag(name, version, key, value)

    def load_alias(self, *, name: str, alias: str) -> Predictor:
        """Load a trusted sklearn model through a registry alias."""
        return cast(Predictor, mlflow.sklearn.load_model(f"models:/{name}@{alias}"))

    def load_version(self, *, name: str, version: str) -> Predictor:
        """Load an immutable numeric version after resolving a deployment alias."""
        return cast(Predictor, mlflow.sklearn.load_model(f"models:/{name}/{version}"))

    def _ensure_registered_model(self, name: str) -> None:
        try:
            self._client.get_registered_model(name)
        except MlflowException:
            self._client.create_registered_model(name)
