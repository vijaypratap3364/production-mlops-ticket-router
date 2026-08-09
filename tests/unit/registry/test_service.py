"""Local SQLite MLflow Model Registry alias and model-loading tests."""

from importlib.metadata import version
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
from sklearn.naive_bayes import ComplementNB  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]

from ticket_router.modeling.mlflow_tracking import create_model_signature, safe_input_example
from ticket_router.registry.service import ModelRegistryService


def test_candidate_registration_alias_promotion_and_champion_loading(tmp_path: Path) -> None:
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    previous_uri = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(tracking_uri)
        experiment_id = mlflow.create_experiment(
            "registry-fixture",
            artifact_location=(tmp_path / "artifacts").as_uri(),
        )
        mlflow.set_experiment(experiment_id=experiment_id)
        texts = ["billing invoice", "payment receipt", "network outage", "router failure"]
        labels = ["Billing", "Billing", "Technical", "Technical"]
        pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("classifier", ComplementNB())]).fit(
            texts, labels
        )
        signature, input_example = create_model_signature(pipeline)
        with mlflow.start_run(run_name="registry-model") as run:
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                signature=signature,
                input_example=input_example,
                pip_requirements=[
                    f"cloudpickle=={version('cloudpickle')}",
                    f"numpy=={version('numpy')}",
                    f"scikit-learn=={version('scikit-learn')}",
                    f"scipy=={version('scipy')}",
                ],
            )
            run_id = run.info.run_id
        service = ModelRegistryService()
        registered = service.register_candidate(
            name="fixture-ticket-router",
            model_uri=f"runs:/{run_id}/model",
            run_id=run_id,
            candidate_alias="candidate",
            tags={"test_macro_f1": "0.8"},
        )

        candidate = service.resolve_alias(name=registered.name, alias="candidate")
        existing = service.find_version_by_run_id(name=registered.name, run_id=run_id)
        assert candidate is not None
        assert candidate.version == registered.version
        assert existing == registered
        assert service.resolve_alias(name=registered.name, alias="champion") is None

        service.assign_alias(
            name=registered.name,
            alias="candidate",
            version=registered.version,
        )

        promoted = service.promote_candidate(
            name=registered.name,
            candidate_alias="candidate",
            champion_alias="champion",
        )
        champion = service.resolve_alias(name=registered.name, alias="champion")
        loaded = service.load_alias(name=registered.name, alias="champion")
        version_loaded = service.load_version(name=registered.name, version=registered.version)

        assert champion is not None
        assert champion.version == promoted.version == registered.version
        examples = tuple(safe_input_example().tolist())
        assert np.array_equal(
            np.asarray(loaded.predict(examples), dtype=object),
            pipeline.predict(examples),
        )
        assert np.array_equal(
            np.asarray(version_loaded.predict(examples), dtype=object),
            pipeline.predict(examples),
        )
    finally:
        mlflow.set_tracking_uri(previous_uri)
