"""Local MLflow logging, signature, fallback, and model-loading tests."""

from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np

from ticket_router.config import Settings
from ticket_router.modeling.baselines import build_baseline_pipelines
from ticket_router.modeling.config import BaselineSettings
from ticket_router.modeling.experiment_config import CandidateExperimentSettings
from ticket_router.modeling.mlflow_tracking import (
    configure_experiment_tracking,
    create_model_signature,
    log_candidate_to_mlflow,
    safe_input_example,
)


def test_unavailable_server_falls_back_to_local_file_tracking(
    tmp_path: Path,
    small_experiment_config: CandidateExperimentSettings,
) -> None:
    settings = Settings.load(env_file=None)
    previous_uri = mlflow.get_tracking_uri()
    try:
        resolution = configure_experiment_tracking(
            settings=settings,
            experiment_config=small_experiment_config,
            project_root=tmp_path,
            availability_check=lambda _uri: False,
        )

        assert resolution.local_fallback_used
        assert resolution.resolved_uri.startswith("sqlite:")
        assert (tmp_path / "mlruns").is_dir()
    finally:
        mlflow.set_tracking_uri(previous_uri)


def test_signature_uses_safe_string_tensor_example(
    tiny_text_classification_data: tuple[list[str], list[str]],
    small_baseline_config: BaselineSettings,
) -> None:
    texts, labels = tiny_text_classification_data
    pipeline = build_baseline_pipelines(small_baseline_config, random_seed=42)[
        "tfidf_complement_nb"
    ].fit(texts, labels)

    signature, example = create_model_signature(pipeline)

    assert signature.inputs is not None
    assert example.shape == (3,)
    assert all("Example" in value for value in example)


def test_logged_model_can_be_loaded_from_local_mlflow_run(
    tmp_path: Path,
    tiny_text_classification_data: tuple[list[str], list[str]],
    small_baseline_config: BaselineSettings,
) -> None:
    texts, labels = tiny_text_classification_data
    pipeline = build_baseline_pipelines(small_baseline_config, random_seed=42)[
        "tfidf_complement_nb"
    ].fit(texts, labels)
    tracking_directory = tmp_path / "mlruns"
    tracking_directory.mkdir()
    tracking_uri = f"sqlite:///{(tracking_directory / 'mlflow.db').resolve().as_posix()}"
    artifact_directory = tmp_path / "evaluation"
    artifact_directory.mkdir()
    (artifact_directory / "metrics.json").write_text("{}\n", encoding="utf-8")
    previous_uri = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(tracking_uri)
        logged = log_candidate_to_mlflow(
            experiment_name="fixture-model-logging",
            run_name="fixture-complement-nb",
            pipeline=pipeline,
            parameters={"alpha": 1.0},
            metrics={"validation_macro_f1": 0.5},
            per_class_metrics=(
                {
                    "class": "Billing",
                    "precision": 0.5,
                    "recall": 0.5,
                    "f1": 0.5,
                    "support": 12,
                },
            ),
            tags={"model_family": "complement_naive_bayes", "test_evaluated": "false"},
            lineage={"dataset_manifest_sha256": "a" * 64, "test_evaluated": False},
            cross_validation_summary={"best_mean_macro_f1": 0.5},
            local_artifact_directory=artifact_directory,
        )
        loaded = mlflow.sklearn.load_model(logged.model_uri)
        expected = pipeline.predict(safe_input_example())
        observed = loaded.predict(safe_input_example())
        run = mlflow.MlflowClient().get_run(logged.run_id)

        assert np.array_equal(observed, expected)
        assert run.data.metrics["validation_macro_f1"] == 0.5
        assert run.data.tags["test_evaluated"] == "false"
    finally:
        mlflow.set_tracking_uri(previous_uri)
