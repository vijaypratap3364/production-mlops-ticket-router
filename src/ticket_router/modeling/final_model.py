"""Frozen final sparse-text pipeline selected by Stage 6 validation evidence."""

from __future__ import annotations

from typing import Any

from sklearn.calibration import CalibratedClassifierCV  # type: ignore[import-untyped]
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.svm import LinearSVC  # type: ignore[import-untyped]

from ticket_router.registry.config import SelectedCandidateSettings


def build_final_pipeline(
    selected: SelectedCandidateSettings,
    *,
    random_seed: int,
) -> Pipeline:
    """Build the exact Stage 6 winner for a fresh train+validation fit."""
    classifier = CalibratedClassifierCV(
        estimator=LinearSVC(
            C=selected.regularization_c,
            class_weight=selected.class_weight,
            dual="auto",
            max_iter=selected.max_iter,
            random_state=random_seed,
        ),
        cv=selected.calibration_cv_folds,
        method=selected.calibration_method,
        n_jobs=1,
    )
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    lowercase=True,
                    max_df=selected.max_df,
                    max_features=selected.max_features,
                    min_df=selected.min_df,
                    ngram_range=selected.word_ngram_range,
                    sublinear_tf=selected.sublinear_tf,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def final_pipeline_parameters(selected: SelectedCandidateSettings) -> dict[str, Any]:
    """Return flat parameters suitable for MLflow and lineage artifacts."""
    return {
        "candidate_name": selected.name,
        "tfidf_analyzer": "word",
        "tfidf_ngram_range": selected.word_ngram_range,
        "tfidf_min_df": selected.min_df,
        "tfidf_max_df": selected.max_df,
        "tfidf_max_features": selected.max_features,
        "tfidf_sublinear_tf": selected.sublinear_tf,
        "classifier": "LinearSVC",
        "classifier_c": selected.regularization_c,
        "classifier_class_weight": selected.class_weight,
        "classifier_max_iter": selected.max_iter,
        "calibration_cv_folds": selected.calibration_cv_folds,
        "calibration_method": selected.calibration_method,
    }
