"""Leakage-safe scikit-learn pipelines for the Stage 5 baselines."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.dummy import DummyClassifier  # type: ignore[import-untyped]
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.multiclass import OneVsRestClassifier  # type: ignore[import-untyped]
from sklearn.naive_bayes import ComplementNB  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import FunctionTransformer  # type: ignore[import-untyped]

from ticket_router.modeling.config import BaselineSettings

DUMMY_MODEL_NAME = "dummy_most_frequent"
LOGISTIC_MODEL_NAME = "tfidf_logistic_regression"
COMPLEMENT_NB_MODEL_NAME = "tfidf_complement_nb"


def build_baseline_pipelines(
    config: BaselineSettings,
    *,
    random_seed: int,
) -> dict[str, Pipeline]:
    """Create unfitted pipelines; callers fit them only with training text."""
    return {
        DUMMY_MODEL_NAME: Pipeline(
            [
                (
                    "constant_features",
                    FunctionTransformer(_constant_feature, validate=False),
                ),
                (
                    "classifier",
                    DummyClassifier(strategy="most_frequent", random_state=random_seed),
                ),
            ]
        ),
        LOGISTIC_MODEL_NAME: Pipeline(
            [
                ("tfidf", _tfidf_vectorizer(config)),
                (
                    "classifier",
                    OneVsRestClassifier(
                        LogisticRegression(
                            C=config.logistic_regression.c,
                            class_weight=config.logistic_regression.class_weight,
                            max_iter=config.logistic_regression.max_iter,
                            random_state=random_seed,
                            solver=config.logistic_regression.solver,
                        ),
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        COMPLEMENT_NB_MODEL_NAME: Pipeline(
            [
                ("tfidf", _tfidf_vectorizer(config)),
                ("classifier", ComplementNB(alpha=config.complement_nb.alpha)),
            ]
        ),
    }


def _tfidf_vectorizer(config: BaselineSettings) -> TfidfVectorizer:
    """Build a fresh word-vectorizer so vocabulary cannot be shared across runs."""
    return TfidfVectorizer(
        lowercase=True,
        max_features=config.tfidf.max_features,
        min_df=config.tfidf.min_df,
        ngram_range=config.tfidf.ngram_range,
        sublinear_tf=config.tfidf.sublinear_tf,
        dtype=np.float64,
    )


def _constant_feature(texts: Sequence[str]) -> NDArray[np.float64]:
    """Give DummyClassifier a shape-safe input without learning text state."""
    return np.ones((len(texts), 1), dtype=np.float64)


def pipeline_components(pipeline: Pipeline) -> tuple[str, ...]:
    """Return stable component names for model configuration artifacts."""
    return tuple(str(name) for name, _ in pipeline.steps)
