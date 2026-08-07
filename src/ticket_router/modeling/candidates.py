"""Focused sparse-text candidate families and search spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from sklearn.base import clone  # type: ignore[import-untyped]
from sklearn.calibration import CalibratedClassifierCV  # type: ignore[import-untyped]
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.multiclass import OneVsRestClassifier  # type: ignore[import-untyped]
from sklearn.naive_bayes import ComplementNB  # type: ignore[import-untyped]
from sklearn.pipeline import FeatureUnion, Pipeline  # type: ignore[import-untyped]
from sklearn.svm import LinearSVC  # type: ignore[import-untyped]

from ticket_router.modeling.experiment_config import CandidateSearchSettings

WORD_LOGISTIC = "word_tfidf_logistic_regression"
CHARACTER_LOGISTIC = "character_tfidf_logistic_regression"
COMBINED_LOGISTIC = "combined_word_character_logistic_regression"
CALIBRATED_LINEAR_SVC = "word_tfidf_calibrated_linear_svc"
WORD_COMPLEMENT_NB = "word_tfidf_complement_nb"


@dataclass(frozen=True)
class CandidateSpec:
    """One candidate estimator and its bounded parameter distribution."""

    name: str
    artifact_slug: str
    family: str
    estimator: Pipeline
    parameter_distributions: dict[str, list[Any]] | list[dict[str, list[Any]]]
    requires_calibration: bool = False


def build_candidate_specs(
    search: CandidateSearchSettings,
    *,
    random_seed: int,
) -> tuple[CandidateSpec, ...]:
    """Create the five required unfitted candidate families."""
    word_parameters = _word_parameters(search)
    logistic_parameters: dict[str, list[Any]] = {
        "classifier__estimator__C": list(search.regularization_c_values),
        "classifier__estimator__class_weight": list(search.class_weight_values),
    }
    return (
        CandidateSpec(
            name=WORD_LOGISTIC,
            artifact_slug="word_lr",
            family="logistic_regression",
            estimator=Pipeline(
                [
                    ("tfidf", _word_vectorizer()),
                    ("classifier", _logistic_classifier(random_seed)),
                ]
            ),
            parameter_distributions=_word_logistic_trials(search),
        ),
        CandidateSpec(
            name=CHARACTER_LOGISTIC,
            artifact_slug="char_lr",
            family="logistic_regression",
            estimator=Pipeline(
                [
                    ("tfidf", _character_vectorizer()),
                    ("classifier", _logistic_classifier(random_seed)),
                ]
            ),
            parameter_distributions={
                "tfidf__ngram_range": list(search.character_ngram_ranges),
                "tfidf__min_df": list(search.min_df_values),
                "tfidf__max_df": list(search.max_df_values),
                "tfidf__max_features": list(search.max_features_values),
                **logistic_parameters,
            },
        ),
        CandidateSpec(
            name=COMBINED_LOGISTIC,
            artifact_slug="word_char_lr",
            family="logistic_regression",
            estimator=Pipeline(
                [
                    (
                        "features",
                        FeatureUnion(
                            [
                                ("word", _word_vectorizer()),
                                ("character", _character_vectorizer()),
                            ]
                        ),
                    ),
                    ("classifier", _logistic_classifier(random_seed)),
                ]
            ),
            parameter_distributions={
                "features__word__ngram_range": list(search.word_ngram_ranges),
                "features__word__min_df": list(search.min_df_values),
                "features__word__max_df": list(search.max_df_values),
                "features__word__max_features": list(search.max_features_values),
                "features__character__ngram_range": list(search.character_ngram_ranges),
                "features__character__min_df": list(search.min_df_values),
                "features__character__max_df": list(search.max_df_values),
                "features__character__max_features": list(search.max_features_values),
                **logistic_parameters,
            },
        ),
        CandidateSpec(
            name=CALIBRATED_LINEAR_SVC,
            artifact_slug="calibrated_svc",
            family="linear_svc",
            estimator=Pipeline(
                [
                    ("tfidf", _word_vectorizer()),
                    (
                        "classifier",
                        LinearSVC(
                            class_weight="balanced",
                            dual="auto",
                            max_iter=5000,
                            random_state=random_seed,
                        ),
                    ),
                ]
            ),
            parameter_distributions={
                **word_parameters,
                "classifier__C": list(search.regularization_c_values),
                "classifier__class_weight": list(search.class_weight_values),
            },
            requires_calibration=True,
        ),
        CandidateSpec(
            name=WORD_COMPLEMENT_NB,
            artifact_slug="complement_nb",
            family="complement_naive_bayes",
            estimator=Pipeline(
                [
                    ("tfidf", _word_vectorizer()),
                    ("classifier", ComplementNB()),
                ]
            ),
            parameter_distributions={
                **word_parameters,
                "classifier__alpha": list(search.complement_nb_alpha_values),
            },
        ),
    )


def calibrated_final_estimator(
    spec: CandidateSpec,
    fitted_best_estimator: Pipeline,
    *,
    calibration_cv_folds: int,
    n_jobs: int,
) -> Pipeline:
    """Calibrate the selected LinearSVC parameters using training folds only."""
    if not spec.requires_calibration:
        return fitted_best_estimator
    vectorizer = clone(fitted_best_estimator.named_steps["tfidf"])
    selected_svc = cast(LinearSVC, fitted_best_estimator.named_steps["classifier"])
    calibrated = CalibratedClassifierCV(
        estimator=LinearSVC(
            C=selected_svc.C,
            class_weight=selected_svc.class_weight,
            dual=selected_svc.dual,
            max_iter=selected_svc.max_iter,
            random_state=selected_svc.random_state,
        ),
        cv=calibration_cv_folds,
        method="sigmoid",
        n_jobs=n_jobs,
    )
    return Pipeline([("tfidf", vectorizer), ("classifier", calibrated)])


def _word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="word", lowercase=True, sublinear_tf=True)


def _character_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char_wb", lowercase=True, sublinear_tf=True)


def _logistic_classifier(random_seed: int) -> OneVsRestClassifier:
    return OneVsRestClassifier(
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_seed,
            solver="liblinear",
        ),
        n_jobs=1,
    )


def _word_parameters(search: CandidateSearchSettings) -> dict[str, list[Any]]:
    return {
        "tfidf__ngram_range": list(search.word_ngram_ranges),
        "tfidf__min_df": list(search.min_df_values),
        "tfidf__max_df": list(search.max_df_values),
        "tfidf__max_features": list(search.max_features_values),
    }


def _word_logistic_trials(
    search: CandidateSearchSettings,
) -> list[dict[str, list[Any]]]:
    """Guarantee the known Stage 5 incumbent plus one bounded alternative.

    RandomizedSearchCV samples both singleton dictionaries when ``n_iter`` is two.
    This prevents a restrained random search from accidentally excluding the proven
    baseline configuration while still comparing a meaningfully different setting.
    """
    if search.iterations_per_candidate not in {1, 2}:
        raise ValueError("word logistic search supports one or two configured iterations")
    incumbent: dict[str, list[Any]] = {
        "tfidf__ngram_range": [max(search.word_ngram_ranges)],
        "tfidf__min_df": [min(search.min_df_values)],
        "tfidf__max_df": [max(search.max_df_values)],
        "tfidf__max_features": [max(search.max_features_values)],
        "classifier__estimator__C": [
            min(search.regularization_c_values, key=lambda value: abs(value - 1.0))
        ],
        "classifier__estimator__class_weight": ["balanced"],
    }
    alternative: dict[str, list[Any]] = {
        "tfidf__ngram_range": [min(search.word_ngram_ranges)],
        "tfidf__min_df": [max(search.min_df_values)],
        "tfidf__max_df": [min(search.max_df_values)],
        "tfidf__max_features": [min(search.max_features_values)],
        "classifier__estimator__C": [min(search.regularization_c_values)],
        "classifier__estimator__class_weight": [None],
    }
    return [incumbent] if search.iterations_per_candidate == 1 else [incumbent, alternative]
