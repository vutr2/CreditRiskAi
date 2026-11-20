"""Tests for credit_risk_ai.models.pipeline.

Two parts:
  1. Small synthetic data: checks how the builder functions behave.
  2. Real data (skipped when data/raw/ is missing): checks that the models still
     reproduce the validation scores of notebook 02 (default models) and the CV
     score of notebook 05 (tuned model), so an accidental change to a model
     definition is caught.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.utils.validation import check_is_fitted

from credit_risk_ai.data.clean import clean, drop_duplicate_rows
from credit_risk_ai.data.features import add_features
from credit_risk_ai.data.load import RAW_DIR, load_train
from credit_risk_ai.models.pipeline import (
    build_hist_gradient_boosting,
    build_logistic_regression,
    build_tuned_hist_gradient_boosting,
)

BUILDERS = [
    build_logistic_regression,
    build_hist_gradient_boosting,
    build_tuned_hist_gradient_boosting,
]


def make_data(n_rows=200, seed=0):
    """Small synthetic training data that looks like the cleaned credit data.

    All values are >= 0 (like the real data, so log1p is safe), "income" has
    missing values, and the target contains both classes.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "income": rng.lognormal(mean=8.5, sigma=0.6, size=n_rows),
        "utilization": rng.uniform(0, 1.5, size=n_rows),
        "age": rng.integers(18, 90, size=n_rows).astype(float),
        "late_payments": rng.poisson(0.3, size=n_rows).astype(float),
    })
    X.loc[X.sample(frac=0.2, random_state=seed).index, "income"] = np.nan
    y = pd.Series((np.arange(n_rows) % 4 == 0).astype(int))  # 25% defaults
    return X, y


# =============================================================================
# Part 1: synthetic data
# =============================================================================


@pytest.mark.parametrize("build", BUILDERS)
def test_each_call_returns_a_new_model(build):
    # Cross-validation fits one model per fold; a shared object would be overwritten.
    assert build() is not build()


@pytest.mark.parametrize("build", BUILDERS)
def test_returned_model_is_not_fitted(build):
    with pytest.raises(NotFittedError):
        check_is_fitted(build())


def test_logistic_regression_steps_are_in_the_right_order():
    # log1p must come before scaling (scaling creates values below -1, where
    # log1p gives NaN), and the leakage check in notebook 02 relies on "impute".
    steps = list(build_logistic_regression().named_steps)
    assert steps == ["impute", "log", "scale", "model"]


@pytest.mark.parametrize("build", BUILDERS)
def test_fits_on_data_with_missing_values_and_returns_probabilities(build):
    X, y = make_data()
    model = build().fit(X, y)
    proba = model.predict_proba(X)

    assert proba.shape == (len(X), 2)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert np.allclose(proba.sum(axis=1), 1)


def test_imputer_learns_medians_from_the_fitting_data_only():
    # Golden rule: the pipeline may only learn from the rows it is fitted on.
    X, y = make_data()
    X_fit, y_fit = X.iloc[:150], y.iloc[:150]

    model = build_logistic_regression().fit(X_fit, y_fit)
    learned = model.named_steps["impute"].statistics_

    assert np.allclose(learned, X_fit.median())
    assert not np.allclose(learned, X.median())


@pytest.mark.parametrize(
    "build", [build_hist_gradient_boosting, build_tuned_hist_gradient_boosting]
)
def test_hist_gradient_boosting_uses_the_given_random_state(build):
    assert build().random_state == 42
    assert build(random_state=7).random_state == 7


def test_tuned_model_has_the_notebook_05_hyperparameters():
    # Values written out here (not imported), so an accidental edit of
    # TUNED_HGB_PARAMS makes this test fail.
    params = build_tuned_hist_gradient_boosting().get_params()
    assert params["max_iter"] == 500
    assert params["learning_rate"] == 0.067
    assert params["max_leaf_nodes"] == 21
    assert params["min_samples_leaf"] == 378
    assert params["l2_regularization"] == 1.1


def test_default_model_keeps_the_default_hyperparameters():
    # Notebooks 02-04 use the default model; tuning must not change it.
    params = build_hist_gradient_boosting().get_params()
    assert params["max_iter"] == 100
    assert params["learning_rate"] == 0.1
    assert params["min_samples_leaf"] == 20


# =============================================================================
# Part 2: real data (skipped when data/raw/ is missing)
# =============================================================================

needs_raw_data = pytest.mark.skipif(
    not (RAW_DIR / "cs-training.csv").exists(),
    reason="raw data not found in data/raw/",
)


@pytest.fixture(scope="module")
def train_val_split():
    """The same split as notebook 02: cleaned, deduplicated, stratified 80/20."""
    data = clean(drop_duplicate_rows(load_train()))
    X = data.drop(columns="SeriousDlqin2yrs")
    y = data["SeriousDlqin2yrs"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


@needs_raw_data
@pytest.mark.parametrize(
    "build, expected_roc_auc",
    [(build_logistic_regression, 0.8548), (build_hist_gradient_boosting, 0.8625)],
    ids=["logistic_regression", "hist_gradient_boosting"],
)
def test_models_reproduce_notebook_02_scores(train_val_split, build, expected_roc_auc):
    X_train, X_val, y_train, y_val = train_val_split
    model = build().fit(X_train, y_train)
    roc_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    # Small tolerance so a scikit-learn upgrade does not break the test,
    # while any real change to the model definition still does.
    assert roc_auc == pytest.approx(expected_roc_auc, abs=0.002)


@needs_raw_data
def test_tuned_model_reproduces_notebook_05_cv_score():
    # Same data and folds as notebook 05: cleaned, deduplicated, add_features(),
    # stratified 5-fold CV with random_state=42.
    data = add_features(clean(drop_duplicate_rows(load_train())))
    X = data.drop(columns="SeriousDlqin2yrs")
    y = data["SeriousDlqin2yrs"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(
        build_tuned_hist_gradient_boosting(), X, y, cv=cv, scoring="roc_auc", n_jobs=-1
    )

    # Tighter than the notebook 02 test: the tuned model beats the default one
    # by only 0.001, so a looser tolerance would not notice if tuning were lost.
    assert scores.mean() == pytest.approx(0.8661, abs=0.0005)
