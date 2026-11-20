"""Tests for credit_risk_ai.train (the parts that do not need MLflow).

Two parts:
  1. A tiny model on synthetic data: checks make_submission().
  2. Real data (skipped when data/raw/ is missing): the submission matches
     Kaggle's sampleEntry.csv format.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from credit_risk_ai.data.load import RAW_DIR
from credit_risk_ai.data.prepare import prepare_test, prepare_train
from credit_risk_ai.train import make_submission


# =============================================================================
# Part 1: synthetic data
# =============================================================================


def test_make_submission_has_the_kaggle_format_and_reorders_columns():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 3)), columns=["a", "b", "c"])
    y = (np.arange(100) % 4 == 0).astype(int)
    model = HistGradientBoostingClassifier(random_state=0).fit(X, y)

    # Test rows with Kaggle-style ids and the columns in a different order
    X_test = pd.DataFrame(
        rng.normal(size=(10, 3)),
        columns=["c", "a", "b"],
        index=pd.Index(range(1, 11), name="id"),
    )
    submission = make_submission(model, X_test)

    assert list(submission.columns) == ["Id", "Probability"]
    assert submission["Id"].tolist() == list(range(1, 11))
    assert submission["Probability"].between(0, 1).all()
    # Same probabilities as with the columns in training order
    expected = model.predict_proba(X_test[["a", "b", "c"]])[:, 1]
    assert np.allclose(submission["Probability"], expected)


# =============================================================================
# Part 2: real data (skipped when data/raw/ is missing)
# =============================================================================

needs_raw_data = pytest.mark.skipif(
    not (RAW_DIR / "cs-training.csv").exists(),
    reason="raw data not found in data/raw/",
)


@needs_raw_data
def test_submission_matches_kaggle_sample_entry():
    X, y = prepare_train()
    # A deliberately small, fast model: this test is about the file format
    model = HistGradientBoostingClassifier(max_iter=10, random_state=0).fit(X, y)
    submission = make_submission(model, prepare_test())
    sample = pd.read_csv(RAW_DIR / "sampleEntry.csv")

    assert list(submission.columns) == list(sample.columns)
    assert submission["Id"].tolist() == sample["Id"].tolist()
    assert submission["Probability"].between(0, 1).all()
