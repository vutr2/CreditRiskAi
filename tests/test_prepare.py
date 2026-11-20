"""Tests for credit_risk_ai.data.prepare (real data; skipped when data/raw/ is missing)."""

import pytest

from credit_risk_ai.data.load import RAW_DIR
from credit_risk_ai.data.prepare import TARGET, prepare_test, prepare_train

needs_raw_data = pytest.mark.skipif(
    not (RAW_DIR / "cs-training.csv").exists(),
    reason="raw data not found in data/raw/",
)


@pytest.fixture(scope="module")
def prepared():
    X, y = prepare_train()
    return X, y, prepare_test()


@needs_raw_data
def test_train_is_deduplicated_cleaned_and_has_the_engineered_feature(prepared):
    X, y, _ = prepared
    assert X.shape == (149_391, 15)
    assert TARGET not in X.columns
    assert "weighted_past_due" in X.columns
    assert y.name == TARGET


@needs_raw_data
def test_train_and_test_get_the_same_columns_in_the_same_order(prepared):
    X, _, X_test = prepared
    assert list(X_test.columns) == list(X.columns)


@needs_raw_data
def test_no_test_row_is_dropped(prepared):
    _, _, X_test = prepared
    assert len(X_test) == 101_503
