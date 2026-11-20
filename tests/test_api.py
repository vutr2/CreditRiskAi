"""Tests for credit_risk_ai.api.

Two parts:
  1. A small model trained on synthetic rows, so these run without the data:
     endpoints, input validation and the missing-model error.
  2. Real data (skipped when data/raw/ is missing): the API must give exactly
     the same probability as batch prediction on Kaggle test rows, i.e. no
     training/serving skew.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import HistGradientBoostingClassifier

import credit_risk_ai.api as api
from credit_risk_ai.api import FIELD_TO_COLUMN, create_app
from credit_risk_ai.data.clean import clean
from credit_risk_ai.data.features import add_features
from credit_risk_ai.data.load import RAW_DIR, load_test
from credit_risk_ai.data.prepare import prepare_test, prepare_train
from credit_risk_ai.models.pipeline import build_hist_gradient_boosting
from credit_risk_ai.train import make_submission

VALID_APPLICANT = {
    "revolving_utilization": 0.3,
    "age": 45,
    "times_30_59_days_late": 0,
    "debt_ratio": 0.35,
    "monthly_income": 5000,
    "open_credit_lines": 8,
    "times_90_days_late": 0,
    "real_estate_loans": 1,
    "times_60_89_days_late": 0,
    "dependents": 1,
}


def fit_small_model():
    """A model trained on 200 synthetic rows that went through clean() and add_features()."""
    rng = np.random.default_rng(0)
    n_rows = 200
    raw = pd.DataFrame(
        {column: rng.integers(0, 5, size=n_rows).astype(float) for column in FIELD_TO_COLUMN.values()}
    )
    raw["RevolvingUtilizationOfUnsecuredLines"] = rng.uniform(0, 1.5, size=n_rows)
    raw["age"] = rng.integers(20, 80, size=n_rows).astype(float)
    raw["MonthlyIncome"] = rng.lognormal(8.5, 0.6, size=n_rows)
    X = add_features(clean(raw))
    y = (np.arange(n_rows) % 4 == 0).astype(int)
    return HistGradientBoostingClassifier(random_state=0).fit(X, y)


@pytest.fixture(scope="module")
def client():
    # "with" runs the app's startup (lifespan), like a real server
    with TestClient(create_app(model=fit_small_model())) as test_client:
        yield test_client


# =============================================================================
# Part 1: synthetic model
# =============================================================================


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_one_probability(client):
    response = client.post("/predict", json=VALID_APPLICANT)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"probability_of_default"}
    assert 0 <= body["probability_of_default"] <= 1


def test_unknown_income_and_dependents_can_be_left_out(client):
    applicant = {k: v for k, v in VALID_APPLICANT.items() if k not in ("monthly_income", "dependents")}
    response = client.post("/predict", json=applicant)
    assert response.status_code == 200


def test_unusual_values_are_accepted_because_clean_handles_them(client):
    # The same oddities as in the training data: age 0, utilization 50,708, code 98
    applicant = VALID_APPLICANT | {"age": 0, "revolving_utilization": 50_708, "times_90_days_late": 98}
    response = client.post("/predict", json=applicant)
    assert response.status_code == 200


@pytest.mark.parametrize(
    "change",
    [
        {"debt_ratio": -1},            # negative value
        {"age": "forty"},              # wrong type
        {"monthly_incom": 5000},       # misspelled field name
    ],
    ids=["negative_value", "wrong_type", "unknown_field"],
)
def test_invalid_input_is_rejected(client, change):
    response = client.post("/predict", json=VALID_APPLICANT | change)
    assert response.status_code == 422


def test_missing_required_field_is_rejected(client):
    applicant = {k: v for k, v in VALID_APPLICANT.items() if k != "age"}
    response = client.post("/predict", json=applicant)
    assert response.status_code == 422


def test_missing_model_file_gives_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing.joblib")
    with pytest.raises(RuntimeError, match="python -m credit_risk_ai.train"):
        with TestClient(create_app()):
            pass


# =============================================================================
# Part 2: real data (skipped when data/raw/ is missing)
# =============================================================================

needs_raw_data = pytest.mark.skipif(
    not (RAW_DIR / "cs-training.csv").exists(),
    reason="raw data not found in data/raw/",
)


@needs_raw_data
def test_api_gives_the_same_probability_as_batch_prediction():
    X, y = prepare_train()
    model = build_hist_gradient_boosting().fit(X, y)
    expected = make_submission(model, prepare_test())["Probability"].to_numpy()

    raw_test = load_test()
    column_to_field = {column: field for field, column in FIELD_TO_COLUMN.items()}
    late_90 = FIELD_TO_COLUMN["times_90_days_late"]
    positions = [
        0,
        1,
        2,
        int(np.flatnonzero(raw_test["MonthlyIncome"].isna())[0]),  # income unknown
        int(np.flatnonzero(raw_test[late_90] >= 96)[0]),           # past-due code 96/98
    ]

    with TestClient(create_app(model=model)) as client:
        for position in positions:
            row = raw_test.iloc[position]
            applicant = {column_to_field[c]: (None if pd.isna(v) else v) for c, v in row.items()}
            response = client.post("/predict", json=applicant)
            assert response.status_code == 200
            assert response.json()["probability_of_default"] == pytest.approx(
                expected[position], abs=1e-12
            )
