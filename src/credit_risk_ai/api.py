"""HTTP API that scores one loan applicant with the trained model.

Start it from the project root, after `python -m credit_risk_ai.train` has saved the model:

    uvicorn credit_risk_ai.api:app --reload

Then open http://127.0.0.1:8000/docs to send requests from the browser.

Each applicant goes through exactly the same clean() and add_features() as the
training data, so the model gets the same columns, built the same way, as when
it was trained (no training/serving skew).
"""

from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from credit_risk_ai.data.clean import clean
from credit_risk_ai.data.features import add_features
from credit_risk_ai.paths import MODEL_PATH

# API field name -> column name in the Kaggle data (the names clean() expects)
FIELD_TO_COLUMN = {
    "revolving_utilization": "RevolvingUtilizationOfUnsecuredLines",
    "age": "age",
    "times_30_59_days_late": "NumberOfTime30-59DaysPastDueNotWorse",
    "debt_ratio": "DebtRatio",
    "monthly_income": "MonthlyIncome",
    "open_credit_lines": "NumberOfOpenCreditLinesAndLoans",
    "times_90_days_late": "NumberOfTimes90DaysLate",
    "real_estate_loans": "NumberRealEstateLoansOrLines",
    "times_60_89_days_late": "NumberOfTime60-89DaysPastDueNotWorse",
    "dependents": "NumberOfDependents",
}


class Applicant(BaseModel):
    """One loan applicant: the 10 input fields of the Kaggle data.

    Unusual values (age 0, utilization 50,000, past-due codes 96/98, income 0)
    are accepted on purpose: clean() handles them the same way as in training.
    """

    # A misspelled field name is an error instead of being silently ignored
    model_config = ConfigDict(extra="forbid")

    revolving_utilization: float = Field(ge=0, description="Credit card balances / credit limits")
    age: int = Field(ge=0, description="Age in years")
    times_30_59_days_late: int = Field(ge=0, description="Times 30-59 days late (last 2 years)")
    debt_ratio: float = Field(ge=0, description="Monthly debt payments / monthly income")
    monthly_income: float | None = Field(default=None, ge=0, description="Monthly income; leave out if unknown")
    open_credit_lines: int = Field(ge=0, description="Open loans and lines of credit")
    times_90_days_late: int = Field(ge=0, description="Times 90 or more days late")
    real_estate_loans: int = Field(ge=0, description="Mortgage and real estate loans")
    times_60_89_days_late: int = Field(ge=0, description="Times 60-89 days late (last 2 years)")
    dependents: int | None = Field(default=None, ge=0, description="Number of dependents; leave out if unknown")


class Prediction(BaseModel):
    probability_of_default: float = Field(
        description="Probability of being 90+ days late within the next 2 years"
    )


def applicant_to_features(applicant):
    """One-row feature table, built exactly like the training data."""
    row = {FIELD_TO_COLUMN[field]: value for field, value in applicant.model_dump().items()}
    raw = pd.DataFrame([row], dtype=float)  # a missing value (None) becomes NaN
    return add_features(clean(raw))


def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}. Train it first with: python -m credit_risk_ai.train"
        )
    return joblib.load(MODEL_PATH)


def create_app(model=None):
    """Build the API. With model=None the saved model is loaded at startup; tests pass a fitted model."""

    @asynccontextmanager
    async def lifespan(app):
        # Load the model once, when the server starts, not on every request
        app.state.model = model if model is not None else load_model()
        yield

    app = FastAPI(
        title="Credit risk API",
        description="Probability that a borrower becomes 90+ days late within 2 years "
        "(Give Me Some Credit model).",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict", response_model=Prediction)
    def predict(applicant: Applicant, request: Request):
        model = request.app.state.model
        X = applicant_to_features(applicant)
        proba = model.predict_proba(X[model.feature_names_in_])[0, 1]
        return Prediction(probability_of_default=float(proba))

    return app


app = create_app()
