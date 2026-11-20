# CreditRiskAI

Predicts the probability that a borrower becomes **90 or more days late on a payment within the next two years**, using Kaggle's [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit) data.

A learning project that goes from a data audit to a tested model, experiment tracking with MLflow and an HTTP API with FastAPI.

## Results

Final model: scikit-learn `HistGradientBoostingClassifier` (default hyperparameters) on cleaned data plus one engineered feature, trained on all 149,391 deduplicated training rows.

| Evaluation | ROC-AUC |
|---|---|
| 5-fold cross-validation (training data) | 0.8651 ± 0.0044 |
| Kaggle public leaderboard | 0.86247 |
| Kaggle private leaderboard | **0.86780** |

The competition's winning private score was about 0.870. The Kaggle scores fall inside the cross-validation range, so the validation setup predicted performance on unseen data correctly.

On out-of-fold predictions the model also has PR-AUC 0.396 (0.067 without a model) and a Brier score of 0.0492 (0.0625 when everyone gets the base rate), and its probabilities are well calibrated.

## How the model was built

Each notebook answers one question and ends with written conclusions.

| Notebook | Question | Answer |
|---|---|---|
| [01_data_audit](notebooks/01_data_audit.ipynb) | What is wrong with the data? | 12 issues, e.g. past-due codes 96/98, utilization above 10, income ≤ 1, age below 18, 609 duplicate rows, only 6.7% defaults |
| [02_baseline_model](notebooks/02_baseline_model.ipynb) | Do simple models work? | Logistic regression 0.855 and HistGradientBoosting 0.862 ROC-AUC on an 80/20 split; the default 0.5 threshold catches only 19% of defaulters |
| [03_cross_validation](notebooks/03_cross_validation.ipynb) | Is HistGradientBoosting really better? | Yes: +0.0098 ROC-AUC, better on 5 of 5 folds |
| [04_feature_engineer](notebooks/04_feature_engineer.ipynb) | Do new features help? | Of 5 candidates, only `weighted_past_due` passed the decision rule (+0.003 for logistic regression, no change for HistGradientBoosting) |
| [05_tune_HistGradientBoosting](notebooks/05_tune_HistGradientBoosting.ipynb) | Does tuning help? | It halves overfitting but adds only +0.00098 ROC-AUC, below the decision rule, so the defaults were kept |
| [06_evaluation_and_threshold](notebooks/06_evaluation_and_threshold.ipynb) | Can the probabilities be trusted, and where should applications be rejected? | Well calibrated; with a missed defaulter costing 5 times a wrongly rejected applicant, reject from a probability of 0.167 |

Every comparison uses the same 5 folds and a decision rule fixed before the experiment (at least 4 of 5 folds better and a mean gain of at least +0.001 ROC-AUC).

## Project structure

```
src/credit_risk_ai/
├── data/
│   ├── load.py        # read the Kaggle CSV files
│   ├── clean.py       # audit rules: invalid values -> NaN + flag columns, duplicate removal
│   ├── features.py    # add_features(): weighted_past_due
│   └── prepare.py     # load -> clean -> add_features, the same for every use
├── models/
│   └── pipeline.py    # functions that build unfitted models
├── train.py           # final training, Kaggle submission, MLflow logging
├── api.py             # FastAPI app: POST /predict, GET /health
├── export_hf.py       # model.skops + model card for the Hugging Face Hub
└── paths.py           # where generated files go
notebooks/             # 01-06, see the table above
tests/                 # pytest; real-data tests are skipped when the data is missing
```

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/vutr2/CreditRiskAi.git
cd CreditRiskAi
uv sync
```

The data is not included (Kaggle competition rules). Download it from the [competition page](https://www.kaggle.com/competitions/GiveMeSomeCredit/data) into `data/raw/`: `cs-training.csv`, `cs-test.csv`, `sampleEntry.csv` and `Data Dictionary.xls`.

## Usage

```bash
uv run pytest                                      # run the tests
uv run python -m credit_risk_ai.train              # train, save models/hgb.joblib, write submissions/submission.csv
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse runs at http://127.0.0.1:5000
uv run uvicorn credit_risk_ai.api:app --reload     # API; interactive docs at http://127.0.0.1:8000/docs
```

Example request once the API is running:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"revolving_utilization": 0.3, "age": 45, "times_30_59_days_late": 0,
       "debt_ratio": 0.35, "monthly_income": 5000, "open_credit_lines": 8,
       "times_90_days_late": 0, "real_estate_loans": 1,
       "times_60_89_days_late": 0, "dependents": 1}'
# {"probability_of_default": 0.0247...}
```

`monthly_income` and `dependents` may be left out when unknown. Every request goes through the same `clean()` and `add_features()` as the training data, and a test checks that the API returns exactly the same probabilities as batch prediction.

## Limitations

- For learning and demonstration only; not for real lending decisions.
- The data come from a 2011 Kaggle competition about US borrowers, and how it was collected is not documented.
- The model uses age as an input. Many countries restrict the use of age in credit decisions, and no fairness analysis was done.
- The 0.167 threshold rests on an assumed 5:1 cost ratio; a real lender would set the ratio from its own losses.
