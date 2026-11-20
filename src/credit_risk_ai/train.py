"""Train the final model, write the Kaggle submission and log the run to MLflow.

The final model follows the decisions of notebooks 03-05: the default
HistGradientBoosting (build_hist_gradient_boosting) on the cleaned data plus
weighted_past_due. Its 5-fold CV score (same folds as the notebooks) is logged
as the expected performance; then it is trained on ALL training rows.

Run (from any folder):

    python -m credit_risk_ai.train

Then look at the logged runs (from the project root):

    mlflow ui --backend-store-uri sqlite:///mlflow.db     -> http://127.0.0.1:5000
"""

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from credit_risk_ai.data.prepare import prepare_test, prepare_train
from credit_risk_ai.models.pipeline import RANDOM_STATE, build_hist_gradient_boosting
from credit_risk_ai.paths import (
    MLFLOW_ARTIFACT_DIR,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    SUBMISSION_PATH,
)

N_SPLITS = 5
EXPERIMENT_NAME = "credit-risk-final-model"

# MLflow saves sklearn models in the skops format, which only loads types it
# knows are safe. HistGradientBoosting's tree class is not on that list, so it
# must be trusted explicitly. That is safe here because we train the model
# ourselves; never trust types in a model file from an unknown source.
SKOPS_TRUSTED_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


def cross_validate_model(build, X, y):
    """5-fold stratified CV with the same folds as notebooks 03-05."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return cross_validate(
        build(), X, y, cv=cv, scoring="roc_auc", return_train_score=True, n_jobs=-1
    )


def make_submission(model, X_test):
    """Kaggle submission: one row per test applicant, columns Id and Probability."""
    # Use the training column order: sklearn refuses columns in a different order
    proba = model.predict_proba(X_test[model.feature_names_in_])[:, 1]
    return pd.DataFrame({"Id": X_test.index, "Probability": proba})


def log_run(model, X, cv_results, submission_path):
    """Log parameters, CV metrics, the submission and the model to MLflow; return the run id."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        # Fixed folder for saved files, so they never land in whatever folder
        # the script happens to be run from
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACT_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    val = cv_results["test_score"]
    train = cv_results["train_score"]

    with mlflow.start_run(run_name="hgb-default") as run:
        mlflow.log_params({
            "model": type(model).__name__,
            "features": ", ".join(X.columns),
            "n_train_rows": len(X),
            "n_features": X.shape[1],
            "cv_folds": N_SPLITS,
        })
        # Every hyperparameter of the model, prefixed so they are easy to find in the UI
        mlflow.log_params({f"hgb_{name}": value for name, value in model.get_params().items()})
        mlflow.log_metrics({
            "cv_roc_auc_mean": val.mean(),
            "cv_roc_auc_std": val.std(),
            "cv_train_val_gap": train.mean() - val.mean(),
        })
        mlflow.log_artifact(str(submission_path))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            # As floats: an integer column in the logged schema could not hold a missing value
            input_example=X.head(5).astype(float),
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

    return run.info.run_id


def main():
    X, y = prepare_train()
    print(f"Training data: {X.shape[0]:,} rows, {X.shape[1]} features")

    cv_results = cross_validate_model(build_hist_gradient_boosting, X, y)
    val = cv_results["test_score"]
    print(f"5-fold CV ROC-AUC: {val.mean():.4f} ± {val.std():.4f}")

    model = build_hist_gradient_boosting().fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    submission = make_submission(model, prepare_test())
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission ({len(submission):,} rows) saved to {SUBMISSION_PATH}")

    run_id = log_run(model, X, cv_results, SUBMISSION_PATH)
    print(f"MLflow run {run_id} logged (experiment '{EXPERIMENT_NAME}')")


if __name__ == "__main__":
    main()
