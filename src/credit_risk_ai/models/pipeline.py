import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier


RANDOM_STATE = 42
def build_logistic_regression():
    """docstring: what the pipeline does and the order of its steps"""
    """return Pipeline with 4 steps, in this order:"""

    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])

def build_hist_gradient_boosting(random_state = RANDOM_STATE):
    return HistGradientBoostingClassifier(random_state=random_state)


# Best hyperparameters from notebook 05: RandomizedSearchCV over 30 candidates with
# the same 5-fold CV as notebooks 03-04, rounded to readable values and re-checked
# with CV (ROC-AUC 0.8661 vs 0.8651 for the default model; train/validation gap
# 0.0125 vs 0.0223). Digits beyond these are noise: changing the 4th decimal of
# learning_rate moves the score by about +-0.0002.
#
# NOT ADOPTED: the mean gain (+0.00098) misses the decision rule fixed before the
# search (+0.001), so the final model uses build_hist_gradient_boosting() (defaults).
# Kept for reference and for later experiments.
TUNED_HGB_PARAMS = {
    "max_iter": 500,  # room for more trees at a small learning rate; early stopping picks the number
    "learning_rate": 0.067,
    "max_leaf_nodes": 21,
    "min_samples_leaf": 378,
    "l2_regularization": 1.1,
}


# skops (the safe file format used by MLflow and the Hugging Face export) only
# loads types it knows are safe. HistGradientBoosting's tree class is not on that
# list, so it must be trusted explicitly. That is safe for a model we trained
# ourselves; never trust types in a model file from an unknown source.
SKOPS_TRUSTED_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


def build_tuned_hist_gradient_boosting(random_state=RANDOM_STATE):
    """Return an unfitted HistGradientBoostingClassifier with the notebook 05 hyperparameters.

    Not the final model (see TUNED_HGB_PARAMS): it is kept for reference.
    build_hist_gradient_boosting() keeps the default settings, which notebooks
    02-04 use and the final model uses.
    """
    return HistGradientBoostingClassifier(random_state=random_state, **TUNED_HGB_PARAMS)
