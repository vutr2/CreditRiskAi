"""Model-ready data: the same steps everywhere (final training, notebooks, tests).

    X, y = prepare_train()      # load -> drop duplicates -> clean -> add_features
    X_test = prepare_test()     # load -> clean -> add_features (no rows dropped)
"""

from credit_risk_ai.data.clean import clean, drop_duplicate_rows
from credit_risk_ai.data.features import add_features
from credit_risk_ai.data.load import load_test, load_train

TARGET = "SeriousDlqin2yrs"


def prepare_train():
    """Cleaned, deduplicated training data with engineered features, as (X, y)."""
    data = add_features(clean(drop_duplicate_rows(load_train())))
    return data.drop(columns=TARGET), data[TARGET]


def prepare_test():
    """Cleaned Kaggle test data with engineered features.

    Duplicates are NOT removed: Kaggle needs a prediction for every test row.
    """
    return add_features(clean(load_test()))
