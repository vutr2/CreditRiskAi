from credit_risk_ai.data.clean import PAST_DUE_COLS
PAST_DUE_WEIGHTS = [1, 2, 3]

def add_features(df):
    out = df.copy()
    out["weighted_past_due"] = 0
    for col, weight in zip(PAST_DUE_COLS, PAST_DUE_WEIGHTS):
        out["weighted_past_due"] += weight * out[col]
    return out