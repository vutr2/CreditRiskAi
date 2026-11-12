"""Row-level cleaning rules for the Give Me Some Credit data.

Every rule here comes from the data audit (notebooks/01_data_audit.ipynb, section 7).

`clean()` only replaces invalid values with NaN and adds 0/1 flag columns.
It never fills in values and never drops rows. Filling in (imputation) and
capping need statistics such as a median, and those must be computed on the
training data only, so they belong in the model pipeline, not here.

Typical use:

    from credit_risk_ai.data.load import load_train, load_test
    from credit_risk_ai.data.clean import clean, drop_duplicate_rows

    train = clean(drop_duplicate_rows(load_train()))
    test = clean(load_test())
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Thresholds found in the data audit
# ---------------------------------------------------------------------------

# The three past-due count columns. On 269 training rows all three hold 96 or
# 98. These are status codes, not counts: real counts never go above 17.
PAST_DUE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
PAST_DUE_CODES = [96, 98]

# Incomes of 0 or 1 are placeholders: their DebtRatio behaves exactly like
# rows with no income at all. So income <= 1 is treated as missing.
MIN_VALID_INCOME = 1

# Utilization = credit card balance / credit limit. Values between 1 and 10 are
# real borrowers over their limit (the riskiest group), so they are kept.
# Values above 10 (up to 50,708) cannot be ratios and are treated as errors.
MAX_UTILIZATION = 10

# A borrower cannot be younger than 18 (one training row has age 0).
MIN_AGE = 18

# Integer columns that may receive NaN. Integer columns cannot hold NaN, so
# clean() converts them to float up front. Doing it always, not only when a
# row is affected, gives train and test the same column types.
INT_COLS_THAT_MAY_GET_NAN = PAST_DUE_COLS + ["age"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the audit's row-level cleaning rules and return a new DataFrame.

    Safe for both the training and the test set: it never uses the target
    column, never drops rows, and never modifies the DataFrame passed in.

    Invalid values are replaced by NaN, and four 0/1 flag columns are added:

        past_due_code        the past-due columns held a 96/98 status code
        income_missing       MonthlyIncome is missing, 0 or 1
        dependents_missing   NumberOfDependents is missing
        utilization_invalid  RevolvingUtilizationOfUnsecuredLines was above 10
    """
    # Work on a copy so the caller's DataFrame is never changed.
    out = df.copy()
    out[INT_COLS_THAT_MAY_GET_NAN] = out[INT_COLS_THAT_MAY_GET_NAN].astype(float)

    # Each rule below follows the same pattern:
    #   1. build a True/False mask that marks the bad rows,
    #   2. write the flag column from that mask,
    #   3. use the same mask to replace the bad values with NaN.
    # Because the flag and the replacement share one mask, the flag always
    # describes the original values.

    # --- Rule 7: past-due status codes 96 / 98 ------------------------------
    # The codes are highly predictive (54.6% default rate), so keep that
    # information as a flag, then blank the three counts because 96/98 are
    # not real counts.
    has_code = out[PAST_DUE_COLS].isin(PAST_DUE_CODES).any(axis=1)
    out["past_due_code"] = has_code.astype(int)
    out.loc[has_code, PAST_DUE_COLS] = np.nan

    # --- Rule 4: incomes of 0 or 1 are placeholders -> missing --------------
    # Rows where income is already NaN are not affected, because a comparison
    # with NaN is always False.
    invalid_income = out["MonthlyIncome"] <= MIN_VALID_INCOME
    out.loc[invalid_income, "MonthlyIncome"] = np.nan

    # --- Rule 3: flag missing income -----------------------------------------
    # ORDER MATTERS: this must run after rule 4, so that the 0 / 1 incomes
    # just turned into NaN are flagged as missing too.
    income_missing = out["MonthlyIncome"].isna()
    out["income_missing"] = income_missing.astype(int)

    # --- Rule 6: DebtRatio is meaningless without income ---------------------
    # DebtRatio = monthly debt / monthly income. Without a usable income the
    # stored values look like dollar amounts instead of ratios.
    # ORDER MATTERS: this must also run after rule 4.
    out.loc[income_missing, "DebtRatio"] = np.nan

    # --- Rule 5: flag missing dependents -------------------------------------
    # The value itself stays NaN; it is filled in later by the model pipeline.
    out["dependents_missing"] = out["NumberOfDependents"].isna().astype(int)

    # --- Rule 8: utilization above 10 is a data error ------------------------
    invalid_util = out["RevolvingUtilizationOfUnsecuredLines"] > MAX_UTILIZATION
    out["utilization_invalid"] = invalid_util.astype(int)
    out.loc[invalid_util, "RevolvingUtilizationOfUnsecuredLines"] = np.nan

    # --- Rule 9: impossible ages -> missing ----------------------------------
    # No flag here: only one training row is affected.
    out.loc[out["age"] < MIN_AGE, "age"] = np.nan

    return out


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows, keeping the first copy (the lowest id).

    Use it on the raw training data only, before `clean()` and before any
    train/validation split, so the same row cannot end up on both sides of
    the split. Never use it on the test set: the submission needs every row.

    pandas' default behaviour already matches what the audit needs:
      - only columns are compared, not the index, so `id` is ignored;
      - the target is compared, so rows that differ only in the target are kept;
      - NaN counts as equal to NaN, so rows with the same missing values match.
    """
    return df.drop_duplicates(keep="first")
