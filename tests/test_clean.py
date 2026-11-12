"""Tests for credit_risk_ai.data.clean.

Two parts:
  1. Small hand-made rows, one test per audit rule. make_rows() builds rows
     with normal, valid values; each test changes only the fields it is about.
  2. Checks on the real data. They are skipped when data/raw/ is missing.
"""

import numpy as np
import pandas as pd
import pytest

from credit_risk_ai.data.clean import clean, drop_duplicate_rows
from credit_risk_ai.data.load import RAW_DIR, load_test, load_train

# Column names are written out here instead of imported from clean.py, so a
# typo in clean.py's constants would make these tests fail instead of hiding it.
TARGET = "SeriousDlqin2yrs"
UTIL = "RevolvingUtilizationOfUnsecuredLines"
PAST_DUE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
FLAGS = ["past_due_code", "income_missing", "dependents_missing", "utilization_invalid"]


def make_rows(*overrides):
    """One row per dict in overrides; unspecified columns get normal, valid values.

    The result looks like load_train() output: same columns, same order,
    same dtypes, and an index named "id" starting at 1.
    """
    base = {
        TARGET: 0,
        UTIL: 0.3,
        "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.35,
        "MonthlyIncome": 5000.0,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1.0,
    }
    rows = [{**base, **o} for o in (overrides or [{}])]
    return pd.DataFrame(rows, index=pd.Index(range(1, len(rows) + 1), name="id"))


# =============================================================================
# Part 1: hand-made rows
# =============================================================================

# --- General behaviour of clean() -------------------------------------------


def test_valid_row_is_unchanged_and_flags_are_zero():
    df = make_rows()
    out = clean(df)
    # Compare values only: clean() always turns some int columns into float.
    pd.testing.assert_frame_equal(out[df.columns], df, check_dtype=False)
    # Exactly the four flag columns are added, all 0 for a valid row.
    assert set(out.columns) - set(df.columns) == set(FLAGS)
    assert (out.loc[1, FLAGS] == 0).all()


def test_input_dataframe_is_not_modified():
    df = make_rows({"age": 0, "MonthlyIncome": 0, UTIL: 50_708})
    before = df.copy()
    clean(df)
    pd.testing.assert_frame_equal(df, before)


def test_rows_and_index_are_kept():
    df = make_rows({}, {"age": 0}, {UTIL: 50_708})
    out = clean(df)
    assert out.index.equals(df.index)


def test_works_without_target_column():
    # The test set has no target column; clean() must not need it.
    out = clean(make_rows().drop(columns=TARGET))
    assert TARGET not in out.columns


def test_column_types_do_not_depend_on_the_data():
    # Train has an age-0 row and test does not. Both must still come out with
    # the same column types.
    clean_rows = clean(make_rows())
    bad_rows = clean(make_rows({"age": 0}, {col: 98 for col in PAST_DUE_COLS}))
    assert clean_rows.dtypes.equals(bad_rows.dtypes)


# --- Rules 3 and 4: income ----------------------------------------------------


@pytest.mark.parametrize(
    "income, becomes_nan, flag",
    [
        (np.nan, True, 1),  # already missing
        (0, True, 1),       # placeholder
        (1, True, 1),       # placeholder
        (2, False, 0),      # the cut-off is <= 1, so 2 is kept
    ],
    ids=["missing", "zero", "one", "two"],
)
def test_income(income, becomes_nan, flag):
    out = clean(make_rows({"MonthlyIncome": income}))
    assert pd.isna(out.loc[1, "MonthlyIncome"]) == becomes_nan
    assert out.loc[1, "income_missing"] == flag


# --- Rule 5: dependents -------------------------------------------------------


def test_missing_dependents_are_flagged_but_not_filled_in():
    out = clean(make_rows({"NumberOfDependents": np.nan}))
    assert pd.isna(out.loc[1, "NumberOfDependents"])
    assert out.loc[1, "dependents_missing"] == 1


# --- Rule 6: DebtRatio --------------------------------------------------------


@pytest.mark.parametrize(
    "income, debt_ratio, becomes_nan",
    [
        (np.nan, 1159, True),  # no income -> DebtRatio is not a ratio
        (0, 800, True),        # checks the order: income 0 must become NaN first
        (5000, 2.5, False),    # real long tail with a reported income is kept
    ],
    ids=["income-missing", "income-zero", "income-reported"],
)
def test_debt_ratio(income, debt_ratio, becomes_nan):
    out = clean(make_rows({"MonthlyIncome": income, "DebtRatio": debt_ratio}))
    assert pd.isna(out.loc[1, "DebtRatio"]) == becomes_nan
    if not becomes_nan:
        assert out.loc[1, "DebtRatio"] == debt_ratio


# --- Rule 7: past-due codes 96 / 98 -------------------------------------------


@pytest.mark.parametrize("code", [96, 98])
def test_past_due_codes_are_flagged_and_set_to_missing(code):
    out = clean(make_rows({col: code for col in PAST_DUE_COLS}))
    # The flag must describe the original values, even though they are removed.
    assert out.loc[1, "past_due_code"] == 1
    assert out.loc[1, PAST_DUE_COLS].isna().all()


def test_real_high_past_due_counts_are_kept():
    # 13, 11 and 17 are the largest real counts in the training data.
    counts = dict(zip(PAST_DUE_COLS, [13, 11, 17]))
    out = clean(make_rows(counts))
    assert out.loc[1, "past_due_code"] == 0
    assert out.loc[1, PAST_DUE_COLS].tolist() == [13, 11, 17]


# --- Rule 8: utilization ------------------------------------------------------


@pytest.mark.parametrize(
    "utilization, becomes_nan, flag",
    [
        (50_708, True, 1),  # cannot be a ratio -> data error
        (1.5, False, 0),    # over the limit: real and the riskiest group
        (10, False, 0),     # the cut-off is > 10, so exactly 10 is kept
    ],
    ids=["error", "over-limit", "boundary"],
)
def test_utilization(utilization, becomes_nan, flag):
    out = clean(make_rows({UTIL: utilization}))
    assert pd.isna(out.loc[1, UTIL]) == becomes_nan
    assert out.loc[1, "utilization_invalid"] == flag


# --- Rule 9: age --------------------------------------------------------------


@pytest.mark.parametrize("age, becomes_nan", [(0, True), (17, True), (18, False)])
def test_age(age, becomes_nan):
    out = clean(make_rows({"age": age}))
    assert pd.isna(out.loc[1, "age"]) == becomes_nan


# --- Rule 10: drop_duplicate_rows() -------------------------------------------


def test_identical_rows_keep_only_the_first():
    out = drop_duplicate_rows(make_rows({}, {}))
    assert out.index.tolist() == [1]


def test_rows_that_differ_only_in_target_are_both_kept():
    out = drop_duplicate_rows(make_rows({}, {TARGET: 1}))
    assert len(out) == 2


def test_rows_with_the_same_missing_values_are_duplicates():
    # 85% of the real duplicates have missing income, so NaN must match NaN.
    out = drop_duplicate_rows(make_rows({"MonthlyIncome": np.nan}, {"MonthlyIncome": np.nan}))
    assert len(out) == 1


# =============================================================================
# Part 2: real data (skipped when data/raw/ is missing)
# =============================================================================

needs_raw_data = pytest.mark.skipif(
    not (RAW_DIR / "cs-training.csv").exists() or not (RAW_DIR / "cs-test.csv").exists(),
    reason="raw data not found in data/raw/",
)


@pytest.fixture(scope="module")
def raw_train():
    return load_train()


@needs_raw_data
def test_real_train_has_no_invalid_values_left(raw_train):
    out = clean(raw_train)
    assert len(out) == len(raw_train)
    assert not (out["MonthlyIncome"] <= 1).any()
    assert not out[PAST_DUE_COLS].isin([96, 98]).any().any()
    assert not (out[UTIL] > 10).any()
    assert not (out["age"] < 18).any()
    # DebtRatio may only have a value where there is a usable income.
    assert out.loc[out["MonthlyIncome"].isna(), "DebtRatio"].isna().all()


@needs_raw_data
def test_real_train_flag_counts_match_the_audit(raw_train):
    flags = clean(raw_train)[FLAGS].sum()
    assert flags["past_due_code"] == 269
    assert flags["income_missing"] == 29_731 + 2_239  # missing + income 0 or 1
    assert flags["dependents_missing"] == 3_924
    assert flags["utilization_invalid"] == 241


@needs_raw_data
def test_real_train_drops_the_609_duplicates(raw_train):
    assert len(raw_train) - len(drop_duplicate_rows(raw_train)) == 609


@needs_raw_data
def test_real_train_and_test_get_the_same_columns_and_types(raw_train):
    train = clean(raw_train).drop(columns=TARGET)
    test = clean(load_test())
    assert list(test.columns) == list(train.columns)
    assert test.dtypes.equals(train.dtypes)
