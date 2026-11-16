"""Tests for credit_risk_ai.data.features.

Two parts:
  1. Small hand-made rows: checks the weighted_past_due formula and how
     add_features() treats its input.
  2. Real data (skipped when data/raw/ is missing): checks the values that
     notebook 04 found for the cleaned training data.
"""

import numpy as np
import pandas as pd
import pytest

from credit_risk_ai.data.clean import clean, drop_duplicate_rows
from credit_risk_ai.data.features import add_features
from credit_risk_ai.data.load import RAW_DIR, load_train

# Column names are written out here instead of imported from the package, so a
# typo in the package's constants makes these tests fail instead of hiding it.
LATE_30 = "NumberOfTime30-59DaysPastDueNotWorse"
LATE_60 = "NumberOfTime60-89DaysPastDueNotWorse"
LATE_90 = "NumberOfTimes90DaysLate"
NEW_COL = "weighted_past_due"


def make_rows(*late_counts):
    """One row per (30-59, 60-89, 90+) tuple, plus one unrelated column.

    The unrelated column ("MonthlyIncome") checks that add_features() leaves
    other columns alone. The index is named "id" and starts at 1, like the
    output of load_train().
    """
    rows = [
        {LATE_30: late_30, LATE_60: late_60, LATE_90: late_90, "MonthlyIncome": 5000.0}
        for late_30, late_60, late_90 in late_counts
    ]
    return pd.DataFrame(rows, index=pd.Index(range(1, len(rows) + 1), name="id"))


# =============================================================================
# Part 1: hand-made rows
# =============================================================================


def test_adds_only_weighted_past_due_and_keeps_other_columns():
    df = make_rows((1, 0, 2), (0, 0, 0))
    out = add_features(df)

    assert set(out.columns) - set(df.columns) == {NEW_COL}
    pd.testing.assert_frame_equal(out[df.columns], df)


def test_input_dataframe_is_not_modified():
    df = make_rows((1, 1, 1))
    before = df.copy()
    add_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_rows_and_index_are_kept():
    df = make_rows((0, 0, 0), (1, 2, 3), (4, 0, 0))
    out = add_features(df)
    assert out.index.equals(df.index)


@pytest.mark.parametrize(
    "late_counts, expected",
    [
        ((0, 0, 0), 0),  # never late
        ((1, 0, 0), 1),  # one 30-59 day delay  -> weight 1
        ((0, 1, 0), 2),  # one 60-89 day delay  -> weight 2
        ((0, 0, 1), 3),  # one 90+ day delay    -> weight 3
        ((1, 1, 1), 6),  # 1 + 2 + 3
        ((2, 0, 1), 5),  # 2x1 + 0x2 + 1x3
    ],
)
def test_weighted_past_due_formula(late_counts, expected):
    # One-hot rows (only one column non-zero) check that each weight is
    # attached to the right column; a swapped weight would fail here.
    out = add_features(make_rows(late_counts))
    assert out.loc[1, NEW_COL] == expected


def test_each_row_gets_its_own_value():
    # Guards against summing down the whole column (Series.sum()), which
    # gives every row the same number.
    out = add_features(make_rows((0, 0, 0), (1, 1, 1), (2, 0, 1)))
    assert out[NEW_COL].tolist() == [0, 6, 5]


@pytest.mark.parametrize(
    "late_counts",
    [(np.nan, np.nan, np.nan), (1, np.nan, 0), (0, 0, np.nan)],
    ids=["all_missing", "60_89_missing", "90_missing"],
)
def test_missing_past_due_gives_missing(late_counts):
    # clean() turns the 96/98 codes into NaN; the feature must stay missing
    # instead of silently treating NaN as 0.
    out = add_features(make_rows(late_counts))
    assert np.isnan(out.loc[1, NEW_COL])


# =============================================================================
# Part 2: real data (skipped when data/raw/ is missing)
# =============================================================================

needs_raw_data = pytest.mark.skipif(
    not (RAW_DIR / "cs-training.csv").exists(),
    reason="raw data not found in data/raw/",
)


@pytest.fixture(scope="module")
def real_features():
    """Cleaned, deduplicated training data with the engineered feature."""
    return add_features(clean(drop_duplicate_rows(load_train())))


@needs_raw_data
def test_real_data_values_match_notebook_04(real_features):
    weighted = real_features[NEW_COL]
    assert weighted.isna().sum() == 225
    assert weighted.min() == 0
    assert weighted.max() == 51


@needs_raw_data
def test_real_data_is_missing_exactly_where_clean_found_a_past_due_code(real_features):
    is_missing = real_features[NEW_COL].isna()
    has_code = real_features["past_due_code"] == 1
    assert (is_missing == has_code).all()
