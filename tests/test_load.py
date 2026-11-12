"""Tests for credit_risk_ai.data.load, run against the real CSV files.

They read data/raw/cs-training.csv and data/raw/cs-test.csv. data/raw/ is
gitignored, so on a machine without the data (e.g. a fresh clone) these
tests are skipped instead of failing.
"""

import pytest

from credit_risk_ai.data.load import RAW_DIR, load_test, load_train

TARGET = "SeriousDlqin2yrs"

# Marker: skip the test when the raw CSV files are not present.
needs_raw_data = pytest.mark.skipif(
    not (RAW_DIR / "cs-training.csv").exists() or not (RAW_DIR / "cs-test.csv").exists(),
    reason="raw data not found in data/raw/",
)


# Reading a CSV takes a moment, so each file is loaded once and shared by all
# tests in this file (scope="module") instead of being re-read for every test.
@pytest.fixture(scope="module")
def raw_train():
    return load_train()


@pytest.fixture(scope="module")
def raw_test():
    return load_test()


@needs_raw_data
def test_load_train_shape_and_index(raw_train):
    assert raw_train.shape == (150_000, 11)
    assert raw_train.index.name == "id"
    assert raw_train.index.is_unique
    # The ids are the original row numbers 1..150000, not a new 0-based index.
    assert raw_train.index.min() == 1
    assert raw_train.index.max() == 150_000
    assert "Unnamed: 0" not in raw_train.columns


@needs_raw_data
def test_load_train_target_is_binary(raw_train):
    assert set(raw_train[TARGET].unique()) == {0, 1}


@needs_raw_data
def test_load_test_shape_and_index(raw_test):
    assert raw_test.shape == (101_503, 10)
    assert raw_test.index.name == "id"
    assert raw_test.index.is_unique
    # The target column is empty in the test file, so load_test() drops it.
    assert TARGET not in raw_test.columns


@needs_raw_data
def test_test_columns_match_train_features(raw_train, raw_test):
    # Same feature columns in the same order, so a model trained on the
    # training data can predict on the test data without any reordering.
    assert list(raw_test.columns) == list(raw_train.columns.drop(TARGET))


@needs_raw_data
def test_default_path_works_from_any_folder(tmp_path, monkeypatch):
    # Run the loader from an unrelated, empty folder. This fails if RAW_DIR
    # depends on the current folder instead of on the location of load.py.
    monkeypatch.chdir(tmp_path)
    assert len(load_train()) == 150_000


@pytest.mark.parametrize("loader", [load_train, load_test])
def test_missing_file_error_names_the_path(tmp_path, loader):
    # Needs no data: point the loader at a file that does not exist and check
    # that the error message says which file was missing.
    with pytest.raises(FileNotFoundError, match="missing.csv"):
        loader(tmp_path / "missing.csv")
