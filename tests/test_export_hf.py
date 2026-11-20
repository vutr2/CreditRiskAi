"""Tests for credit_risk_ai.export_hf (a small model on synthetic data; no real data needed)."""

import numpy as np
import pandas as pd
from skops.io import load
from sklearn.ensemble import HistGradientBoostingClassifier

from credit_risk_ai.export_hf import export
from credit_risk_ai.models.pipeline import SKOPS_TRUSTED_TYPES


def fit_small_model():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["age", "DebtRatio", "weighted_past_due"])
    y = (np.arange(200) % 4 == 0).astype(int)
    return HistGradientBoostingClassifier(random_state=0).fit(X, y), X


def test_export_writes_the_model_and_the_card(tmp_path):
    model, _ = fit_small_model()
    out_dir = export(model, out_dir=tmp_path / "hf", repo_id="someone/credit-risk-hgb")

    assert sorted(p.name for p in out_dir.iterdir()) == ["README.md", "model.skops"]


def test_exported_model_loads_with_the_trusted_types_and_predicts_the_same(tmp_path):
    model, X = fit_small_model()
    out_dir = export(model, out_dir=tmp_path / "hf")

    loaded = load(out_dir / "model.skops", trusted=SKOPS_TRUSTED_TYPES)
    assert np.array_equal(loaded.predict_proba(X), model.predict_proba(X))


def test_model_card_has_metadata_repo_id_features_and_results(tmp_path):
    model, _ = fit_small_model()
    card = (export(model, out_dir=tmp_path / "hf", repo_id="someone/credit-risk-hgb")
            / "README.md").read_text(encoding="utf-8")

    assert card.startswith("---\nlibrary_name: sklearn")   # Hugging Face metadata block
    assert 'repo_id="someone/credit-risk-hgb"' in card
    assert "1. `age`" in card and "3. `weighted_past_due`" in card
    assert "0.86780" in card                                 # Kaggle private score
    assert "Do not use it for real lending decisions" in card
    assert "$" not in card                                   # every placeholder was filled in
