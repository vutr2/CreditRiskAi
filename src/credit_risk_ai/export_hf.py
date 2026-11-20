"""Export the trained model for the Hugging Face Hub: model.skops + a model card (README.md).

Run after `python -m credit_risk_ai.train` has saved the model:

    python -m credit_risk_ai.export_hf --repo-id <your-hf-username>/credit-risk-hgb

Then upload the folder (needs a Hugging Face account and `hf auth login` once):

    hf upload <your-hf-username>/credit-risk-hgb hf_export . --repo-type model

skops is used instead of joblib/pickle: loading a pickle file can run any code,
so a public model should not be shared that way.
"""

import argparse
from string import Template

import joblib
import sklearn
from skops.io import dump

from credit_risk_ai.models.pipeline import SKOPS_TRUSTED_TYPES
from credit_risk_ai.paths import MODEL_PATH, PROJECT_ROOT

EXPORT_DIR = PROJECT_ROOT / "hf_export"
GITHUB_URL = "https://github.com/vutr2/CreditRiskAi"
DEFAULT_REPO_ID = "<your-hf-username>/credit-risk-hgb"

# Model card. Numbers come from notebooks 03-06, the MLflow run and Kaggle.
# $-placeholders are filled in by model_card().
MODEL_CARD = Template('''---
library_name: sklearn
pipeline_tag: tabular-classification
tags:
- tabular-classification
- sklearn
- skops
- credit-risk
metrics:
- roc_auc
---

# Credit default risk model (Give Me Some Credit)

A HistGradientBoosting classifier that estimates the probability that a borrower
becomes **90 or more days late on a payment within the next two years**.
Built as a learning project: data audit, cleaning, feature engineering,
cross-validation, tuning, calibration and a cost-based decision threshold.
Code, notebooks and tests: $github_url

## Results

| Evaluation | ROC-AUC |
|---|---|
| 5-fold cross-validation (training data) | 0.8651 ± 0.0044 |
| Kaggle public leaderboard (about 30% of the test set) | 0.86247 |
| Kaggle private leaderboard (about 70% of the test set) | **0.86780** |

The competition's winning private score was about 0.870.

Out-of-fold predictions on all 149,391 training rows:

| Metric | This model | Everyone gets the base rate (6.7%) |
|---|---|---|
| ROC-AUC | 0.865 | 0.5 |
| PR-AUC | 0.396 | 0.067 |
| Brier score (lower is better) | 0.0492 | 0.0625 |

The probabilities are well calibrated: in each of 10 equal-size risk groups the mean
prediction is close to the observed default rate (for example 36.9% predicted vs
36.7% observed in the riskiest group), so no extra calibration step is used.

## Intended use and limitations

- **For learning and demonstration only. Do not use it for real lending decisions.**
- The data come from a 2011 Kaggle competition about US borrowers; how it was
  collected is not documented, and today's borrowers may look different.
- The model uses **age** as an input. Many countries restrict the use of age and
  other personal attributes in credit decisions. No fairness analysis was done.
- A decision threshold depends on business costs (see below), not on the model alone.

## Input

The model expects these $n_features columns, in this order:

$feature_list

They are built from the 10 raw Kaggle columns by the project's preprocessing
(`credit_risk_ai.data.clean.clean` then `credit_risk_ai.data.features.add_features`):
invalid values become missing (NaN) with 4 flag columns, and
`weighted_past_due = 1 x (30-59 days late) + 2 x (60-89 days late) + 3 x (90+ days late)`.
Missing values (NaN) are allowed.

## How to use

```python
from huggingface_hub import hf_hub_download
from skops.io import load

path = hf_hub_download(repo_id="$repo_id", filename="model.skops")
# skops only loads types it knows are safe; this one tree class must be trusted
# explicitly (check a file with skops.io.get_untrusted_types before trusting it)
model = load(path, trusted=$trusted_types)

# Build the input with the project's preprocessing ($github_url):
from credit_risk_ai.data.clean import clean
from credit_risk_ai.data.features import add_features

X = add_features(clean(raw_applicants))  # raw_applicants: DataFrame with the 10 Kaggle input columns
probability_of_default = model.predict_proba(X[model.feature_names_in_])[:, 1]
```

## Decision threshold

If a missed defaulter costs 5 times as much as a wrongly rejected good borrower,
the lowest expected cost comes from rejecting when the probability of default is at
least 1 / (1 + 5) ≈ **0.167**. On out-of-fold predictions this rejects about 11% of
applicants and catches about 57% of defaulters. For another cost ratio *r*, use
1 / (1 + *r*) (this works because the probabilities are well calibrated).

## Training

- **Data:** Kaggle "Give Me Some Credit", `cs-training.csv`: 150,000 borrowers;
  609 exact duplicate rows removed, leaving 149,391 rows with 6.7% defaults.
  The data are not included here (Kaggle competition rules).
- **Model:** scikit-learn $sklearn_version `HistGradientBoostingClassifier` with default
  hyperparameters and `random_state=42`. Tuning 30 hyperparameter combinations
  gained only +0.001 ROC-AUC, below the project's decision rule, so the defaults were kept.
- **Final fit:** on all 149,391 rows, after cross-validation.
''')


def model_card(model, repo_id=DEFAULT_REPO_ID):
    """Model card text (Markdown with Hugging Face metadata) for this model."""
    features = list(model.feature_names_in_)
    return MODEL_CARD.substitute(
        github_url=GITHUB_URL,
        repo_id=repo_id,
        n_features=len(features),
        feature_list="\n".join(f"{i}. `{name}`" for i, name in enumerate(features, start=1)),
        trusted_types=SKOPS_TRUSTED_TYPES,
        sklearn_version=sklearn.__version__,
    )


def export(model, out_dir=EXPORT_DIR, repo_id=DEFAULT_REPO_ID):
    """Write model.skops and README.md (the model card) into out_dir; return out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dump(model, out_dir / "model.skops")
    (out_dir / "README.md").write_text(model_card(model, repo_id), encoding="utf-8")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Export the trained model for the Hugging Face Hub.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                        help="Hugging Face repo, e.g. your-username/credit-risk-hgb")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(f"Model file not found: {MODEL_PATH}. Run: python -m credit_risk_ai.train")

    out_dir = export(joblib.load(MODEL_PATH), repo_id=args.repo_id)
    print(f"Exported to {out_dir}: model.skops, README.md")
    print("Upload with:")
    print(f"    hf upload {args.repo_id} {out_dir.relative_to(PROJECT_ROOT)} . --repo-type model")


if __name__ == "__main__":
    main()
