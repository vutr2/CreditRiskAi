"""Project paths shared by training (train.py) and serving (api.py).

Everything written to these paths is generated (model file, Kaggle submission,
MLflow data) and is ignored by git, see .gitignore.
"""

from pathlib import Path

# src/credit_risk_ai/paths.py -> parents[2] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "hgb.joblib"
SUBMISSION_PATH = PROJECT_ROOT / "submissions" / "submission.csv"

# MLflow keeps run data in a SQLite file and saved files (model, submission) in a folder
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_ARTIFACT_DIR = PROJECT_ROOT / "mlruns"
