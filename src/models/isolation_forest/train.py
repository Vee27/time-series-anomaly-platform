"""
train.py
--------
Build, fit, save, and load the IsolationForest model.

Keeping training separate from prediction means you can:
  - retrain without re-running the full pipeline
  - load a saved model at inference time without touching this file
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest

DEFAULT_MODEL_PATH = "data/processed/isolation_forest.pkl"


def build_model(cfg: dict) -> IsolationForest:
    """
    Instantiate IsolationForest from config params.

    Key params:
      contamination : expected fraction of anomalies in training data.
                      Sets sklearn's internal decision threshold automatically.
                      Start at 0.05; raise if too few anomalies are flagged.
      n_estimators  : number of isolation trees. 200 is robust; use 50-100
                      during development for faster iteration.
      n_jobs=-1     : uses all CPU cores — no config param needed.
    """
    if_cfg = cfg["isolation_forest"]
    return IsolationForest(
        n_estimators  = if_cfg["n_estimators"],
        contamination = if_cfg["contamination"],
        max_features  = if_cfg["max_features"],
        random_state  = if_cfg["random_state"],
        n_jobs        = -1,
    )


def fit_model(model: IsolationForest,
              train_df: pd.DataFrame,
              feature_cols: list) -> IsolationForest:
    """Fit on the training split. Returns the fitted model."""
    X_train = train_df[feature_cols].to_numpy()
    print(f"  Fitting IsolationForest on {X_train.shape}...")
    model.fit(X_train)
    print("  Done.")
    return model


def save_model(model: IsolationForest,
               path: str = DEFAULT_MODEL_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"  Saved model → {path}")


def load_model(path: str = DEFAULT_MODEL_PATH) -> IsolationForest:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No saved model at {path}. Run train.py or detector.run() first."
        )
    return joblib.load(path)