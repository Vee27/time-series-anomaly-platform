"""
predict.py
----------
Score the full dataset with a fitted IsolationForest and flag anomalies.

Also handles inference on new data (single row or batch) using a
saved model — so this module is reused both in the training pipeline
and at production inference time.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

VITALS = [
    "heart_rate", "spo2", "resp_rate",
    "temperature", "systolic_bp", "diastolic_bp",
]


def score_and_flag(model: IsolationForest,
                   df: pd.DataFrame,
                   feature_cols: list,
                   split_idx: int) -> pd.DataFrame:
    """
    Run IsolationForest on the full dataset and build the results DataFrame.

    Columns in output:
      timestamp       : original timestamp (for alignment with other models)
      <vitals>        : raw vital sign values (human-readable context)
      anomaly_if      : sklearn raw output (-1 = anomaly, 1 = normal)
      anomaly         : 1/0 flag  (consistent with Prophet results naming)
      anomaly_score   : normalised [0,1], higher = more anomalous
                        derived from -decision_function so direction is intuitive
      split           : 'train' or 'test'

    Why negate decision_function?
      sklearn's decision_function returns lower values for anomalies.
      Negating makes higher = more suspicious, which matches the Prophet
      anomaly_score convention and is easier to threshold.
    """
    X_full = df[feature_cols].to_numpy()

    raw_pred = model.predict(X_full)                # -1 or 1
    raw_scores = -model.decision_function(X_full)   # negate: higher = more anomalous

    # Normalise to [0, 1] across the full dataset
    s_min, s_max = raw_scores.min(), raw_scores.max()
    scores_norm  = (raw_scores - s_min) / (s_max - s_min + 1e-8)

    # Build result — keep timestamp + vitals for readability
    meta_cols  = ["timestamp"] + [c for c in ["patient_id", "was_missing"] if c in df.columns]
    result     = df[meta_cols].copy()
    for v in VITALS:
        if v in df.columns:
            result[v] = df[v].values

    result["anomaly_if"]    = raw_pred
    result["anomaly"]       = (raw_pred == -1).astype(int)
    result["anomaly_score"] = scores_norm
    result["split"]         = "train"
    result.loc[result.index >= split_idx, "split"] = "test"

    n_anom = result["anomaly"].sum()
    print(f"\n  Anomalies flagged: {n_anom} / {len(result)}  "
          f"({n_anom / len(result):.1%})")
    return result


def infer(model: IsolationForest,
          X: np.ndarray) -> dict:
    """
    Inference entry point for a single row or small batch (no DataFrame needed).

    Used by streamlit_app.py and any real-time scoring pipeline.

    Args:
        model : fitted IsolationForest (loaded via train.load_model)
        X     : np.ndarray shape (n_samples, n_features), already scaled
                with feature_scaler.pkl

    Returns:
        dict with keys:
          anomaly       : np.ndarray int (1=anomaly, 0=normal) per sample
          anomaly_score : np.ndarray float [0,1] normalised per sample
    """
    raw_pred   = model.predict(X)
    raw_scores = -model.decision_function(X)
    s_min, s_max = raw_scores.min(), raw_scores.max()
    scores_norm  = (raw_scores - s_min) / (s_max - s_min + 1e-8)
    return {
        "anomaly":       (raw_pred == -1).astype(int),
        "anomaly_score": scores_norm,
    }


def print_metrics(result: pd.DataFrame) -> None:
    print("\n── Isolation Forest metrics ─────────────────────────────")
    for split in ["train", "test"]:
        sub = result[result["split"] == split]
        if sub.empty:
            continue
        rate       = sub["anomaly"].mean()
        mean_score = sub["anomaly_score"].mean()
        print(f"  {split:5s}  anomaly_rate={rate:.2%}  "
              f"mean_score={mean_score:.4f}  n={len(sub)}")
    print("─────────────────────────────────────────────────────────\n")