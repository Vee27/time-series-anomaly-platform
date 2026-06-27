"""
predict.py
----------
Compute reconstruction error (MSE) per sequence and flag anomalies.

Threshold strategy:
  - Compute per-sample MSE on X_train after training.
  - Threshold = mean(train_mse) + multiplier × std(train_mse)
  - Default multiplier=3.0 (from config) → ~0.3% false positive rate
    under Gaussian assumption on residuals.

  CRITICAL: threshold is computed from train MSE only, never from test.
  Using test MSE to set the threshold is data leakage — the threshold
  must be fixed before ever looking at test data.
"""

import numpy as np
import pandas as pd
from tensorflow import keras


def compute_mse(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """
    Compute per-sample reconstruction MSE.

    Args:
        model : fitted autoencoder
        X     : sequences shape (N, 60, 6)

    Returns:
        mse : np.ndarray shape (N,)
              Each value = mean squared error across 60 timesteps × 6 features
              for that window. Higher = model reconstructed it worse = more anomalous.
    """
    X_hat = model.predict(X, verbose=0)         # (N, 60, 6) reconstructed
    mse   = np.mean(np.square(X - X_hat),       # element-wise squared error
                    axis=(1, 2))                # mean over timesteps + features
    return mse.astype(np.float32)


def compute_threshold(train_mse: np.ndarray,
                      multiplier: float = 3.0) -> float:
    """
    Set threshold from train MSE distribution.

    threshold = mean + multiplier × std
    Points above this on the TEST set are flagged as anomalies.
    Never recompute this from test data.
    """
    threshold = float(train_mse.mean() + multiplier * train_mse.std())
    print(f"  Train MSE — mean={train_mse.mean():.6f}  std={train_mse.std():.6f}")
    print(f"  Threshold (×{multiplier}): {threshold:.6f}")
    return threshold


def flag_anomalies(mse: np.ndarray,
                   threshold: float,
                   split_idx: int,
                   df_clean: pd.DataFrame,
                   sequence_length: int) -> pd.DataFrame:
    """
    Build the results DataFrame aligning MSE scores back to timestamps.

    Each MSE value corresponds to the LAST timestep of its 60-step window
    (this matches the index convention in feature_engineering.build_lstm_sequences).

    Args:
        mse              : per-sample MSE, shape (N_windows,)
        threshold        : float from compute_threshold
        split_idx        : window index of first test sequence
        df_clean         : cleaned vitals CSV (for timestamps + raw values)
        sequence_length  : 60 (from config)

    Returns:
        DataFrame with columns:
            timestamp, heart_rate, ..., mse, threshold,
            anomaly (1/0), split (train/test)
    """
    # Window i corresponds to df_clean rows [i : i+sequence_length]
    # The "label" timestamp = last row of the window = row (i + sequence_length - 1)
    n_windows = len(mse)
    row_indices = np.arange(sequence_length - 1,
                            sequence_length - 1 + n_windows)

    result = df_clean.iloc[row_indices].copy().reset_index(drop=True)
    result["mse"]       = mse
    result["threshold"] = threshold
    result["anomaly"]   = (mse > threshold).astype(int)
    result["split"]     = "train"
    result.loc[result.index >= split_idx, "split"] = "test"

    n_anom = result["anomaly"].sum()
    print(f"\n  Anomalies flagged: {n_anom} / {len(result)}  "
          f"({n_anom / len(result):.1%})")
    return result


def print_metrics(result: pd.DataFrame) -> None:
    print("\n── LSTM Autoencoder metrics ─────────────────────────────")
    for split in ["train", "test"]:
        sub = result[result["split"] == split]
        if sub.empty:
            continue
        rate     = sub["anomaly"].mean()
        mean_mse = sub["mse"].mean()
        max_mse  = sub["mse"].max()
        print(f"  {split:5s}  anomaly_rate={rate:.2%}  "
              f"mean_mse={mean_mse:.6f}  max_mse={max_mse:.6f}  n={len(sub)}")
    print("─────────────────────────────────────────────────────────\n")