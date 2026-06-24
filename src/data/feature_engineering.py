"""
feature_engineering.py
-----------------------
Feature transformation for Prophet and LSTM autoencoder pipelines.

Operates on the single-stream cleaned CSV produced by preprocess.py.
All rolling/lag features are computed globally (no groupby) because the
entire dataset is one continuous stream (patient_id = 1 throughout).
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Rolling features ──────────────────────────────────────────────────────────

def add_rolling_features(df, vitals, windows):
    """
    Add rolling mean and std for each vital over each window size.

    No groupby — entire df is one continuous stream so we roll globally.
    Uses direct Series.rolling() which hits the pandas C extension fast path.

    Columns added: {vital}_roll_mean_{w}, {vital}_roll_std_{w}
    for every vital in vitals and every w in windows.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    for col in vitals:
        if col not in df.columns:
            print(f"  SKIP rolling: '{col}' not in df")
            continue
        for w in windows:
            roll = df[col].rolling(window=w, min_periods=1)
            df[f"{col}_roll_mean_{w}"] = roll.mean()
            df[f"{col}_roll_std_{w}"]  = roll.std().fillna(0)

    return df


# ── Lag features ──────────────────────────────────────────────────────────────

def add_lag_features(df, vitals, lags):
    """
    Add lagged values for each vital at each lag step.

    No groupby — single stream, global shift is correct.
    All lags for one vital are batched into a single concat to avoid
    repeated DataFrame mutations.

    Columns added: {vital}_lag_{lag} for every vital and lag.
    NaNs at the start of the series (first max(lags) rows) are filled
    with 0 later in run() — not here, so the caller can audit them first.
    """
    for col in vitals:
        if col not in df.columns:
            print(f"  SKIP lag: '{col}' not in df")
            continue
        lag_dict = {f"{col}_lag_{lag}": df[col].shift(lag) for lag in lags}
        df = pd.concat([df, pd.DataFrame(lag_dict, index=df.index)], axis=1)

    return df


# ── Time features ─────────────────────────────────────────────────────────────

def add_time_features(df, timestamp_col="timestamp"):
    """
    Cyclical (sin/cos) encoding of time-of-day.

    Gives LSTM an explicit circadian signal and helps Prophet identify
    daily seasonality without relying purely on its internal Fourier terms.

    sin_time and cos_time together encode the full 24h cycle without
    the discontinuity that a raw 'hour' integer would have at midnight.
    """
    minute_of_day    = df[timestamp_col].dt.hour * 60 + df[timestamp_col].dt.minute
    df["sin_time"]   = np.sin(2 * np.pi * minute_of_day / 1440)
    df["cos_time"]   = np.cos(2 * np.pi * minute_of_day / 1440)
    return df


# ── Scaler ────────────────────────────────────────────────────────────────────

def scale_features(df, feature_cols, scaler_type="standard", save_path=None):
    """
    Fit and apply StandardScaler or MinMaxScaler to feature_cols in-place.

    Saves the fitted scaler to disk — CRITICAL for Day 3 LSTM inference:
    reconstruction-error thresholds are calibrated on scaled data, so the
    same scaler must be applied to any new data at inference time.

    Raises ValueError with a clear message if df is empty or feature_cols
    is empty — surfaces the problem rather than letting sklearn crash.
    """
    if df.empty or len(feature_cols) == 0:
        raise ValueError(
            f"scale_features received empty df (shape={df.shape}) or "
            f"no feature columns (len={len(feature_cols)}). "
            "Check that dropna() did not wipe all rows."
        )

    remaining_nans = df[feature_cols].isna().sum().sum()
    if remaining_nans > 0:
        raise ValueError(
            f"Found {remaining_nans} NaNs in feature_cols before scaling. "
            "Ensure fillna() ran before scale_features()."
        )

    scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, save_path)
        print(f"Saved scaler → {save_path}")

    return df, scaler


# ── LSTM sequence builder ─────────────────────────────────────────────────────

def build_lstm_sequences(df, feature_cols, sequence_length):
    """
    Build sliding-window sequences for the LSTM autoencoder.

    Uses numpy stride_tricks for a zero-copy sliding window view — avoids
    millions of list.append() calls and the expensive np.array() stack at
    the end that the naive loop approach requires.

    Returns:
        sequences : np.ndarray, shape (N, sequence_length, n_features)
                    dtype float32 — matches Keras default, halves memory vs float64
        indices   : list of df index values for the LAST timestep of each window
                    (use to align predictions back to timestamps)
    """
    values  = df[feature_cols].to_numpy(dtype=np.float32)
    n       = len(values)
    n_feat  = len(feature_cols)

    if n < sequence_length:
        raise ValueError(
            f"Dataset has {n} rows but sequence_length={sequence_length}. "
            "Reduce lstm.sequence_length in config.yaml or use more data."
        )

    n_windows = n - sequence_length + 1
    shape     = (n_windows, sequence_length, n_feat)
    strides   = (values.strides[0], values.strides[0], values.strides[1])

    # stride_tricks gives a VIEW — .copy() so we own the memory safely
    sequences = np.lib.stride_tricks.as_strided(
        values, shape=shape, strides=strides
    ).copy()

    # Index of the last timestep in each window (for traceability)
    indices = df.index[sequence_length - 1:].tolist()

    return sequences, indices


# ── Sanity check ──────────────────────────────────────────────────────────────

def sanity_check(df, vitals, feature_cols):
    """
    Print a quick report to confirm rolling/lag features are non-degenerate.
    This is the check that would have caught the groupby-per-1-row bug early.

    Red flags:
      - roll_std variance == 0  →  all windows collapsed to single values
      - lag non-zero rate < 90% →  lags mostly empty (group size too small)
    """
    print("\n── Feature sanity check ─────────────────────────────────")

    roll_std_cols = [c for c in feature_cols if "roll_std" in c]
    if roll_std_cols:
        variances = df[roll_std_cols].var().round(4)
        zero_var  = variances[variances == 0]
        if len(zero_var):
            print(f"  WARNING: {len(zero_var)} roll_std columns have zero variance:")
            print(f"  {list(zero_var.index)}")
            print("  → Rolling windows may be larger than group size. "
                  "Check patient_id assignment in preprocess.py.")
        else:
            print(f"  OK — all {len(roll_std_cols)} roll_std columns have non-zero variance")

    lag_cols = [c for c in feature_cols if "_lag_" in c]
    if lag_cols:
        nonzero_rate = (df[lag_cols] != 0).mean().mean()
        if nonzero_rate < 0.90:
            print(f"  WARNING: lag features non-zero rate = {nonzero_rate:.1%} "
                  "(expected >90%). Lag fill may be too aggressive.")
        else:
            print(f"  OK — lag features non-zero rate = {nonzero_rate:.1%}")

    print(f"  Total feature columns : {len(feature_cols)}")
    print(f"  DataFrame shape       : {df.shape}")
    print(f"  NaN count             : {df[feature_cols].isna().sum().sum()}")
    print("─────────────────────────────────────────────────────────\n")


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run(config_path="config.yaml"):
    cfg      = load_config(config_path)
    vitals   = cfg["vitals"]
    fe_cfg   = cfg["feature_engineering"]
    lstm_cfg = cfg["lstm"]

    print(f"\n{'='*60}")
    print("Loading processed data...")
    print(f"{'='*60}")
    df = pd.read_csv(cfg["data"]["processed_path"], parse_dates=["timestamp"])
    print(f"Loaded: {df.shape}")
    print(f"NaNs in vitals:\n{df[vitals].isna().sum().to_string()}")

    # Single copy at pipeline entry — all mutations in-place after this
    df = df.copy()

    print(f"\n{'='*60}")
    print("STEP 1 — Rolling features")
    print(f"{'='*60}")
    df = add_rolling_features(df, vitals, fe_cfg["rolling_windows"])
    print(f"  Shape after rolling: {df.shape}")

    print(f"\n{'='*60}")
    print("STEP 2 — Lag features")
    print(f"{'='*60}")
    df = add_lag_features(df, vitals, fe_cfg["lag_features"])
    print(f"  Shape after lag: {df.shape}")

    print(f"\n{'='*60}")
    print("STEP 3 — Cyclical time features")
    print(f"{'='*60}")
    df = add_time_features(df)
    print(f"  Shape after time features: {df.shape}")

    # ── Identify feature columns (after all additions, before any dropping) ──
    exclude_cols  = {"timestamp", "patient_id", "was_missing",
                     "is_anomaly", "risk_category", "Gender", "gender"}
    feature_cols  = [
        c for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    print(f"\nFeature columns identified: {len(feature_cols)}")

    # ── Fill warm-up NaNs from lag shifts; drop only if vital itself is NaN ──
    print(f"\n{'='*60}")
    print("STEP 4 — Handle NaNs")
    print(f"{'='*60}")
    nan_before = df[feature_cols].isna().sum().sum()
    df[feature_cols] = df[feature_cols].fillna(0)   # lag warm-up → 0
    before_drop = len(df)
    df.dropna(subset=vitals, inplace=True)           # only drop if raw vital missing
    df.reset_index(drop=True, inplace=True)
    print(f"  Filled {nan_before} warm-up NaNs (lag boundary zeros)")
    print(f"  Dropped {before_drop - len(df)} rows with missing vitals")
    print(f"  Shape after NaN handling: {df.shape}")

    if df.empty:
        raise RuntimeError(
            "DataFrame is empty after dropna(subset=vitals). "
            "Check that preprocess.py produced valid vital sign values "
            "in the processed CSV."
        )

    # ── Sanity check BEFORE scaling (human-readable values) ──────────────────
    sanity_check(df, vitals, feature_cols)

    # ── Scale ─────────────────────────────────────────────────────────────────
    print(f"{'='*60}")
    print(f"STEP 5 — Scale ({fe_cfg['scaler']})")
    print(f"{'='*60}")
    df, scaler = scale_features(
        df, feature_cols,
        scaler_type=fe_cfg["scaler"],
        save_path="data/processed/feature_scaler.pkl",
    )

    # ── Save feature CSV ──────────────────────────────────────────────────────
    out_path = Path("data/processed/icu_vitals_features.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved feature dataset → {out_path}  shape={df.shape}")

    # ── Build & save LSTM sequences ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 6 — Build LSTM sequences")
    print(f"{'='*60}")
    seq_array, seq_indices = build_lstm_sequences(
        df, vitals, lstm_cfg["sequence_length"]
    )
    print(f"  Sequence array shape : {seq_array.shape}")
    print(f"  dtype                : {seq_array.dtype}")
    print(f"  Memory               : {seq_array.nbytes / 1e6:.1f} MB")

    seq_path = Path("data/processed/lstm_sequences.npz")
    np.savez_compressed(seq_path, sequences=seq_array)
    print(f"  Saved → {seq_path}")

    return df, seq_array, seq_indices


if __name__ == "__main__":
    run()