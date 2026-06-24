"""
preprocess.py
-------------
Clean the Kaggle Human Vital Signs dataset and produce a single continuous
minute-level time series ready for Prophet, Isolation Forest, and LSTM.

Key design decisions (document in README):
  - The Kaggle dataset is cross-sectional: one snapshot row per patient.
  - Each row already carries a Timestamp 1 minute apart (descending).
  - We sort by Timestamp ascending and treat the full dataset as ONE
    continuous monitoring stream (patient_id = 1).
  - This gives ~20,000 minutes (~14 days) of plausible vitals data.
  - Limitation: adjacent rows are different patients, not longitudinal
    tracking of one person. 
"""

import argparse
import numpy as np
import pandas as pd
import yaml
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Step 1: Load & rename columns ─────────────────────────────────────────────

def load_and_rename(raw_path, column_mapping):
    """
    Load raw CSV and rename Kaggle column names to internal standard names
    defined in config.yaml under `column_mapping`.
    Prints actual columns found so mismatches are caught immediately.
    """
    df = pd.read_csv(raw_path)
    print(f"Raw shape: {df.shape}")
    print(f"Raw columns: {list(df.columns)}")

    # Only rename columns that actually exist in the file
    mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
    missing = [k for k in column_mapping if k not in df.columns]
    if missing:
        print(f"WARNING: These config column_mapping keys not found in CSV: {missing}")

    df = df.rename(columns=mapping)
    print(f"Renamed columns: {list(df.columns)}")
    return df


# ── Step 2: Parse & sort timestamps ───────────────────────────────────────────

def parse_and_sort_timestamps(df, timestamp_col="timestamp"):
    """
    Parse the existing Timestamp column (already present in Kaggle dataset,
    spaced 1 minute apart) and sort ascending so time flows forward.
    Assigns a clean sequential index after sort.
    """
    if timestamp_col not in df.columns:
        raise KeyError(
            f"Column '{timestamp_col}' not found after renaming. "
            "Check column_mapping in config.yaml matches your CSV headers exactly."
        )

    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    print(f"Timestamp range: {df[timestamp_col].min()} → {df[timestamp_col].max()}")
    print(f"Total duration: {df[timestamp_col].max() - df[timestamp_col].min()}")
    return df


# ── Step 3: Assign single patient_id ──────────────────────────────────────────

def assign_single_stream(df):
    """
    Treat entire dataset as one continuous monitoring stream.
    Overwrites any existing Patient ID with patient_id = 1.

    Why: each Kaggle Patient ID has exactly 1 row so groupby-based rolling
    windows and lag features produce degenerate results (window size > group
    size). A single stream gives ~20k consecutive minutes — enough for
    meaningful rolling stats, seasonality, and LSTM sequences.
    """
    df["patient_id"] = 1
    print(f"Assigned patient_id=1 to all {len(df)} rows (single stream mode)")
    return df


# ── Step 4: Clip physiological limits ─────────────────────────────────────────

def clip_physiological_limits(df, limits, vitals):
    """
    Set values outside physiologically plausible ranges to NaN.
    They will be interpolated in the next step.
    Prints a per-column count of clipped values.
    """
    df = df.copy()
    for col in vitals:
        if col not in df.columns:
            print(f"  SKIP clip: '{col}' not in df")
            continue
        lo, hi = limits[col]
        mask = (df[col] < lo) | (df[col] > hi)
        n_clipped = mask.sum()
        if n_clipped:
            print(f"  Clipped {n_clipped} out-of-range values in '{col}' "
                  f"(range [{lo}, {hi}])")
        df.loc[mask, col] = np.nan
    return df


# ── Step 5: Resample to 1-min & interpolate ───────────────────────────────────

def resample_and_interpolate(df, freq, vitals, max_gap_minutes=10):
    """
    Resample to target frequency (1min) using mean aggregation,
    then linearly interpolate short gaps (up to max_gap_minutes).
    Rows that remain NaN after interpolation (long gaps) are flagged
    with was_missing=1 but kept so the time index stays gapless.

    Note: with the Kaggle dataset already at 1-min intervals, this step
    mostly validates the cadence and handles any duplicate timestamps.
    """
    df = df.set_index("timestamp").sort_index()

    agg_dict = {col: "mean" for col in vitals if col in df.columns}
    agg_dict["patient_id"] = "first"

    resampled = df.resample(freq).agg(agg_dict)
    resampled["was_missing"] = resampled[vitals].isna().any(axis=1).astype(int)

    present_vitals = [c for c in vitals if c in resampled.columns]
    resampled[present_vitals] = resampled[present_vitals].interpolate(
        method="linear",
        limit=max_gap_minutes,
        limit_direction="both"
    )

    resampled = resampled.reset_index()
    print(f"After resample: {len(resampled)} rows at {freq} cadence")
    print(f"Long-gap unfilled rows (was_missing=1): {resampled['was_missing'].sum()}")
    return resampled


# ── Step 6: Drop fully empty rows ─────────────────────────────────────────────

def drop_empty_rows(df, vitals):
    """Drop rows where ALL vitals are still NaN after interpolation."""
    before = len(df)
    df = df.dropna(subset=vitals, how="all").reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} fully-empty rows")
    return df


# ── Step 7: Final report ──────────────────────────────────────────────────────

def report(df, vitals):
    print("\n── Cleaned dataset summary ──────────────────────────────")
    print(f"Shape          : {df.shape}")
    print(f"Timestamp range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"patient_id     : {sorted(df['patient_id'].unique())}")
    print(f"\nMissing values per vital:")
    for col in vitals:
        if col in df.columns:
            n = df[col].isna().sum()
            print(f"  {col:30s}: {n}")
    print(f"\nDescriptive stats:")
    print(df[[c for c in vitals if c in df.columns]].describe().round(2))
    print("─────────────────────────────────────────────────────────\n")


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run_pipeline(config_path="config.yaml"):
    cfg            = load_config(config_path)
    vitals         = cfg["vitals"]
    limits         = cfg["physiological_limits"]
    freq           = cfg["data"]["resample_freq"]
    column_mapping = cfg["column_mapping"]
    raw_path       = Path(cfg["data"]["raw_path"])
    processed_path = Path(cfg["data"]["processed_path"])

    print(f"\n{'='*60}")
    print(f"STEP 1 — Load & rename columns")
    print(f"{'='*60}")
    df = load_and_rename(raw_path, column_mapping)

    print(f"\n{'='*60}")
    print(f"STEP 2 — Parse & sort timestamps")
    print(f"{'='*60}")
    df = parse_and_sort_timestamps(df)

    print(f"\n{'='*60}")
    print(f"STEP 3 — Assign single monitoring stream")
    print(f"{'='*60}")
    df = assign_single_stream(df)

    print(f"\n{'='*60}")
    print(f"STEP 4 — Clip physiological limits")
    print(f"{'='*60}")
    df = clip_physiological_limits(df, limits, vitals)

    print(f"\n{'='*60}")
    print(f"STEP 5 — Resample to {freq} & interpolate")
    print(f"{'='*60}")
    df = resample_and_interpolate(df, freq, vitals)

    print(f"\n{'='*60}")
    print(f"STEP 6 — Drop fully empty rows")
    print(f"{'='*60}")
    df = drop_empty_rows(df, vitals)

    report(df, vitals)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Saved cleaned data → {processed_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_pipeline(args.config)