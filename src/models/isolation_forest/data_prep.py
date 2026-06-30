"""
data_prep.py
------------
Load the scaled feature matrix produced by feature_engineering.py
and perform a chronological train/test split.

No scaling is done here — feature_scaler.pkl was already applied
upstream. Do NOT refit or re-apply the scaler in this module.
"""



from src.utils.logger import get_logger

log = get_logger(__name__)
import pandas as pd
from pathlib import Path


def load_feature_matrix(cfg: dict):
    """
    Load icu_vitals_features.csv and return (df, feature_cols).

    Excludes metadata columns from the feature matrix using the same
    exclusion list as feature_engineering.py so the column set is
    always consistent.

    Returns:
        df           : full DataFrame with timestamp intact for alignment
        feature_cols : list of column names fed to IsolationForest
    """
    path = Path("data/processed/icu_vitals_features.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Feature CSV not found at {path}. "
            "Run feature_engineering.py first."
        )

    df = pd.read_csv(path, parse_dates=["timestamp"])
    log.info(f"Loaded feature matrix: {df.shape}")

    exclude = {
        "timestamp", "patient_id", "was_missing",
        "is_anomaly", "risk_category", "Gender", "gender",
    }
    feature_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    log.info(f"Feature columns for IF: {len(feature_cols)}")
    return df, feature_cols


def chronological_split(df: pd.DataFrame, train_split: float):
    """
    Strict chronological split — never shuffle time series data.

    Returns:
        train_df  : first train_split fraction of rows
        test_df   : remaining rows
        split_idx : integer index of the first test row
    """
    split_idx = int(len(df) * train_split)
    train_df  = df.iloc[:split_idx].copy()
    test_df   = df.iloc[split_idx:].copy()
    log.info(f"  Train: {len(train_df)} rows "
          f"({train_df['timestamp'].min()} → {train_df['timestamp'].max()})")
    log.info(f"  Test : {len(test_df)} rows "
          f"({test_df['timestamp'].min()} → {test_df['timestamp'].max()})")
    return train_df, test_df, split_idx