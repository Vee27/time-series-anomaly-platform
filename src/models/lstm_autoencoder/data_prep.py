"""
data_prep.py
------------
Load lstm_sequences.npz and perform a chronological split.

Sequences were built by feature_engineering.py using vitals-only columns
(6 features) with a 60-step sliding window. The scaler was already applied
upstream — do NOT rescale here.
"""

import numpy as np
from pathlib import Path


def load_sequences(cfg: dict):
    """
    Load pre-built LSTM sequences from disk.

    Returns:
        sequences : np.ndarray shape (N, 60, 6) float32
        seq_path  : Path - for logging
    """
    seq_path = Path("data/processed/lstm_sequences.npz")
    if not seq_path.exists():
        raise FileNotFoundError(
            f"Sequences not found at {seq_path}. "
            "Run feature_engineering.py first."
        )
    data = np.load(seq_path)
    sequences = data["sequences"]          # (N, 60, 6)
    print(f"Loaded sequences: {sequences.shape}  dtype={sequences.dtype}")
    return sequences, seq_path


def chronological_split(sequences: np.ndarray, train_split: float):
    """
    Strict chronological split — no shuffling, ever.

    Each sequence window corresponds to a specific 60-minute block in the
    time series. Shuffling would leak future patterns into training.

    Returns:
        X_train    : np.ndarray (N_train, 60, 6)
        X_test     : np.ndarray (N_test,  60, 6)
        split_idx  : int — first test index
    """
    N = len(sequences)
    split_idx = int(N * train_split)
    X_train   = sequences[:split_idx]
    X_test    = sequences[split_idx:]
    print(f"  Train: {X_train.shape}   Test: {X_test.shape}")
    return X_train, X_test, split_idx