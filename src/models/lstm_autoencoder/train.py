"""
train.py
--------
Fit the autoencoder on training sequences and save to disk.

Key training decisions:
  - Train ONLY on X_train (no labels needed — fully unsupervised).
    The model never sees X_test during training to avoid leakage.
  - Use X_train as BOTH input AND target (autoencoder: reconstruct input).
  - EarlyStopping on val_loss with patience=5 avoids overfitting and
    speeds up training — model stops when validation loss stops improving.
  - validation_split=0.1 holds out 10% of training data for early stopping.
    This is still chronological (Keras takes the last 10% of the array).
"""



from src.utils.logger import get_logger

log = get_logger(__name__)
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras


DEFAULT_MODEL_PATH = "data/processed/lstm_autoencoder.keras"


def fit_model(model: keras.Model,
              X_train: np.ndarray,
              epochs: int,
              batch_size: int) -> keras.callbacks.History:
    """
    Fit the autoencoder. Input = Target (reconstruction objective).

    Returns Keras History object for loss curve plotting.
    """
    log.info(f"  Training on {X_train.shape}  epochs={epochs}  batch={batch_size}")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    history = model.fit(
        X_train, X_train,        # input == target
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,    # last 10% of train for early stopping
        callbacks=callbacks,
        verbose=2,
    )
    return history


def save_model(model: keras.Model,
               path: str = DEFAULT_MODEL_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    log.info(f"  Saved model → {path}")


def load_model(path: str = DEFAULT_MODEL_PATH) -> keras.Model:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No saved model at {path}. Run train.py or detector.run() first."
        )
    return keras.models.load_model(path)