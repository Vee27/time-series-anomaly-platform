"""
model.py
--------
Build the LSTM autoencoder architecture.

Architecture: Encoder-Bottleneck-Decoder
  Input  → LSTM(64) → Dense(latent_dim=32) → RepeatVector(60)
         → LSTM(32, return_seq) → LSTM(64, return_seq) → Dense(6)

Why this architecture:
  - Encoder LSTM(64) compresses the 60-step sequence into a single vector.
    return_sequences=False forces it to summarise the whole window.
  - Dense(32) is the bottleneck — the model must encode 60×6=360 values
    into 32 numbers. Reconstruction forces it to learn normal patterns.
  - RepeatVector(60) expands the bottleneck back to sequence length so
    the decoder LSTM layers can operate step-by-step.
  - Dense(6) output has the same shape as input — MSE(input, output)
    measures how well the model reconstructed normal vital patterns.
  - Anomaly = high reconstruction error (the model was trained only on
    normal data so it reconstructs anomalies poorly).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_autoencoder(sequence_length: int,
                      n_features: int,
                      latent_dim: int,
                      learning_rate: float = 0.001) -> keras.Model:
    """
    Build and compile the LSTM autoencoder.

    Args:
        sequence_length : timesteps per window (60 from config)
        n_features      : number of vitals (6)
        latent_dim      : bottleneck size (32 from config)
        learning_rate   : Adam lr (0.001 from config)

    Returns:
        Compiled Keras model, input and output shape (None, 60, 6)
    """
    inputs = keras.Input(shape=(sequence_length, n_features))

    # ── Encoder ──────────────────────────────────────────────────────────────
    x = layers.LSTM(64, return_sequences=False)(inputs)
    # Bottleneck: compress to latent_dim
    encoded = layers.Dense(latent_dim, activation="relu")(x)

    # ── Bridge: expand bottleneck back to sequence length ─────────────────────
    x = layers.RepeatVector(sequence_length)(encoded)

    # ── Decoder ──────────────────────────────────────────────────────────────
    x = layers.LSTM(32, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    # Reconstruct original n_features at each timestep
    outputs = layers.TimeDistributed(layers.Dense(n_features))(x)

    model = keras.Model(inputs, outputs, name="lstm_autoencoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",    # MSE loss = reconstruction error
    )
    return model