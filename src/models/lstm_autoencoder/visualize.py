"""
Plots for LSTM reconstruction error and anomaly detection results.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def plot_training_loss(history, save_dir: str = "results/figures") -> None:
    """Loss curve — confirms model converged and didn't overfit."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"],     label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("LSTM Autoencoder — Training Loss")
    plt.legend()
    path = Path(save_dir) / "lstm_training_loss.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_reconstruction_error(result: pd.DataFrame,
                               save_dir: str = "results/figures") -> None:
    """
    Three-panel figure:
      Panel 1 — MSE timeline with threshold line and train/test boundary
      Panel 2 — MSE distribution (normal vs anomaly histogram)
      Panel 3 — Anomaly rate per vital (which vital drives anomalies)
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    threshold = result["threshold"].iloc[0]

    # Panel 1: MSE timeline
    ax = axes[0]
    normal = result[result["anomaly"] == 0]
    anom   = result[result["anomaly"] == 1]
    ax.scatter(normal["timestamp"], normal["mse"],
               s=0.5, alpha=0.3, color="steelblue", label="normal")
    ax.scatter(anom["timestamp"], anom["mse"],
               s=2, alpha=0.7, color="red",
               label=f"anomaly (n={len(anom):,})")
    ax.axhline(threshold, color="orange", linestyle="--",
               linewidth=1.2, label=f"threshold={threshold:.4f}")
    boundary = result[result["split"] == "test"]["timestamp"].min()
    ax.axvline(boundary, color="purple", linestyle="--",
               linewidth=1, label="train/test split")
    ax.set_ylabel("reconstruction MSE")
    ax.set_title("LSTM Autoencoder — Reconstruction Error Over Time")
    ax.legend(loc="upper right", fontsize=8, markerscale=5)

    # Panel 2: distribution
    ax2 = axes[1]
    ax2.hist(normal["mse"],  bins=80, alpha=0.6,
             color="steelblue", density=True, label="normal")
    ax2.hist(anom["mse"],    bins=80, alpha=0.6,
             color="red",       density=True, label="anomaly")
    ax2.axvline(threshold, color="orange", linestyle="--",
                linewidth=1.2, label="threshold")
    ax2.set_xlabel("MSE")
    ax2.set_ylabel("density")
    ax2.set_title("MSE Distribution — Normal vs Anomaly")
    ax2.legend(fontsize=9)

    # Panel 3: per-vital mean MSE for anomalous windows
    ax3 = axes[2]
    vitals = ["heart_rate", "spo2", "resp_rate",
              "temperature", "systolic_bp", "diastolic_bp"]
    present = [v for v in vitals if v in result.columns]
    if present:
        anom_sub = result[result["anomaly"] == 1]
        # Use deviation from overall mean as proxy for which vital is extreme
        means_all  = result[present].mean()
        means_anom = anom_sub[present].mean() if not anom_sub.empty else means_all
        deviation  = (means_anom - means_all).abs()
        ax3.bar(present, deviation.values,
                color="salmon", edgecolor="darkred", linewidth=0.5)
        ax3.set_ylabel("|mean_anomaly - mean_all| (scaled units)")
        ax3.set_title("Which vitals deviate most in anomalous windows")
        ax3.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    path = Path(save_dir) / "lstm_reconstruction_error.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")