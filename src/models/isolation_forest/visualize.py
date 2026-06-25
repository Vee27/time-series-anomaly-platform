"""
visualize.py
------------
All plotting functions for Isolation Forest results.

Kept separate so notebooks can import just the plots without
pulling in sklearn or the training pipeline.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

VITALS = [
    "heart_rate", "spo2", "resp_rate",
    "temperature", "systolic_bp", "diastolic_bp",
]


def plot_score_timeline(result: pd.DataFrame, ax: plt.Axes) -> None:
    """
    Panel 1 — anomaly score scatter over time.
    Normal points in blue (small), anomalies in red (larger).
    Vertical dashed line marks the train/test boundary.
    """
    normal = result[result["anomaly"] == 0]
    anom   = result[result["anomaly"] == 1]

    ax.scatter(normal["timestamp"], normal["anomaly_score"],
               s=0.5, alpha=0.3, color="steelblue", label="normal")
    ax.scatter(anom["timestamp"], anom["anomaly_score"],
               s=2, alpha=0.7, color="red",
               label=f"anomaly (n={len(anom):,})")

    boundary = result[result["split"] == "test"]["timestamp"].min()
    if pd.notna(boundary):
        ax.axvline(boundary, color="orange", linestyle="--",
                   linewidth=1.2, label="train/test split")

    ax.set_ylabel("anomaly score (normalised [0,1])")
    ax.set_title("Isolation Forest — anomaly score over time")
    ax.legend(loc="upper right", fontsize=8, markerscale=5)


def plot_vital_anomaly_rates(result: pd.DataFrame, ax: plt.Axes) -> None:
    """
    Panel 2 — per-vital anomaly rate bar chart.
    For each vital, shows what fraction of anomalous rows have an
    above-median score for that vital. Gives a rough sense of which
    vitals are driving the anomaly flags.
    """
    present = [v for v in VITALS if v in result.columns]
    if not present:
        ax.set_visible(False)
        return

    anom    = result[result["anomaly"] == 1]
    median_score = result["anomaly_score"].median()

    rates = []
    for v in present:
        if anom.empty:
            rates.append(0.0)
        else:
            # Fraction of anomalous rows where the vital is in the upper half
            # of its own distribution (proxy for "this vital was extreme")
            threshold = result[v].median()
            rates.append((anom[v] > threshold).mean())

    bars = ax.bar(present, rates, color="salmon",
                  edgecolor="darkred", linewidth=0.5)
    ax.set_ylabel("fraction of anomalies above vital median")
    ax.set_title("Which vitals are elevated in anomalous rows")
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylim(0, 1)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{rate:.0%}", ha="center", va="bottom", fontsize=9)


def plot_score_distribution(result: pd.DataFrame, ax: plt.Axes) -> None:
    """
    Panel 3 — histogram of anomaly scores split by normal/anomaly.
    Well-separated distributions = IF is discriminating well.
    """
    normal = result[result["anomaly"] == 0]["anomaly_score"]
    anom   = result[result["anomaly"] == 1]["anomaly_score"]

    ax.hist(normal, bins=60, alpha=0.6, color="steelblue",
            density=True, label="normal")
    ax.hist(anom, bins=60, alpha=0.6, color="red",
            density=True, label="anomaly")
    ax.set_xlabel("anomaly score")
    ax.set_ylabel("density")
    ax.set_title("Score distribution — normal vs anomaly")
    ax.legend(fontsize=9)


def plot_results(result: pd.DataFrame,
                 save_dir: str = "results/figures") -> None:
    """
    Three-panel figure combining all plots above.
    Saved to results/figures/isolation_forest_results.png.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    plot_score_timeline(result, axes[0])
    plot_vital_anomaly_rates(result, axes[1])
    plot_score_distribution(result, axes[2])

    plt.tight_layout()
    save_path = Path(save_dir) / "isolation_forest_results.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot → {save_path}")