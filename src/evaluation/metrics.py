"""
metrics.py
----------
Compare Prophet, Isolation Forest, and LSTM anomaly detection results.

No ground-truth labels exist (unsupervised) so evaluation focuses on:
  1. Agreement between models (higher = more confident anomalies)
  2. Anomaly rate per model per split
  3. Ensemble vote (flag if ≥2 models agree)
  4. Top anomalous timestamps (all 3 models agree)
"""



from src.utils.logger import get_logger

log = get_logger(__name__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_all_results(results_dir: str = "results") -> dict:
    """
    Load result CSVs for all three models.
    Returns dict: {"prophet": df, "if": df, "lstm": df}
    """
    paths = {
        "prophet": Path(results_dir) / "prophet_heart_rate_results.csv",
        "if":      Path(results_dir) / "isolation_forest_results.csv",
        "lstm":    Path(results_dir) / "lstm_results.csv",
    }
    dfs = {}
    for name, path in paths.items():
        if not path.exists():
            log.info(f"  WARNING: {path} not found — skipping {name}")
            continue
        df = pd.read_csv(path, parse_dates=["timestamp" if name != "prophet" else "ds"])
        if name == "prophet":
            df = df.rename(columns={"ds": "timestamp", "anomaly_residual": "anomaly"})
        dfs[name] = df
        log.info(f"  Loaded {name}: {len(df)} rows, "
              f"anomaly_rate={df['anomaly'].mean():.2%}")
    return dfs


def align_on_timestamp(dfs: dict) -> pd.DataFrame:
    """
    Inner-join all model results on timestamp.
    Returns a single DataFrame with columns:
        timestamp, anomaly_prophet, anomaly_if, anomaly_lstm, split
    """
    merged = None
    rename_map = {
        "prophet": "anomaly_prophet",
        "if":      "anomaly_if",
        "lstm":    "anomaly_lstm",
    }
    for name, df in dfs.items():
        sub = df[["timestamp", "anomaly", "split"]].copy()
        sub = sub.rename(columns={"anomaly": rename_map[name]})
        if merged is None:
            merged = sub
        else:
            merged = merged.merge(sub[["timestamp", rename_map[name]]],
                                  on="timestamp", how="inner")
    return merged.sort_values("timestamp").reset_index(drop=True)


def compute_agreement(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Add ensemble columns to merged DataFrame.

    vote_count    : how many models flagged this row (0–3)
    anomaly_ensemble : 1 if ≥2 models agree, else 0
    """
    anom_cols = [c for c in merged.columns if c.startswith("anomaly_")]
    merged["vote_count"]        = merged[anom_cols].sum(axis=1)
    merged["anomaly_ensemble"]  = (merged["vote_count"] >= 2).astype(int)
    return merged


def print_comparison(merged: pd.DataFrame) -> None:
    """Print pairwise agreement and per-model anomaly rates."""
    log.info("\n── Model Comparison ─────────────────────────────────────")

    # Pairwise agreement
    pairs = [
        ("anomaly_prophet", "anomaly_if",   "Prophet↔IF"),
        ("anomaly_prophet", "anomaly_lstm",  "Prophet↔LSTM"),
        ("anomaly_if",      "anomaly_lstm",  "IF↔LSTM"),
    ]
    log.info("\n  Pairwise agreement (both flag same row):")
    for col_a, col_b, label in pairs:
        if col_a in merged.columns and col_b in merged.columns:
            agree = (merged[col_a] == merged[col_b]).mean()
            log.info(f"    {label:20s}: {agree:.1%}")

    # Per-model rates per split
    log.info("\n  Anomaly rate by model and split:")
    anom_cols = [c for c in merged.columns if c.startswith("anomaly_")
                 and c != "anomaly_ensemble"]
    header = f"  {'split':6s}" + "".join(f"  {c.replace('anomaly_',''):10s}" for c in anom_cols)
    log.info(header)
    for split in ["train", "test"]:
        sub = merged[merged["split"] == split]
        if sub.empty:
            continue
        row = f"  {split:6s}"
        for col in anom_cols:
            row += f"  {sub[col].mean():10.2%}"
        log.info(row)

    # Ensemble
    log.info(f"\n  Ensemble (≥2 models agree):")
    for split in ["train", "test", "all"]:
        sub = merged if split == "all" else merged[merged["split"] == split]
        n = sub["anomaly_ensemble"].sum()
        r = sub["anomaly_ensemble"].mean()
        log.info(f"    {split:6s}: {n:6,} anomalies  ({r:.2%})")

    log.info("─────────────────────────────────────────────────────────\n")


def top_anomalies(merged: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """
    Return the top N timestamps where all 3 models agree on anomaly.
    These are the most confident/severe anomalies.
    """
    unanimous = merged[merged["vote_count"] == 3].copy()
    log.info(f"  Unanimous anomalies (all 3 models): {len(unanimous):,}")
    return unanimous.head(n)


def plot_comparison(merged: pd.DataFrame,
                    save_dir: str = "results/figures") -> None:
    """
    Two-panel comparison plot:
      Panel 1 — vote count over time (0–3 models flagging each point)
      Panel 2 — ensemble anomaly flags
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # Panel 1: vote count heatmap-style scatter
    ax = axes[0]
    colors = {0: "steelblue", 1: "gold", 2: "orange", 3: "red"}
    for vote in [0, 1, 2, 3]:
        sub = merged[merged["vote_count"] == vote]
        label = f"{vote} model{'s' if vote != 1 else ''} flagged (n={len(sub):,})"
        ax.scatter(sub["timestamp"], sub["vote_count"],
                   s=1, alpha=0.4, color=colors[vote], label=label)
    ax.set_ylabel("models in agreement (0–3)")
    ax.set_title("Model Agreement Over Time")
    ax.legend(loc="upper right", fontsize=8, markerscale=6)
    ax.set_yticks([0, 1, 2, 3])

    # Panel 2: ensemble flag
    ax2 = axes[1]
    normal = merged[merged["anomaly_ensemble"] == 0]
    anom   = merged[merged["anomaly_ensemble"] == 1]
    ax2.scatter(normal["timestamp"], [0] * len(normal),
                s=0.3, alpha=0.2, color="steelblue")
    ax2.scatter(anom["timestamp"], [1] * len(anom),
                s=2, alpha=0.7, color="red",
                label=f"ensemble anomaly (n={len(anom):,})")
    ax2.set_ylabel("anomaly flag")
    ax2.set_title("Ensemble Anomaly Flag (≥2 models agree)")
    ax2.legend(loc="upper right", fontsize=8, markerscale=5)
    ax2.set_yticks([0, 1])

    boundary = merged[merged["split"] == "test"]["timestamp"].min()
    for ax in axes:
        ax.axvline(boundary, color="purple", linestyle="--",
                   linewidth=1, alpha=0.6, label="train/test split")

    plt.tight_layout()
    path = Path(save_dir) / "model_comparison.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved → {path}")


def run(results_dir: str = "results"):
    dfs    = load_all_results(results_dir)
    merged = align_on_timestamp(dfs)
    merged = compute_agreement(merged)
    print_comparison(merged)
    top    = top_anomalies(merged)
    log.info("\nTop unanimous anomalies:")
    log.info(top[["timestamp", "split", "vote_count"]].to_string(index=False))
    plot_comparison(merged)

    out = Path(results_dir) / "model_comparison.csv"
    merged.to_csv(out, index=False)
    log.info(f"\n  Saved comparison → {out}")
    return merged


if __name__ == "__main__":
    run()