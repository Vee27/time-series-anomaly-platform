"""
prophet_model.py
----------------
Prophet-based anomaly detection for ICU vital signs.

Strategy:
  - Train Prophet on the first 80% of the time series (train_split from config).
  - Predict on the full series so anomalies in the train window are visible too.
  - A point is flagged as an anomaly when the actual value falls OUTSIDE
    Prophet's [yhat_lower, yhat_upper] uncertainty interval.
  - Returns a results DataFrame with yhat, bounds, residuals, and anomaly flag.

Usage:
    from src.models.prophet_model import ProphetAnomalyDetector
    detector = ProphetAnomalyDetector(config_path="config.yaml")
    results  = detector.run()          # train + predict + flag
    detector.save_results(results)
    detector.plot(results)
"""



from src.utils.logger import get_logger

log = get_logger(__name__)
import warnings
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import os
os.environ["PROPHET_BACKEND"] = "PYMC"
warnings.filterwarnings("ignore")          # suppress Prophet/Stan verbosity

try:
    from prophet import Prophet
except ImportError as e:
    raise ImportError(
        "prophet is not installed. Run: pip install prophet"
    ) from e


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Data prep ─────────────────────────────────────────────────────────────────

def prepare_prophet_df(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Prophet requires exactly two columns: 'ds' (datestamp) and 'y' (target).
    Drops rows where target is NaN — Prophet cannot handle them.
    """
    prophet_df = df[["timestamp", target_col]].rename(
        columns={"timestamp": "ds", target_col: "y"}
    )
    n_before = len(prophet_df)
    prophet_df = prophet_df.dropna(subset=["y"]).reset_index(drop=True)
    n_dropped = n_before - len(prophet_df)
    if n_dropped:
        log.info(f"  Dropped {n_dropped} NaN rows from '{target_col}' before Prophet fit")
    return prophet_df


def train_test_split(prophet_df: pd.DataFrame, train_split: float):
    """
    Chronological split — no shuffling, ever, for time series.
    Returns (train_df, test_df).
    """
    split_idx = int(len(prophet_df) * train_split)
    train_df  = prophet_df.iloc[:split_idx].copy()
    test_df   = prophet_df.iloc[split_idx:].copy()
    log.info(f"  Train: {len(train_df)} rows  ({train_df['ds'].min()} → {train_df['ds'].max()})")
    log.info(f"  Test : {len(test_df)} rows  ({test_df['ds'].min()} → {test_df['ds'].max()})")
    return train_df, test_df


# ── Model ─────────────────────────────────────────────────────────────────────

def build_prophet(changepoint_prior_scale: float = 0.05) -> Prophet:
    """
    Build a Prophet model tuned for ICU vital signs:

    - daily_seasonality=True   : captures circadian rhythms (24h cycle)
    - weekly_seasonality=False : irrelevant for short ICU stays
    - yearly_seasonality=False : irrelevant for <30 day monitoring
    - changepoint_prior_scale  : controls trend flexibility.
                                  0.05 (default) = conservative.
                                  Increase to 0.1-0.5 for highly variable vitals.
    - interval_width     : wide interval so only genuine outliers are flagged.
                                  In clinical settings, false positives are costly —
                                  err on the side of fewer alerts.
    """
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=False,
        yearly_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        interval_width=0.80,        # 99% prediction interval
        uncertainty_samples=200,    # Monte Carlo samples for interval (default=1000)
    )
    return model


def fit_prophet(model: Prophet, train_df: pd.DataFrame) -> Prophet:
    """Fit Prophet on training data. Returns the fitted model."""
    log.info("  Fitting Prophet...")
    model.fit(train_df)
    log.info("  Done.")
    return model


# ── Prediction & anomaly flagging ─────────────────────────────────────────────

def predict_and_flag(
    model: Prophet,
    prophet_df: pd.DataFrame,
    train_size: int,
) -> pd.DataFrame:
    """
    Run Prophet forecast on the full ds range, then merge actuals back in.

    Anomaly logic:
        anomaly = 1  if  y < yhat_lower  OR  y > yhat_upper
        anomaly = 0  otherwise

    Also computes:
        residual       = y - yhat
        anomaly_score  = |residual| / (interval_width / 2)
                         > 1 means outside the band; larger = more extreme

    Args:
        model       : fitted Prophet model
        prophet_df  : full ds+y DataFrame (train+test)
        train_size  : number of training rows (used to add a 'split' label)

    Returns:
        DataFrame with columns:
            ds, y, yhat, yhat_lower, yhat_upper,
            residual, anomaly_score, anomaly, split
    """
    forecast = model.predict(prophet_df[["ds"]])

    result = prophet_df.copy()
    result = result.merge(
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        on="ds",
        how="left",
    )

    result["residual"]     = result["y"] - result["yhat"]
    half_width             = (result["yhat_upper"] - result["yhat_lower"]) / 2
    # Avoid division by zero for degenerate intervals
    result["anomaly_score"] = np.where(
        half_width > 0,
        result["residual"].abs() / half_width,
        0.0,
    )
    result["anomaly"] = (
        (result["y"] < result["yhat_lower"]) |
        (result["y"] > result["yhat_upper"])
    ).astype(int)

    result["split"] = "train"
    result.loc[result.index >= train_size, "split"] = "test"

    n_total    = len(result)
    n_anomaly  = result["anomaly"].sum()
    log.info(f"\n  Anomalies flagged: {n_anomaly} / {n_total}  "
          f"({n_anomaly / n_total:.1%})")
    return result


# ── Evaluation helpers ────────────────────────────────────────────────────────

def print_metrics(result: pd.DataFrame) -> None:
    """
    Print MAE, RMSE, and anomaly rate split by train/test.
    No sklearn dependency — computed in numpy for portability.
    """
    log.info("\n── Prophet metrics ──────────────────────────────────────")
    for split in ["train", "test"]:
        sub = result[result["split"] == split]
        if sub.empty:
            continue
        mae  = sub["residual"].abs().mean()
        rmse = np.sqrt((sub["residual"] ** 2).mean())
        rate = sub["anomaly"].mean()
        log.info(f"  {split:5s}  MAE={mae:.4f}  RMSE={rmse:.4f}  "
              f"anomaly_rate={rate:.2%}")
    log.info("─────────────────────────────────────────────────────────\n")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(
    result: pd.DataFrame,
    target_col: str,
    save_dir: str = "results/figures",
) -> None:
    """
    Two-panel figure:
      Panel 1 — actual vs forecast with uncertainty band (train=grey, test=white bg)
      Panel 2 — anomaly score with threshold line at 1.0
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    train = result[result["split"] == "train"]
    test  = result[result["split"] == "test"]

    # ── Panel 1: forecast vs actual ──────────────────────────────────────────
    ax = axes[0]

    # Shade train region
    if not train.empty:
        ax.axvspan(train["ds"].iloc[0], train["ds"].iloc[-1],
                   alpha=0.08, color="steelblue", label="train region")

    # Uncertainty band
    ax.fill_between(
        result["ds"], result["yhat_lower"], result["yhat_upper"],
        alpha=0.25, color="orange", label="99% interval"
    )
    ax.plot(result["ds"], result["yhat"],
            color="orange", linewidth=1.2, label="yhat")
    ax.plot(result["ds"], result["y"],
            color="steelblue", linewidth=0.6, alpha=0.8, label="actual")

    # Anomaly markers
    anom = result[result["anomaly"] == 1]
    ax.scatter(anom["ds"], anom["y"],
               color="red", s=12, zorder=5, label=f"anomaly (n={len(anom)})")

    ax.set_ylabel(target_col)
    ax.set_title(f"Prophet forecast — {target_col}")
    ax.legend(loc="upper right", fontsize=8)

    # ── Panel 2: anomaly score ────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(result["ds"], result["anomaly_score"],
             linewidth=0.6, color="purple", alpha=0.8)
    ax2.axhline(1.0, color="red", linestyle="--", linewidth=1.2,
                label="threshold (score=1)")
    ax2.set_ylabel("anomaly score")
    ax2.set_xlabel("timestamp")
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    save_path = Path(save_dir) / f"prophet_{target_col}.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved plot → {save_path}")


# ── Save results ──────────────────────────────────────────────────────────────

def save_results(result: pd.DataFrame, target_col: str, out_dir: str = "results") -> Path:
    """Save the full result DataFrame to CSV for downstream use."""
    out_path = Path(out_dir) / f"prophet_{target_col}_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    log.info(f"  Saved results → {out_path}")
    return out_path


# ── High-level class ──────────────────────────────────────────────────────────

class ProphetAnomalyDetector:
    """
    Convenience wrapper for the Prophet anomaly detection pipeline.

    Usage:
        detector = ProphetAnomalyDetector("config.yaml")
        results  = detector.run()
        detector.plot(results)
        detector.save_results(results)

    Attributes:
        cfg         : parsed config dict
        target_col  : vital to model (from cfg["prophet"]["target_column"])
        model       : fitted Prophet instance (set after run())
        train_size  : number of training rows (set after run())
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.cfg        = load_config(config_path)
        self.p_cfg      = self.cfg["prophet"]
        self.target_col = self.p_cfg["target_column"]
        self.model      = None
        self.train_size = None

    def run(self) -> pd.DataFrame:
        """Execute the full train → predict → flag pipeline."""
        log.info(f"\n{'='*60}")
        log.info(f"Prophet Anomaly Detection — target: {self.target_col}")
        log.info(f"{'='*60}")

        # Load processed (unscaled) data — Prophet works best on original units
        df = pd.read_csv(
            self.cfg["data"]["processed_path"],
            parse_dates=["timestamp"],
        )
        log.info(f"Loaded: {df.shape}")

        prophet_df = prepare_prophet_df(df, self.target_col)

        train_df, _ = train_test_split(prophet_df, self.p_cfg["train_split"])
        self.train_size = len(train_df)

        self.model = build_prophet(self.p_cfg["changepoint_prior_scale"])
        fit_prophet(self.model, train_df)

        result = predict_and_flag(self.model, prophet_df, self.train_size)
        print_metrics(result)

        return result

    def plot(self, result: pd.DataFrame, save_dir: str = "results/figures") -> None:
        plot_results(result, self.target_col, save_dir)

    def save_results(self, result: pd.DataFrame, out_dir: str = "results") -> Path:
        return save_results(result, self.target_col, out_dir)

    def run_all_vitals(self) -> dict[str, pd.DataFrame]:
        """
        Run the detector on every vital in config['vitals'].
        Returns a dict mapping vital name → results DataFrame.
        Useful for the notebook and for comparing anomaly rates across vitals.
        """
        all_results = {}
        original_target = self.target_col

        for vital in self.cfg["vitals"]:
            log.info(f"\n{'─'*60}")
            self.target_col = vital
            self.p_cfg["target_column"] = vital
            try:
                result = self.run()
                self.plot(result)
                self.save_results(result)
                all_results[vital] = result
            except Exception as exc:
                log.info(f"  ERROR on {vital}: {exc}")

        self.target_col = original_target     # restore
        return all_results



if __name__ == "__main__":
    detector = ProphetAnomalyDetector("config.yaml")
    results  = detector.run()
    detector.plot(results)
    detector.save_results(results)