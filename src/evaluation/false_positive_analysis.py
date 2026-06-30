"""
false_positive_analysis.py
--------------------------
For each model's anomaly flags, check whether the flagged rows
show physiologically meaningful deviations.

Without ground truth, we use:
  1. Physiological plausibility — are flagged vitals outside normal reference ranges?
  2. Multi-model agreement — unanimous flags are more likely true positives
  3. Anomaly score distribution — are high-score rows visually distinct?
"""



from src.utils.logger import get_logger

log = get_logger(__name__)
import pandas as pd
import numpy as np

# Clinical reference ranges (not the same as physiological limits in config)
CLINICAL_NORMAL = {
    "heart_rate":   (60, 100),
    "spo2":         (95, 100),
    "resp_rate":    (12, 20),
    "temperature":  (36.1, 37.2),
    "systolic_bp":  (90, 120),
    "diastolic_bp": (60, 80),
}

def clinical_deviation_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each anomalous row, count how many vitals are outside
    clinical normal ranges. Returns summary DataFrame.
    """
    anom = df[df["anomaly"] == 1].copy()
    vitals = list(CLINICAL_NORMAL.keys())
    present = [v for v in vitals if v in anom.columns]

    for v in present:
        lo, hi = CLINICAL_NORMAL[v]
        anom[f"{v}_outside"] = ((anom[v] < lo) | (anom[v] > hi)).astype(int)

    outside_cols = [f"{v}_outside" for v in present]
    anom["n_vitals_outside_range"] = anom[outside_cols].sum(axis=1)

    summary = anom["n_vitals_outside_range"].value_counts().sort_index()
    log.info("\n── Clinical deviation in anomalous rows ─────────────────")
    log.info("  # vitals outside clinical range | # rows")
    for k, v in summary.items():
        log.info(f"    {k} vitals outside range: {v:,} rows ({v/len(anom):.1%})")
    log.info("─────────────────────────────────────────────────────────")
    return anom