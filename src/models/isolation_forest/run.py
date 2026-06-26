"""
run.py
------
IsolationForestDetector: high-level class that wires together
data_prep → train → predict → visualize.
"""

import yaml
import pandas as pd
from pathlib import Path

from data_prep  import load_feature_matrix, chronological_split
from train      import build_model, fit_model, save_model, load_model
from predict    import score_and_flag, print_metrics
from visualize  import plot_results


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_results(result: pd.DataFrame,
                 out_dir: str = "results") -> Path:
    out_path = Path(out_dir) / "isolation_forest_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"  Saved results → {out_path}")
    return out_path


class IsolationForestDetector:
    """
    End-to-end Isolation Forest anomaly detection pipeline.

    Attributes set after .run():
        model     : fitted IsolationForest
        split_idx : index of first test row (for alignment)
        result    : last result DataFrame (cached for convenience)
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.cfg       = load_config(config_path)
        self.if_cfg    = self.cfg["isolation_forest"]
        self.model     = None
        self.split_idx = None
        self.result    = None

    def run(self) -> pd.DataFrame:
        print(f"\n{'='*60}")
        print("Isolation Forest Anomaly Detection")
        print(f"{'='*60}")

        # 1. Data
        df, feature_cols = load_feature_matrix(self.cfg)

        print(f"\n{'='*60}")
        print("Splitting data")
        print(f"{'='*60}")
        train_df, _, self.split_idx = chronological_split(
            df, self.if_cfg["train_split"]
        )

        # 2. Train
        print(f"\n{'='*60}")
        print("Training")
        print(f"{'='*60}")
        self.model = build_model(self.cfg)
        fit_model(self.model, train_df, feature_cols)
        save_model(self.model)

        # 3. Predict
        print(f"\n{'='*60}")
        print("Scoring full dataset")
        print(f"{'='*60}")
        self.result = score_and_flag(
            self.model, df, feature_cols, self.split_idx
        )
        print_metrics(self.result)

        return self.result

    def plot(self, result: pd.DataFrame = None,
             save_dir: str = "results/figures") -> None:
        plot_results(result if result is not None else self.result, save_dir)

    def save_results(self, result: pd.DataFrame = None,
                     out_dir: str = "results") -> Path:
        return save_results(result if result is not None else self.result, out_dir)

    @classmethod
    def from_saved_model(cls, config_path: str = "config.yaml",
                         model_path: str = "data/processed/isolation_forest.pkl"):
        """
        Load a previously trained model from disk — skips training.
        Use this for inference without retraining.

            detector = IsolationForestDetector.from_saved_model()
            results  = detector.run_inference(new_df, feature_cols)
        """
        instance = cls(config_path)
        instance.model = load_model(model_path)
        return instance

    def run_inference(self, df: pd.DataFrame,
                      feature_cols: list) -> pd.DataFrame:
        """
        Score new data with the loaded model (no retraining).
        Assumes df is already scaled with the same feature_scaler.pkl.
        """
        if self.model is None:
            raise RuntimeError(
                "No model loaded. Call .run() or use .from_saved_model() first."
            )
        from .predict import infer
        import numpy as np
        X = df[feature_cols].to_numpy()
        preds = infer(self.model, X)
        result = df[["timestamp"]].copy() if "timestamp" in df.columns else pd.DataFrame()
        result["anomaly"]       = preds["anomaly"]
        result["anomaly_score"] = preds["anomaly_score"]
        return result


if __name__ == "__main__":
    detector = IsolationForestDetector("config.yaml")
    results  = detector.run()
    detector.plot(results)
    detector.save_results(results)