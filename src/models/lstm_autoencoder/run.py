"""
LSTMAnomalyDetector — wires data_prep → model → train → predict → visualize.
"""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # silence TF C++ logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # silence oneDNN
sys.stdout.reconfigure(line_buffering=True)  # force line-buffered output

# ── Path fix: allows both `python run.py` AND package import ─────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# ─────────────────────────────────────────────────────────────────────────────

import yaml
import pandas as pd
import numpy as np
from pathlib import Path

from data_prep  import load_sequences, chronological_split
from model      import build_autoencoder
from train      import fit_model, save_model, load_model
from predict    import compute_mse, compute_threshold, flag_anomalies, print_metrics
from visualize  import plot_training_loss, plot_reconstruction_error




def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_results(result: pd.DataFrame, out_dir: str = "results") -> Path:
    out_path = Path(out_dir) / "lstm_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"  Saved results → {out_path}")
    return out_path


class LSTMAnomalyDetector:
    def __init__(self, config_path: str = "config.yaml"):
        self.cfg       = load_config(config_path)
        self.lstm_cfg  = self.cfg["lstm"]
        self.model     = None
        self.history   = None
        self.threshold = None
        self.split_idx = None

    def run(self) -> pd.DataFrame:
        print(f"\n{'='*60}")
        print("LSTM Autoencoder Anomaly Detection")
        print(f"{'='*60}")

        # 1. Load sequences
        sequences, _ = load_sequences(self.cfg)
        seq_len   = self.lstm_cfg["sequence_length"]   # 60
        n_feat    = sequences.shape[2]                  # 6

        # 2. Split
        print(f"\n{'='*60}\nSplitting\n{'='*60}")
        X_train, X_test, self.split_idx = chronological_split(
            sequences, self.lstm_cfg["train_split"]
        )

        # 3. Build + train
        print(f"\n{'='*60}\nBuilding model\n{'='*60}")
        self.model = build_autoencoder(
            sequence_length = seq_len,
            n_features      = n_feat,
            latent_dim      = self.lstm_cfg["latent_dim"],
            learning_rate   = self.lstm_cfg["learning_rate"],
        )
        self.model.summary()
        sys.stdout.flush()
        print("Model built. Starting training...", flush=True)

        print(f"\n{'='*60}\nTraining\n{'='*60}")
        self.history = fit_model(
            self.model, X_train,
            epochs     = self.lstm_cfg["epochs"],
            batch_size = self.lstm_cfg["batch_size"],
        )
        save_model(self.model)

        # 4. Threshold from train MSE only
        print(f"\n{'='*60}\nComputing threshold\n{'='*60}")
        train_mse = compute_mse(self.model, X_train)
        self.threshold = compute_threshold(
            train_mse, self.lstm_cfg["threshold_multiplier"]
        )

        # 5. Score full dataset
        print(f"\n{'='*60}\nScoring full dataset\n{'='*60}")
        full_mse = compute_mse(self.model, sequences)
        df_clean = pd.read_csv(
            self.cfg["data"]["processed_path"],
            parse_dates=["timestamp"]
        )
        result = flag_anomalies(
            full_mse, self.threshold,
            self.split_idx, df_clean, seq_len
        )
        print_metrics(result)
        return result

    def plot(self, result: pd.DataFrame = None,
             save_dir: str = "results/figures") -> None:
        if self.history:
            plot_training_loss(self.history, save_dir)
        plot_reconstruction_error(
            result if result is not None else pd.DataFrame(), save_dir
        )

    def save_results(self, result: pd.DataFrame = None,
                     out_dir: str = "results") -> Path:
        return save_results(result if result is not None else pd.DataFrame(), out_dir)

    @classmethod
    def from_saved_model(cls, config_path: str = "config.yaml"):
        instance = cls(config_path)
        instance.model = load_model()
        return instance

if __name__ == "__main__":
    detector = LSTMAnomalyDetector("config.yaml")
    results  = detector.run()
    detector.plot(results)
    detector.save_results(results)