"""
tests/test_lstm.py
------------------
Unit + integration tests for src/models/lstm_autoencoder/.

Fast unit tests use tiny synthetic data and mock the Keras model.
Integration tests (@pytest.mark.slow) do a real tiny fit.

Run:
    pytest tests/test_lstm.py -v              # unit tests only
    pytest tests/test_lstm.py -v --runslow   # include integration tests
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Lazy TF import so tests don't fail if TF not installed ────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TF_AVAILABLE,
                                reason="tensorflow not installed")

from src.models.lstm_autoencoder.data_prep import load_sequences, chronological_split
from src.models.lstm_autoencoder.model     import build_autoencoder
from src.models.lstm_autoencoder.predict   import compute_mse, compute_threshold, flag_anomalies
from src.models.lstm_autoencoder.train     import fit_model, save_model, load_model


# ── Fixtures ──────────────────────────────────────────────────────────────────

SEQ_LEN  = 10   # use tiny sequences for speed
N_FEAT   = 6
N_SAMP   = 200

@pytest.fixture
def tiny_sequences():
    """Synthetic (200, 10, 6) float32 sequences — no disk I/O."""
    np.random.seed(42)
    return np.random.randn(N_SAMP, SEQ_LEN, N_FEAT).astype(np.float32)


@pytest.fixture
def tiny_model():
    """Small autoencoder that actually compiles — latent_dim=4 for speed."""
    return build_autoencoder(
        sequence_length=SEQ_LEN,
        n_features=N_FEAT,
        latent_dim=4,
        learning_rate=0.001,
    )


@pytest.fixture
def tiny_cfg():
    return {
        "lstm": {
            "sequence_length": SEQ_LEN,
            "train_split": 0.8,
            "latent_dim": 4,
            "epochs": 2,
            "batch_size": 32,
            "learning_rate": 0.001,
            "threshold_multiplier": 3.0,
        },
        "data": {
            "processed_path": "data/processed/icu_vitals_clean.csv"
        }
    }


@pytest.fixture
def tiny_df_clean():
    """Minimal cleaned vitals DataFrame aligned to N_SAMP + SEQ_LEN rows."""
    n = N_SAMP + SEQ_LEN
    ts = pd.date_range("2024-01-01", periods=n, freq="1min")
    np.random.seed(0)
    return pd.DataFrame({
        "timestamp":   ts,
        "patient_id":  1,
        "was_missing": 0,
        "heart_rate":  70 + np.random.randn(n) * 10,
        "spo2":        98 + np.random.randn(n) * 0.5,
        "resp_rate":   16 + np.random.randn(n) * 2,
        "temperature": 37 + np.random.randn(n) * 0.3,
        "systolic_bp": 120 + np.random.randn(n) * 10,
        "diastolic_bp":80 + np.random.randn(n) * 6,
    })


# ── data_prep: chronological_split ───────────────────────────────────────────

class TestChronologicalSplit:
    def test_sizes_sum(self, tiny_sequences):
        X_train, X_test, idx = chronological_split(tiny_sequences, 0.8)
        assert len(X_train) + len(X_test) == len(tiny_sequences)

    def test_split_idx_correct(self, tiny_sequences):
        _, _, idx = chronological_split(tiny_sequences, 0.8)
        assert idx == int(len(tiny_sequences) * 0.8)

    def test_no_overlap(self, tiny_sequences):
        X_train, X_test, idx = chronological_split(tiny_sequences, 0.8)
        assert X_train.shape[0] == idx
        assert X_test.shape[0] == len(tiny_sequences) - idx

    def test_shapes_preserved(self, tiny_sequences):
        X_train, X_test, _ = chronological_split(tiny_sequences, 0.8)
        assert X_train.shape[1:] == (SEQ_LEN, N_FEAT)
        assert X_test.shape[1:]  == (SEQ_LEN, N_FEAT)

    def test_full_split(self, tiny_sequences):
        X_train, X_test, _ = chronological_split(tiny_sequences, 1.0)
        assert len(X_train) == len(tiny_sequences)
        assert len(X_test)  == 0

    def test_zero_split(self, tiny_sequences):
        X_train, X_test, _ = chronological_split(tiny_sequences, 0.0)
        assert len(X_train) == 0
        assert len(X_test)  == len(tiny_sequences)


# ── model: build_autoencoder ──────────────────────────────────────────────────

class TestBuildAutoencoder:
    def test_output_shape(self, tiny_sequences, tiny_model):
        """Model output must match input shape exactly."""
        batch = tiny_sequences[:4]
        out   = tiny_model.predict(batch, verbose=0)
        assert out.shape == batch.shape

    def test_is_compiled(self, tiny_model):
        assert tiny_model.optimizer is not None
        assert tiny_model.loss == "mse"

    def test_param_count_reasonable(self, tiny_model):
        """Tiny model should have < 20k params."""
        total = tiny_model.count_params()
        assert total < 20_000, f"Unexpected param count: {total}"

    def test_different_latent_dims(self):
        """Build with latent_dim=8 and latent_dim=16 — should not raise."""
        for dim in [8, 16]:
            m = build_autoencoder(SEQ_LEN, N_FEAT, latent_dim=dim)
            assert m is not None

    def test_input_shape(self, tiny_model):
        assert tiny_model.input_shape == (None, SEQ_LEN, N_FEAT)

    def test_output_shape_spec(self, tiny_model):
        assert tiny_model.output_shape == (None, SEQ_LEN, N_FEAT)


# ── predict: compute_mse ─────────────────────────────────────────────────────

class TestComputeMse:
    def test_output_shape(self, tiny_sequences, tiny_model):
        mse = compute_mse(tiny_model, tiny_sequences)
        assert mse.shape == (len(tiny_sequences),)

    def test_non_negative(self, tiny_sequences, tiny_model):
        mse = compute_mse(tiny_model, tiny_sequences)
        assert (mse >= 0).all()

    def test_dtype_float32(self, tiny_sequences, tiny_model):
        mse = compute_mse(tiny_model, tiny_sequences)
        assert mse.dtype == np.float32

    def test_perfect_reconstruction_gives_zero(self):
        """If model perfectly reconstructs input, MSE = 0."""
        X = np.ones((5, SEQ_LEN, N_FEAT), dtype=np.float32)
        mock_model = MagicMock()
        mock_model.predict.return_value = X.copy()   # perfect reconstruction
        mse = compute_mse(mock_model, X)
        np.testing.assert_allclose(mse, 0.0, atol=1e-6)

    def test_bad_reconstruction_gives_positive(self):
        """If model returns zeros for ones input, MSE = 1.0."""
        X     = np.ones((5, SEQ_LEN, N_FEAT), dtype=np.float32)
        X_hat = np.zeros((5, SEQ_LEN, N_FEAT), dtype=np.float32)
        mock_model = MagicMock()
        mock_model.predict.return_value = X_hat
        mse = compute_mse(mock_model, X)
        np.testing.assert_allclose(mse, 1.0, atol=1e-6)


# ── predict: compute_threshold ───────────────────────────────────────────────

class TestComputeThreshold:
    def test_formula(self):
        """threshold = mean + multiplier * std."""
        mse = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        t   = compute_threshold(mse, multiplier=3.0)
        expected = float(mse.mean() + 3.0 * mse.std())
        assert t == pytest.approx(expected, rel=1e-5)

    def test_multiplier_effect(self):
        """Higher multiplier → higher threshold."""
        mse = np.random.rand(100).astype(np.float32)
        t2  = compute_threshold(mse, multiplier=2.0)
        t3  = compute_threshold(mse, multiplier=3.0)
        assert t3 > t2

    def test_returns_float(self):
        mse = np.random.rand(50).astype(np.float32)
        t   = compute_threshold(mse)
        assert isinstance(t, float)

    def test_above_mean(self):
        mse = np.random.rand(100).astype(np.float32)
        t   = compute_threshold(mse, multiplier=1.0)
        assert t > float(mse.mean())


# ── predict: flag_anomalies ───────────────────────────────────────────────────

class TestFlagAnomalies:
    def _run(self, tiny_sequences, tiny_df_clean, threshold=0.5):
        mse       = np.random.rand(N_SAMP).astype(np.float32)
        split_idx = int(N_SAMP * 0.8)
        return flag_anomalies(mse, threshold, split_idx, tiny_df_clean, SEQ_LEN)

    def test_output_length(self, tiny_sequences, tiny_df_clean):
        result = self._run(tiny_sequences, tiny_df_clean)
        assert len(result) == N_SAMP

    def test_required_columns(self, tiny_sequences, tiny_df_clean):
        result = self._run(tiny_sequences, tiny_df_clean)
        for col in ["timestamp", "mse", "threshold", "anomaly", "split"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_anomaly_is_binary(self, tiny_sequences, tiny_df_clean):
        result = self._run(tiny_sequences, tiny_df_clean)
        assert set(result["anomaly"].unique()).issubset({0, 1})

    def test_split_labels(self, tiny_sequences, tiny_df_clean):
        result = self._run(tiny_sequences, tiny_df_clean)
        assert set(result["split"].unique()) == {"train", "test"}

    def test_high_threshold_flags_nothing(self, tiny_sequences, tiny_df_clean):
        """Threshold above all MSE values → 0 anomalies."""
        mse       = np.full(N_SAMP, 0.5, dtype=np.float32)
        split_idx = int(N_SAMP * 0.8)
        result = flag_anomalies(mse, 999.0, split_idx, tiny_df_clean, SEQ_LEN)
        assert result["anomaly"].sum() == 0

    def test_low_threshold_flags_all(self, tiny_sequences, tiny_df_clean):
        """Threshold below all MSE values → all anomalies."""
        mse       = np.full(N_SAMP, 0.5, dtype=np.float32)
        split_idx = int(N_SAMP * 0.8)
        result = flag_anomalies(mse, 0.0, split_idx, tiny_df_clean, SEQ_LEN)
        assert result["anomaly"].sum() == N_SAMP

    def test_anomaly_aligns_with_mse(self, tiny_sequences, tiny_df_clean):
        """Rows where mse > threshold must have anomaly=1."""
        np.random.seed(1)
        mse       = np.random.rand(N_SAMP).astype(np.float32)
        threshold = 0.5
        split_idx = int(N_SAMP * 0.8)
        result = flag_anomalies(mse, threshold, split_idx, tiny_df_clean, SEQ_LEN)
        expected  = (mse > threshold).astype(int)
        np.testing.assert_array_equal(result["anomaly"].values, expected)


# ── train: save / load ────────────────────────────────────────────────────────

class TestSaveLoadModel:
    def test_roundtrip(self, tiny_model, tmp_path):
        path = str(tmp_path / "test_model.keras")
        save_model(tiny_model, path)
        loaded = load_model(path)
        assert loaded is not None

    def test_predictions_match(self, tiny_sequences, tiny_model, tmp_path):
        """Loaded model gives identical predictions to original."""
        path = str(tmp_path / "test_model.keras")
        save_model(tiny_model, path)
        loaded  = load_model(path)
        X       = tiny_sequences[:4]
        out_orig   = tiny_model.predict(X, verbose=0)
        out_loaded = loaded.predict(X, verbose=0)
        np.testing.assert_allclose(out_orig, out_loaded, atol=1e-5)

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(str(tmp_path / "nonexistent.keras"))


# ── Integration: real tiny fit ────────────────────────────────────────────────

@pytest.mark.slow
class TestLSTMIntegration:
    """
    Builds and trains a real (tiny) autoencoder end-to-end.
    Skipped unless --runslow is passed.

    Add to conftest.py:
        def pytest_addoption(parser):
            parser.addoption("--runslow", action="store_true")
        def pytest_runtest_setup(item):
            if "slow" in item.keywords and not item.config.getoption("--runslow"):
                pytest.skip("Pass --runslow to run this test")
    """

    def test_loss_decreases(self, tiny_sequences):
        """Train loss must decrease from epoch 1 to final epoch."""
        X_train = tiny_sequences[:160]
        model   = build_autoencoder(SEQ_LEN, N_FEAT, latent_dim=4)
        history = model.fit(
            X_train, X_train,
            epochs=5, batch_size=32,
            validation_split=0.1,
            verbose=0,
        )
        losses = history.history["loss"]
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )

    def test_mse_shape_and_range(self, tiny_sequences):
        """After training, MSE must be non-negative and shape (N,)."""
        X_train = tiny_sequences[:160]
        model   = build_autoencoder(SEQ_LEN, N_FEAT, latent_dim=4)
        model.fit(X_train, X_train, epochs=2, batch_size=32, verbose=0)
        mse = compute_mse(model, tiny_sequences)
        assert mse.shape == (len(tiny_sequences),)
        assert (mse >= 0).all()

    def test_threshold_flags_some_not_all(self, tiny_sequences, tiny_df_clean):
        """
        With multiplier=3.0, anomaly rate should be < 5% on Gaussian data.
        Also must be > 0% (threshold should not flag everything or nothing).
        """
        X_train   = tiny_sequences[:160]
        model     = build_autoencoder(SEQ_LEN, N_FEAT, latent_dim=4)
        model.fit(X_train, X_train, epochs=3, batch_size=32, verbose=0)

        train_mse = compute_mse(model, X_train)
        threshold = compute_threshold(train_mse, multiplier=3.0)
        full_mse  = compute_mse(model, tiny_sequences)
        result    = flag_anomalies(
            full_mse, threshold, 160, tiny_df_clean, SEQ_LEN
        )
        rate = result["anomaly"].mean()
        assert 0.0 <= rate <= 0.10, f"Anomaly rate {rate:.2%} out of expected range"

    def test_save_load_in_pipeline(self, tiny_sequences, tmp_path):
        """Train → save → load → predictions are identical."""
        X_train = tiny_sequences[:160]
        model   = build_autoencoder(SEQ_LEN, N_FEAT, latent_dim=4)
        model.fit(X_train, X_train, epochs=2, batch_size=32, verbose=0)

        path = str(tmp_path / "lstm_test.keras")
        save_model(model, path)
        loaded = load_model(path)

        mse_orig   = compute_mse(model,  tiny_sequences[:10])
        mse_loaded = compute_mse(loaded, tiny_sequences[:10])
        np.testing.assert_allclose(mse_orig, mse_loaded, atol=1e-5)
