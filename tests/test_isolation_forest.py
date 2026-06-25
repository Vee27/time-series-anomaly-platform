"""tests/test_isolation_forest.py"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "models"))

from isolation_forest import (
    load_feature_matrix, chronological_split, build_isolation_forest,
    fit_isolation_forest, predict_and_score, save_model, load_model,
    IsolationForestDetector,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_cfg():
    return {
        "isolation_forest": {
            "train_split": 0.8,
            "n_estimators": 10,      # tiny for test speed
            "contamination": 0.05,
            "max_features": 1.0,
            "random_state": 42,
        }
    }

@pytest.fixture
def tiny_df():
    np.random.seed(42)
    n = 500
    ts = pd.date_range("2024-01-01", periods=n, freq="1min")
    df = pd.DataFrame({
        "timestamp":   ts,
        "patient_id":  1,
        "was_missing": 0,
        "heart_rate":  70 + np.random.randn(n) * 10,
        "spo2":        98 + np.random.randn(n) * 1,
        "resp_rate":   16 + np.random.randn(n) * 2,
        "temperature": 37 + np.random.randn(n) * 0.5,
        "systolic_bp": 120 + np.random.randn(n) * 15,
        "diastolic_bp":80 + np.random.randn(n) * 10,
    })
    return df

@pytest.fixture
def feature_cols(tiny_df):
    exclude = {"timestamp", "patient_id", "was_missing"}
    return [c for c in tiny_df.columns if c not in exclude]

# ── chronological_split ───────────────────────────────────────────────────────

class TestChronologicalSplit:
    def test_sizes_sum(self, tiny_df):
        train, test, idx = chronological_split(tiny_df, 0.8)
        assert len(train) + len(test) == len(tiny_df)

    def test_train_before_test(self, tiny_df):
        train, test, _ = chronological_split(tiny_df, 0.8)
        assert train["timestamp"].max() < test["timestamp"].min()

    def test_split_idx_correct(self, tiny_df):
        _, _, idx = chronological_split(tiny_df, 0.8)
        assert idx == int(len(tiny_df) * 0.8)

# ── build_isolation_forest ────────────────────────────────────────────────────

class TestBuildIsolationForest:
    def test_returns_correct_type(self, tiny_cfg):
        from sklearn.ensemble import IsolationForest
        model = build_isolation_forest(tiny_cfg)
        assert isinstance(model, IsolationForest)

    def test_params_applied(self, tiny_cfg):
        model = build_isolation_forest(tiny_cfg)
        assert model.n_estimators == 10
        assert model.contamination == pytest.approx(0.05)
        assert model.random_state == 42

# ── fit + predict ─────────────────────────────────────────────────────────────

class TestFitPredict:
    def test_predict_values_in_set(self, tiny_df, tiny_cfg, feature_cols):
        train, _, split_idx = chronological_split(tiny_df, 0.8)
        model = build_isolation_forest(tiny_cfg)
        fit_isolation_forest(model, train, feature_cols)
        result = predict_and_score(model, tiny_df, feature_cols, split_idx)
        assert set(result["anomaly_if"].unique()).issubset({-1, 1})

    def test_anomaly_flag_is_binary(self, tiny_df, tiny_cfg, feature_cols):
        train, _, split_idx = chronological_split(tiny_df, 0.8)
        model = build_isolation_forest(tiny_cfg)
        fit_isolation_forest(model, train, feature_cols)
        result = predict_and_score(model, tiny_df, feature_cols, split_idx)
        assert set(result["anomaly"].unique()).issubset({0, 1})

    def test_anomaly_if_maps_to_anomaly(self, tiny_df, tiny_cfg, feature_cols):
        train, _, split_idx = chronological_split(tiny_df, 0.8)
        model = build_isolation_forest(tiny_cfg)
        fit_isolation_forest(model, train, feature_cols)
        result = predict_and_score(model, tiny_df, feature_cols, split_idx)
        # -1 in anomaly_if → 1 in anomaly
        assert (result.loc[result["anomaly_if"]==-1, "anomaly"] == 1).all()
        assert (result.loc[result["anomaly_if"]==1,  "anomaly"] == 0).all()

    def test_score_range(self, tiny_df, tiny_cfg, feature_cols):
        train, _, split_idx = chronological_split(tiny_df, 0.8)
        model = build_isolation_forest(tiny_cfg)
        fit_isolation_forest(model, train, feature_cols)
        result = predict_and_score(model, tiny_df, feature_cols, split_idx)
        assert result["anomaly_score"].min() >= 0.0
        assert result["anomaly_score"].max() <= 1.0

    def test_output_length(self, tiny_df, tiny_cfg, feature_cols):
        train, _, split_idx = chronological_split(tiny_df, 0.8)
        model = build_isolation_forest(tiny_cfg)
        fit_isolation_forest(model, train, feature_cols)
        result = predict_and_score(model, tiny_df, feature_cols, split_idx)
        assert len(result) == len(tiny_df)

    def test_split_labels(self, tiny_df, tiny_cfg, feature_cols):
        train, _, split_idx = chronological_split(tiny_df, 0.8)
        model = build_isolation_forest(tiny_cfg)
        fit_isolation_forest(model, train, feature_cols)
        result = predict_and_score(model, tiny_df, feature_cols, split_idx)
        assert set(result["split"].unique()) == {"train", "test"}

    def test_anomaly_rate_near_contamination(self, tiny_df, tiny_cfg, feature_cols):
        """Train anomaly rate should ≈ contamination (sklearn sets threshold this way)."""
        train, _, split_idx = chronological_split(tiny_df, 0.8)
        model = build_isolation_forest(tiny_cfg)
        fit_isolation_forest(model, train, feature_cols)
        result = predict_and_score(model, tiny_df, feature_cols, split_idx)
        train_rate = result[result["split"]=="train"]["anomaly"].mean()
        assert abs(train_rate - 0.05) < 0.02

# ── save / load ────────────────────────────────────────────────────────────────

class TestSaveLoad:
    def test_roundtrip(self, tiny_df, tiny_cfg, feature_cols, tmp_path):
        train, _, split_idx = chronological_split(tiny_df, 0.8)
        model = build_isolation_forest(tiny_cfg)
        fit_isolation_forest(model, train, feature_cols)
        path = str(tmp_path / "if.pkl")
        save_model(model, path)
        loaded = load_model(path)
        # Scores from original and loaded model must be identical
        X = tiny_df[feature_cols].to_numpy()
        np.testing.assert_array_equal(
            model.decision_function(X),
            loaded.decision_function(X)
        )