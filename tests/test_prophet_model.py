"""
tests/test_prophet_model.py
---------------------------
Unit + integration tests for src/models/prophet_model.py.

Run with:
    pytest tests/test_prophet_model.py -v

All tests use synthetic DataFrames — no disk I/O, no config.yaml,
no actual Prophet fit for fast unit tests. The one integration test
(test_full_pipeline) does a real tiny Prophet fit and is marked slow.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

# ── Import helpers — adjust path if running from repo root ───────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "models"))

from prophet_model import (
    prepare_prophet_df,
    train_test_split,
    build_prophet,
    predict_and_flag,
    print_metrics,
    save_results,
    ProphetAnomalyDetector,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_df():
    """200-row DataFrame mimicking icu_vitals_clean.csv."""
    np.random.seed(42)
    n = 200
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1min")
    return pd.DataFrame({
        "timestamp":   timestamps,
        "heart_rate":  70 + 5 * np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.randn(n),
        "spo2":        98 + np.random.randn(n) * 0.5,
        "patient_id":  1,
        "was_missing": 0,
    })


@pytest.fixture
def tiny_prophet_df(tiny_df):
    return prepare_prophet_df(tiny_df, "heart_rate")


@pytest.fixture
def train_test(tiny_prophet_df):
    return train_test_split(tiny_prophet_df, 0.8)


# ── prepare_prophet_df ────────────────────────────────────────────────────────

class TestPrepareProphetDf:
    def test_columns_renamed(self, tiny_df):
        result = prepare_prophet_df(tiny_df, "heart_rate")
        assert "ds" in result.columns
        assert "y"  in result.columns
        assert "timestamp"  not in result.columns
        assert "heart_rate" not in result.columns

    def test_only_two_columns(self, tiny_df):
        result = prepare_prophet_df(tiny_df, "heart_rate")
        assert list(result.columns) == ["ds", "y"]

    def test_nan_rows_dropped(self, tiny_df):
        tiny_df = tiny_df.copy()
        tiny_df.loc[0, "heart_rate"] = np.nan
        tiny_df.loc[5, "heart_rate"] = np.nan
        result = prepare_prophet_df(tiny_df, "heart_rate")
        assert len(result) == len(tiny_df) - 2
        assert result["y"].isna().sum() == 0

    def test_no_nans_unchanged(self, tiny_df):
        result = prepare_prophet_df(tiny_df, "heart_rate")
        assert len(result) == len(tiny_df)

    def test_missing_column_raises(self, tiny_df):
        with pytest.raises(KeyError):
            prepare_prophet_df(tiny_df, "nonexistent_vital")

    def test_ds_is_datetime(self, tiny_df):
        result = prepare_prophet_df(tiny_df, "heart_rate")
        assert pd.api.types.is_datetime64_any_dtype(result["ds"])


# ── train_test_split ──────────────────────────────────────────────────────────

class TestTrainTestSplit:
    def test_sizes(self, tiny_prophet_df):
        train, test = train_test_split(tiny_prophet_df, 0.8)
        assert len(train) + len(test) == len(tiny_prophet_df)
        assert len(train) == int(len(tiny_prophet_df) * 0.8)

    def test_train_before_test(self, tiny_prophet_df):
        train, test = train_test_split(tiny_prophet_df, 0.8)
        assert train["ds"].max() <= test["ds"].min()

    def test_no_overlap(self, tiny_prophet_df):
        train, test = train_test_split(tiny_prophet_df, 0.8)
        assert len(set(train.index) & set(test.index)) == 0

    def test_full_split_is_all_train(self, tiny_prophet_df):
        train, test = train_test_split(tiny_prophet_df, 1.0)
        assert len(train) == len(tiny_prophet_df)
        assert len(test) == 0

    def test_zero_split_is_all_test(self, tiny_prophet_df):
        train, test = train_test_split(tiny_prophet_df, 0.0)
        assert len(train) == 0
        assert len(test) == len(tiny_prophet_df)


# ── build_prophet ─────────────────────────────────────────────────────────────

class TestBuildProphet:
    def test_returns_prophet_instance(self):
        from prophet import Prophet
        model = build_prophet()
        assert isinstance(model, Prophet)

    def test_daily_seasonality_on(self):
        model = build_prophet()
        assert model.daily_seasonality is True

    def test_weekly_seasonality_off(self):
        model = build_prophet()
        assert model.weekly_seasonality is False

    def test_yearly_seasonality_off(self):
        model = build_prophet()
        assert model.yearly_seasonality is False

    def test_changepoint_prior_scale_respected(self):
        model = build_prophet(changepoint_prior_scale=0.2)
        assert model.changepoint_prior_scale == pytest.approx(0.2)

    def test_interval_width_is_99(self):
        model = build_prophet()
        assert model.interval_width == pytest.approx(0.99)


# ── predict_and_flag ──────────────────────────────────────────────────────────

class TestPredictAndFlag:
    """
    Uses a mock Prophet model so tests don't require an actual fit.
    """

    def _make_mock_model(self, prophet_df):
        """Return a mock model whose .predict() returns a minimal forecast."""
        n = len(prophet_df)
        forecast = pd.DataFrame({
            "ds":          prophet_df["ds"].values,
            "yhat":        prophet_df["y"].values,           # perfect forecast
            "yhat_lower":  prophet_df["y"].values - 5.0,
            "yhat_upper":  prophet_df["y"].values + 5.0,
        })
        model = MagicMock()
        model.predict.return_value = forecast
        return model

    def test_output_columns(self, tiny_prophet_df):
        model = self._make_mock_model(tiny_prophet_df)
        result = predict_and_flag(model, tiny_prophet_df, train_size=160)
        for col in ["ds", "y", "yhat", "yhat_lower", "yhat_upper",
                    "residual", "anomaly_score", "anomaly", "split"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_anomalies_when_perfect(self, tiny_prophet_df):
        """Perfect forecast: all y inside [y-5, y+5] → 0 anomalies."""
        model = self._make_mock_model(tiny_prophet_df)
        result = predict_and_flag(model, tiny_prophet_df, train_size=160)
        assert result["anomaly"].sum() == 0

    def test_anomaly_flagged_when_outside_band(self, tiny_prophet_df):
        """Inject a spike larger than the ±5 band — must be flagged."""
        df = tiny_prophet_df.copy()
        df.loc[10, "y"] = df.loc[10, "y"] + 100   # guaranteed outside band
        model = self._make_mock_model(tiny_prophet_df)   # band still ±5
        # Rebuild forecast for the original df; the spike row will be out of band
        n = len(df)
        forecast = pd.DataFrame({
            "ds":         df["ds"].values,
            "yhat":       tiny_prophet_df["y"].values,
            "yhat_lower": tiny_prophet_df["y"].values - 5.0,
            "yhat_upper": tiny_prophet_df["y"].values + 5.0,
        })
        model.predict.return_value = forecast
        result = predict_and_flag(model, df, train_size=160)
        assert result.loc[10, "anomaly"] == 1

    def test_split_labels_correct(self, tiny_prophet_df):
        model = self._make_mock_model(tiny_prophet_df)
        result = predict_and_flag(model, tiny_prophet_df, train_size=160)
        assert (result[result["split"] == "train"].index < 160).all()
        assert (result[result["split"] == "test"].index >= 160).all()

    def test_residual_is_y_minus_yhat(self, tiny_prophet_df):
        model = self._make_mock_model(tiny_prophet_df)
        result = predict_and_flag(model, tiny_prophet_df, train_size=160)
        expected = result["y"] - result["yhat"]
        pd.testing.assert_series_equal(result["residual"], expected, check_names=False)

    def test_anomaly_score_zero_for_perfect(self, tiny_prophet_df):
        """residual=0 → anomaly_score=0 for a perfect forecast."""
        model = self._make_mock_model(tiny_prophet_df)
        result = predict_and_flag(model, tiny_prophet_df, train_size=160)
        assert (result["anomaly_score"] == pytest.approx(0.0)).all()

    def test_anomaly_score_above_one_for_outliers(self, tiny_prophet_df):
        """Score > 1 for points outside the band."""
        df = tiny_prophet_df.copy()
        df.loc[10, "y"] = df.loc[10, "y"] + 100
        n = len(df)
        forecast = pd.DataFrame({
            "ds":         df["ds"].values,
            "yhat":       tiny_prophet_df["y"].values,
            "yhat_lower": tiny_prophet_df["y"].values - 5.0,
            "yhat_upper": tiny_prophet_df["y"].values + 5.0,
        })
        model = MagicMock()
        model.predict.return_value = forecast
        result = predict_and_flag(model, df, train_size=160)
        assert result.loc[10, "anomaly_score"] > 1.0


# ── save_results ──────────────────────────────────────────────────────────────

class TestSaveResults:
    def test_file_created(self, tiny_prophet_df, tmp_path):
        result = tiny_prophet_df.copy()
        result["yhat"] = result["y"]
        result["yhat_lower"] = result["y"] - 5
        result["yhat_upper"] = result["y"] + 5
        result["residual"] = 0.0
        result["anomaly_score"] = 0.0
        result["anomaly"] = 0
        result["split"] = "train"

        out_path = save_results(result, "heart_rate", out_dir=str(tmp_path))
        assert out_path.exists()

    def test_csv_roundtrip(self, tiny_prophet_df, tmp_path):
        result = tiny_prophet_df.copy()
        result["yhat"] = result["y"]
        result["yhat_lower"] = result["y"] - 5
        result["yhat_upper"] = result["y"] + 5
        result["residual"] = 0.0
        result["anomaly_score"] = 0.0
        result["anomaly"] = 0
        result["split"] = "train"

        out_path = save_results(result, "heart_rate", out_dir=str(tmp_path))
        loaded = pd.read_csv(out_path)
        assert len(loaded) == len(result)
        assert "anomaly" in loaded.columns


# ── ProphetAnomalyDetector (integration, slow) ────────────────────────────────

@pytest.mark.slow
class TestProphetAnomalyDetectorIntegration:
    """
    Runs a real (tiny) Prophet fit end-to-end.
    Skipped in CI unless --runslow is passed.
    Add to conftest.py:
        def pytest_addoption(parser):
            parser.addoption("--runslow", action="store_true")
        def pytest_runtest_setup(item):
            if "slow" in item.keywords and not item.config.getoption("--runslow"):
                pytest.skip("Pass --runslow to run this test")
    """

    def _make_temp_config(self, tmp_path, processed_path):
        """Write a minimal config.yaml pointing to a temp processed CSV."""
        cfg = {
            "data": {
                "raw_path":       str(tmp_path / "raw.csv"),
                "processed_path": str(processed_path),
                "resample_freq":  "1min",
                "freq_minutes":   1,
            },
            "vitals": ["heart_rate", "spo2"],
            "prophet": {
                "target_column":           "heart_rate",
                "train_split":             0.8,
                "changepoint_prior_scale": 0.05,
            },
        }
        import yaml
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(cfg))
        return str(config_path)

    def _make_processed_csv(self, tmp_path):
        """Write a 500-row synthetic processed CSV."""
        np.random.seed(0)
        n = 500
        timestamps = pd.date_range("2024-01-01", periods=n, freq="1min")
        df = pd.DataFrame({
            "timestamp":   timestamps,
            "heart_rate":  70 + 5 * np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.randn(n),
            "spo2":        98 + np.random.randn(n) * 0.5,
            "patient_id":  1,
            "was_missing": 0,
        })
        p = tmp_path / "icu_vitals_clean.csv"
        df.to_csv(p, index=False)
        return p

    def test_run_returns_dataframe(self, tmp_path):
        processed_path = self._make_processed_csv(tmp_path)
        config_path    = self._make_temp_config(tmp_path, processed_path)

        detector = ProphetAnomalyDetector(config_path=config_path)
        result   = detector.run()

        assert isinstance(result, pd.DataFrame)
        assert "anomaly" in result.columns
        assert result["anomaly"].isin([0, 1]).all()

    def test_anomaly_rate_reasonable(self, tmp_path):
        """For a nearly-sinusoidal signal, anomaly rate should be well under 10%."""
        processed_path = self._make_processed_csv(tmp_path)
        config_path    = self._make_temp_config(tmp_path, processed_path)

        detector = ProphetAnomalyDetector(config_path=config_path)
        result   = detector.run()

        anomaly_rate = result["anomaly"].mean()
        assert anomaly_rate < 0.10, (
            f"Anomaly rate {anomaly_rate:.1%} too high for a nearly-smooth signal. "
            "Check interval_width or changepoint_prior_scale."
        )

    def test_split_labels_present(self, tmp_path):
        processed_path = self._make_processed_csv(tmp_path)
        config_path    = self._make_temp_config(tmp_path, processed_path)

        detector = ProphetAnomalyDetector(config_path=config_path)
        result   = detector.run()

        assert set(result["split"].unique()) == {"train", "test"}