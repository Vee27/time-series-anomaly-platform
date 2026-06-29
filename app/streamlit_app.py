"""
streamlit_app.py
----------------
ICU Anomaly Detection Dashboard.

Loads results from three model CSVs and the ensemble comparison CSV.
Run from repo root:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ICU Anomaly Detection",
    page_icon="🏥",
    layout="wide",
)


def add_vline_datetime(fig, x, color="orange", dash="dash", label=""):
    """
    Workaround for Plotly add_vline bug with datetime x-axes.
    Uses add_shape + add_annotation instead.
    """
    x_str = str(x)
    fig.add_shape(
        type="line",
        x0=x_str, x1=x_str,
        y0=0, y1=1,
        yref="paper",
        line=dict(color=color, dash=dash, width=1.5),
    )
    if label:
        fig.add_annotation(
            x=x_str, y=1.02, yref="paper",
            text=label, showarrow=False,
            font=dict(size=11, color=color),
            xanchor="left",
        )

VITALS = ["heart_rate", "spo2", "resp_rate",
          "temperature", "systolic_bp", "diastolic_bp"]

# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data
def load_comparison():
    path = Path("results/model_comparison.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df

@st.cache_data
def load_if():
    path = Path("results/isolation_forest_results.csv")
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["timestamp"])

@st.cache_data
def load_lstm():
    path = Path("results/lstm_results.csv")
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["timestamp"])

@st.cache_data
def load_prophet():
    path = Path("results/prophet_heart_rate_results.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["ds"])
    df = df.rename(columns={"ds": "timestamp"})
    # Use residual Z-score col if available, else raw interval col
    if "anomaly_residual" not in df.columns:
        df["anomaly_residual"] = df["anomaly"]
    return df

# ── Load all data ─────────────────────────────────────────────────────────────
comparison = load_comparison()
df_if      = load_if()
df_lstm    = load_lstm()
df_prophet = load_prophet()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🏥 ICU Anomaly Detection")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "View",
    ["Overview", "Isolation Forest", "LSTM Autoencoder", "Prophet", "Ensemble"]
)

split_filter = st.sidebar.selectbox("Data split", ["All", "Train", "Test"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset**")
st.sidebar.markdown("Kaggle Human Vital Signs  \n200,020 rows · 1-min cadence  \n~138 days · 6 vitals")

# ── Helper: apply split filter ────────────────────────────────────────────────
def apply_split(df, col="split"):
    if split_filter == "All" or df is None:
        return df
    return df[df[col] == split_filter.lower()].copy()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("ICU Anomaly Detection — Overview")
    st.markdown("Three-model anomaly detection pipeline on ICU vital signs.")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)

    if df_if is not None:
        f = apply_split(df_if)
        col1.metric("IF Anomaly Rate",
                    f"{f['anomaly'].mean():.2%}",
                    f"{f['anomaly'].sum():,} flagged")

    if df_lstm is not None:
        f = apply_split(df_lstm)
        col2.metric("LSTM Anomaly Rate",
                    f"{f['anomaly'].mean():.2%}",
                    f"{f['anomaly'].sum():,} flagged")

    if df_prophet is not None:
        f = apply_split(df_prophet)
        col3.metric("Prophet Anomaly Rate",
                    f"{f['anomaly_residual'].mean():.2%}",
                    f"{f['anomaly_residual'].sum():,} flagged")

    if comparison is not None:
        f = apply_split(comparison)
        col4.metric("Ensemble (≥2 agree)",
                    f"{f['anomaly_ensemble'].mean():.2%}",
                    f"{f['anomaly_ensemble'].sum():,} flagged")

    st.markdown("---")

    # Model summary table
    st.subheader("Model Summary")
    summary_data = {
        "Model":          ["Prophet", "Isolation Forest", "LSTM Autoencoder"],
        "Input":          ["heart_rate only", "6 vitals + 56 features", "6 vitals (60-step windows)"],
        "Method":         ["Residual Z-score (z_thresh=3.0)",
                           "Unsupervised tree isolation (contamination=0.05)",
                           "Reconstruction MSE (mean + 3σ threshold)"],
        "Anomaly Rate":   [
            f"{df_prophet['anomaly_residual'].mean():.2%}" if df_prophet is not None else "N/A",
            f"{df_if['anomaly'].mean():.2%}"               if df_if      is not None else "N/A",
            f"{df_lstm['anomaly'].mean():.2%}"             if df_lstm    is not None else "N/A",
        ],
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    # Ensemble vote distribution (if comparison exists)
    if comparison is not None:
        st.markdown("---")
        st.subheader("Ensemble — Model Agreement Distribution")
        f = apply_split(comparison)
        vote_counts = f["vote_count"].value_counts().sort_index()
        fig = px.bar(
            x=[f"{v} model{'s' if v != 1 else ''}" for v in vote_counts.index],
            y=vote_counts.values,
            labels={"x": "Models in agreement", "y": "Row count"},
            color=vote_counts.index.astype(str),
            color_discrete_map={"0": "#4A90D9", "1": "#F5A623", "2": "#F8E71C", "3": "#D0021B"},
            title="How many models agree on each row",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ISOLATION FOREST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Isolation Forest":
    st.title("🌲 Isolation Forest")

    if df_if is None:
        st.error("results/isolation_forest_results.csv not found. Run isolation_forest/run.py first.")
        st.stop()

    f = apply_split(df_if)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total rows",    f"{len(f):,}")
    c2.metric("Anomalies",     f"{f['anomaly'].sum():,}")
    c3.metric("Anomaly rate",  f"{f['anomaly'].mean():.2%}")
    c4.metric("Mean score",    f"{f['anomaly_score'].mean():.4f}")

    st.markdown("---")

    # Score timeline
    st.subheader("Anomaly Score Over Time")
    sample = f.sample(min(20000, len(f)), random_state=42).sort_values("timestamp")
    fig = px.scatter(
        sample, x="timestamp", y="anomaly_score",
        color=sample["anomaly"].map({0: "normal", 1: "anomaly"}),
        color_discrete_map={"normal": "#4A90D9", "anomaly": "#D0021B"},
        opacity=0.4, size_max=3,
        labels={"anomaly_score": "Anomaly Score [0-1]", "color": ""},
        title="Isolation Forest — Anomaly Score (sampled 20k rows)",
    )
    # Add train/test boundary
    boundary = str(df_if[df_if["split"] == "test"]["timestamp"].min())
    add_vline_datetime(fig, boundary, color="orange", label="train/test split")
    st.plotly_chart(fig, use_container_width=True)

    # Score distribution
    st.subheader("Score Distribution — Normal vs Anomaly")
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=f[f["anomaly"]==0]["anomaly_score"],
        name="normal", opacity=0.6,
        marker_color="#4A90D9", histnorm="density", nbinsx=60
    ))
    fig2.add_trace(go.Histogram(
        x=f[f["anomaly"]==1]["anomaly_score"],
        name="anomaly", opacity=0.7,
        marker_color="#D0021B", histnorm="density", nbinsx=60
    ))
    fig2.update_layout(barmode="overlay",
                       xaxis_title="Anomaly Score",
                       yaxis_title="Density")
    st.plotly_chart(fig2, use_container_width=True)

    # Per-vital box plots
    st.subheader("Vital Sign Distribution — Normal vs Anomaly")
    vital_choice = st.selectbox("Select vital", VITALS)
    if vital_choice in f.columns:
        fig3 = px.box(
            f, x=f["anomaly"].map({0: "normal", 1: "anomaly"}),
            y=vital_choice,
            color=f["anomaly"].map({0: "normal", 1: "anomaly"}),
            color_discrete_map={"normal": "#4A90D9", "anomaly": "#D0021B"},
            title=f"{vital_choice} — normal vs anomaly",
            labels={"x": "", "y": vital_choice},
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Top anomalies table
    st.subheader("Top 20 Highest-Score Anomalies")
    top = f[f["anomaly"]==1].nlargest(20, "anomaly_score")
    show_cols = ["timestamp","anomaly_score","split"] + [v for v in VITALS if v in top.columns]
    st.dataframe(top[show_cols].reset_index(drop=True), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — LSTM AUTOENCODER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "LSTM Autoencoder":
    st.title("🧠 LSTM Autoencoder")

    if df_lstm is None:
        st.error("results/lstm_results.csv not found. Run lstm_autoencoder/run.py first.")
        st.stop()

    f = apply_split(df_lstm)
    threshold = f["threshold"].iloc[0]

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total windows",  f"{len(f):,}")
    c2.metric("Anomalies",      f"{f['anomaly'].sum():,}")
    c3.metric("Anomaly rate",   f"{f['anomaly'].mean():.2%}")
    c4.metric("Threshold",      f"{threshold:.4f}")

    st.markdown("---")

    # Threshold slider — live re-threshold without retraining
    st.subheader("Threshold Sensitivity")
    train_mse = df_lstm[df_lstm["split"]=="train"]["mse"]
    mu, sigma = float(train_mse.mean()), float(train_mse.std())
    multiplier = st.slider("Threshold multiplier (mean + N×std)",
                           min_value=1.5, max_value=5.0,
                           value=3.0, step=0.1)
    live_thresh = mu + multiplier * sigma
    live_anomalies = (f["mse"] > live_thresh).sum()
    live_rate = (f["mse"] > live_thresh).mean()

    st.info(f"**multiplier={multiplier:.1f}** → threshold={live_thresh:.6f} "
            f"→ **{live_anomalies:,} anomalies ({live_rate:.2%})**")

    # MSE timeline
    st.subheader("Reconstruction MSE Over Time")
    sample = f.sample(min(20000, len(f)), random_state=42).sort_values("timestamp")
    live_anom_col = (sample["mse"] > live_thresh).map({True: "anomaly", False: "normal"})
    fig = px.scatter(
        sample, x="timestamp", y="mse",
        color=live_anom_col,
        color_discrete_map={"normal": "#4A90D9", "anomaly": "#D0021B"},
        opacity=0.4,
        labels={"mse": "Reconstruction MSE", "color": ""},
        title="LSTM — Reconstruction Error (sampled 20k rows)",
    )
    fig.add_hline(y=live_thresh, line_dash="dash", line_color="orange",
                  annotation_text=f"threshold={live_thresh:.4f}")
    boundary = str(df_lstm[df_lstm["split"]=="test"]["timestamp"].min())
    add_vline_datetime(fig, boundary, color="purple", label="train/test split")
    st.plotly_chart(fig, use_container_width=True)

    # MSE distribution
    st.subheader("MSE Distribution — Normal vs Anomaly")
    live_flag = (f["mse"] > live_thresh).astype(int)
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=f[live_flag==0]["mse"], name="normal",
        opacity=0.6, marker_color="#4A90D9", histnorm="density", nbinsx=80
    ))
    fig2.add_trace(go.Histogram(
        x=f[live_flag==1]["mse"], name="anomaly",
        opacity=0.7, marker_color="#D0021B", histnorm="density", nbinsx=80
    ))
    fig2.add_vline(x=live_thresh, line_dash="dash", line_color="orange",
                   annotation_text="threshold")
    fig2.update_layout(barmode="overlay",
                       xaxis_title="MSE", yaxis_title="Density")
    st.plotly_chart(fig2, use_container_width=True)

    # Top anomalies
    st.subheader("Top 20 Highest-MSE Windows")
    top = f.nlargest(20, "mse")
    show_cols = ["timestamp","mse","threshold","anomaly","split"] + \
                [v for v in VITALS if v in top.columns]
    st.dataframe(top[show_cols].reset_index(drop=True), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PROPHET
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Prophet":
    st.title("📈 Prophet")

    if df_prophet is None:
        st.error("results/prophet_heart_rate_results.csv not found. Run prophet_model.py first.")
        st.stop()

    f = apply_split(df_prophet)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total rows",      f"{len(f):,}")
    c2.metric("Anomalies (Z-score)", f"{f['anomaly_residual'].sum():,}")
    c3.metric("Anomaly rate",    f"{f['anomaly_residual'].mean():.2%}")
    c4.metric("MAE (bpm)",       f"{f['residual'].abs().mean():.2f}")

    if "anomaly_residual" not in df_prophet.columns or \
       df_prophet["anomaly_residual"].equals(df_prophet["anomaly"]):
        st.warning("⚠️ `anomaly_residual` column not found — showing raw interval anomaly. "
                   "Re-run prophet_model.py with `flag_by_residual()` to get Z-score flags.")

    st.markdown("---")

    # Forecast vs actual
    st.subheader("Forecast vs Actual — heart_rate")
    sample = f.sample(min(10000, len(f)), random_state=42).sort_values("timestamp")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample["timestamp"], y=sample["y"],
        mode="lines", name="actual",
        line=dict(color="#4A90D9", width=0.8), opacity=0.8
    ))
    fig.add_trace(go.Scatter(
        x=sample["timestamp"], y=sample["yhat"],
        mode="lines", name="yhat",
        line=dict(color="#F5A623", width=1.2, dash="dash")
    ))
    if "yhat_upper" in sample.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([sample["timestamp"], sample["timestamp"][::-1]]),
            y=pd.concat([sample["yhat_upper"], sample["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(245,166,35,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="99% interval"
        ))
    # Anomaly markers
    anom_sample = sample[sample["anomaly_residual"]==1]
    fig.add_trace(go.Scatter(
        x=anom_sample["timestamp"], y=anom_sample["y"],
        mode="markers", name="anomaly (Z-score)",
        marker=dict(color="#D0021B", size=5)
    ))
    fig.update_layout(xaxis_title="Timestamp",
                      yaxis_title="heart_rate (bpm)")
    st.plotly_chart(fig, use_container_width=True)

    # Residual distribution
    st.subheader("Residual Distribution")
    fig2 = px.histogram(f, x="residual", nbins=100,
                         color=f["anomaly_residual"].map({0:"normal",1:"anomaly"}),
                         color_discrete_map={"normal":"#4A90D9","anomaly":"#D0021B"},
                         barmode="overlay", histnorm="density",
                         labels={"residual":"y − yhat (bpm)","color":""},
                         title="Prophet residuals — normal vs anomaly")
    if "residual_zscore" in f.columns:
        for thresh in [3.0, -3.0]:
            z_val = f["residual"].mean() + thresh * f["residual"].std()
            fig2.add_vline(x=z_val, line_dash="dash", line_color="orange",
                           annotation_text=f"z={thresh}")
    st.plotly_chart(fig2, use_container_width=True)

    # Anomaly score timeline
    if "anomaly_score" in f.columns:
        st.subheader("Anomaly Score Over Time")
        sample2 = f.sample(min(10000, len(f)), random_state=42).sort_values("timestamp")
        fig3 = px.scatter(
            sample2, x="timestamp", y="anomaly_score",
            color=sample2["anomaly_residual"].map({0:"normal",1:"anomaly"}),
            color_discrete_map={"normal":"#4A90D9","anomaly":"#D0021B"},
            opacity=0.4,
            labels={"anomaly_score":"Anomaly Score","color":""},
            title="Prophet — Anomaly Score (sampled)"
        )
        fig3.add_hline(y=1.0, line_dash="dash", line_color="orange",
                       annotation_text="score=1 (band edge)")
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Ensemble":
    st.title("🗳️ Ensemble — All Three Models")

    if comparison is None:
        st.error("results/model_comparison.csv not found. Run src/evaluation/metrics.py first.")
        st.stop()

    f = apply_split(comparison)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prophet anomalies",  f"{f['anomaly_prophet'].sum():,}",
              f"{f['anomaly_prophet'].mean():.2%}")
    c2.metric("IF anomalies",       f"{f['anomaly_if'].sum():,}",
              f"{f['anomaly_if'].mean():.2%}")
    c3.metric("LSTM anomalies",     f"{f['anomaly_lstm'].sum():,}",
              f"{f['anomaly_lstm'].mean():.2%}")
    c4.metric("Ensemble (≥2 agree)", f"{f['anomaly_ensemble'].sum():,}",
              f"{f['anomaly_ensemble'].mean():.2%}")

    st.markdown("---")

    # Vote count timeline
    st.subheader("Model Agreement Over Time")
    sample = f.sample(min(20000, len(f)), random_state=42).sort_values("timestamp")
    color_map = {"0":"#4A90D9","1":"#F5A623","2":"#F8E71C","3":"#D0021B"}
    fig = px.scatter(
        sample, x="timestamp", y="vote_count",
        color=sample["vote_count"].astype(str),
        color_discrete_map=color_map,
        opacity=0.4,
        labels={"vote_count":"Models flagging this row (0–3)","color":"vote count"},
        title="How many models flagged each row",
        category_orders={"color":["0","1","2","3"]},
    )
    boundary = str(f[f["split"]=="test"]["timestamp"].min())
    add_vline_datetime(fig, boundary, color="purple", label="train/test split")
    fig.update_yaxes(tickvals=[0,1,2,3])
    st.plotly_chart(fig, use_container_width=True)

    # Pairwise agreement
    st.subheader("Pairwise Model Agreement")
    pairs = [
        ("anomaly_prophet", "anomaly_if",   "Prophet ↔ IF"),
        ("anomaly_prophet", "anomaly_lstm",  "Prophet ↔ LSTM"),
        ("anomaly_if",      "anomaly_lstm",  "IF ↔ LSTM"),
    ]
    agree_data = []
    for col_a, col_b, label in pairs:
        if col_a in f.columns and col_b in f.columns:
            agree = (f[col_a] == f[col_b]).mean()
            both  = ((f[col_a]==1) & (f[col_b]==1)).sum()
            agree_data.append({"Pair": label,
                                "Agreement %": f"{agree:.1%}",
                                "Both flag": f"{both:,}"})
    st.dataframe(pd.DataFrame(agree_data), use_container_width=True, hide_index=True)

    # Unanimous anomalies table
    st.subheader("Unanimous Anomalies — All 3 Models Agree")
    unanimous = f[f["vote_count"]==3].sort_values("timestamp")
    st.markdown(f"**{len(unanimous):,} rows flagged by all three models** "
                f"({len(unanimous)/len(f):.2%} of total) — highest confidence anomalies.")
    if not unanimous.empty:
        st.dataframe(unanimous.reset_index(drop=True), use_container_width=True)

    # Ensemble flag timeline
    st.subheader("Ensemble Anomaly Flag Over Time")
    fig2 = px.scatter(
        sample, x="timestamp",
        y=sample["anomaly_ensemble"].map({0:0, 1:1}),
        color=sample["anomaly_ensemble"].map({0:"normal",1:"anomaly"}),
        color_discrete_map={"normal":"#4A90D9","anomaly":"#D0021B"},
        opacity=0.4,
        labels={"y":"anomaly (0/1)","color":""},
        title="Ensemble flag (≥2 models agree)",
    )
    fig2.update_yaxes(tickvals=[0,1], ticktext=["normal","anomaly"])
    st.plotly_chart(fig2, use_container_width=True)