# =============================================================
#  STREAMLIT DASHBOARD — Time Series Module (Phase 3)
#  File: app_timeseries.py
#  Run: python -m streamlit run app_timeseries.py
# =============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
from datetime import datetime, timedelta

from timeseries_model import (
    analyze_vitals, plot_vitals,
    generate_demo_data, load_or_generate,
    NORMAL_RANGES, get_timeseries_risk_input,
)

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Health Trends | Healthcare AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e8eaed; }

    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #2e3250;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        margin: 4px 0;
    }
    .metric-val { font-size: 1.8rem; font-weight: 800; }
    .metric-lbl { font-size: 0.78rem; color: #6c757d; margin-top: 3px; }
    .metric-unit{ font-size: 0.85rem; color: #adb5bd; }

    .trend-badge {
        display: inline-block;
        border-radius: 8px;
        padding: 6px 18px;
        font-size: 1rem;
        font-weight: 700;
        margin: 4px 0;
    }
    .trend-up   { background:#3a1a1a; border:1px solid #e74c3c; color:#e74c3c; }
    .trend-flat { background:#1a3a2a; border:1px solid #2ecc71; color:#2ecc71; }
    .trend-down { background:#1a2a3a; border:1px solid #3498db; color:#3498db; }

    .anomaly-box {
        background: #2d1a00;
        border: 1px solid #f39c12;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 0.88rem;
        color: #f8c471;
        margin: 8px 0;
    }
    .normal-box {
        background: #0d2318;
        border: 1px solid #2ecc71;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 0.88rem;
        color: #82e0aa;
        margin: 8px 0;
    }

    .forecast-item {
        background: #1a1f2e;
        border-left: 3px solid #e74c3c;
        padding: 6px 14px;
        margin: 4px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        display: flex;
        justify-content: space-between;
    }

    .vital-row {
        background: #1e2130;
        border: 1px solid #2e3250;
        border-radius: 10px;
        padding: 10px 16px;
        margin: 5px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .disclaimer {
        background: #2a1f00;
        border: 1px solid #f39c12;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 0.82rem;
        color: #f8c471;
        margin-top: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Healthcare Analytics")
    st.markdown("---")
    st.markdown("### 📋 Module Status")
    st.success("✅ NLP Module — Active")
    st.success("✅ Image Module — Active")
    st.success("✅ Time Series — Active")
    st.markdown("---")

    st.markdown("### ⚙️ Settings")
    metric_choice = st.selectbox(
        "📊 Vital Sign",
        list(NORMAL_RANGES.keys()),
        index=0,
    )

    data_source = st.radio(
        "📥 Data Source",
        ["Demo Data", "Upload CSV", "Manual Entry"],
        index=0,
    )

    if data_source == "Demo Data":
        scenario = st.select_slider(
            "Scenario",
            options=["improving", "stable", "worsening", "critical"],
            value="worsening",
        )

    ma_window = st.slider("📉 Moving Avg Window", 3, 10, 5)
    forecast_steps = st.slider("🔮 Forecast Steps", 3, 10, 5)
    st.markdown("---")
    st.caption("⚠️ Not a substitute for clinical monitoring.")


# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:20px 0 10px 0;'>
    <h1 style='font-size:2.4rem; font-weight:900;
               background:linear-gradient(135deg,#4a90e2,#7b5ea7);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        📈 Health Trend Forecasting
    </h1>
    <p style='color:#6c757d; font-size:1rem;'>
        Moving average smoothing · Trend detection · Vital sign forecasting · Anomaly alerts
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# ── Data Loading ─────────────────────────────────────────────

df = None

if data_source == "Demo Data":
    df = generate_demo_data(metric=metric_choice, n_points=30, scenario=scenario)
    st.info(f"📊 Using demo data — {metric_choice} | Scenario: **{scenario.title()}**")

elif data_source == "Upload CSV":
    uploaded = st.file_uploader(
        "Upload CSV with columns: timestamp, value",
        type=["csv"],
        help="CSV must have 'timestamp' and 'value' columns"
    )
    if uploaded:
        try:
            df = pd.read_csv(uploaded, parse_dates=["timestamp"])
            df.columns = df.columns.str.lower().str.strip()
            df["metric"] = metric_choice
            df = df[["timestamp", "value", "metric"]].dropna()
            st.success(f"✅ Loaded {len(df)} readings from CSV")
        except Exception as e:
            st.error(f"❌ Error reading CSV: {e}")
            st.info("Make sure your CSV has columns: `timestamp`, `value`")

elif data_source == "Manual Entry":
    st.markdown("### ✏️ Enter Readings Manually")
    st.caption("Enter comma-separated values (e.g. 72, 75, 80, 78, 82)")
    raw_input = st.text_area(
        "Values:",
        placeholder="72, 75, 80, 78, 82, 85, 88, 90, 87, 92",
        height=80,
    )
    if raw_input.strip():
        try:
            vals = [float(v.strip()) for v in raw_input.split(",") if v.strip()]
            now  = datetime.now()
            timestamps = [now - timedelta(hours=len(vals)-i) for i in range(len(vals))]
            df = pd.DataFrame({
                "timestamp": timestamps,
                "value":     vals,
                "metric":    metric_choice,
            })
            st.success(f"✅ Parsed {len(df)} readings")
        except Exception as e:
            st.error(f"❌ Could not parse values: {e}")


# ── Analysis ─────────────────────────────────────────────────

if df is not None and len(df) >= 5:
    result = analyze_vitals(df, window=ma_window, forecast_steps=forecast_steps)
    rng    = NORMAL_RANGES.get(metric_choice, {})

    st.markdown("## 📊 Analysis Results")

    # ── Top metric cards ──────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    trend_color = {"Increasing":"#e74c3c","Stable":"#2ecc71","Decreasing":"#3498db"}
    tc = trend_color.get(result["trend"], "#adb5bd")

    cards = [
        (f"{result['latest_value']}", result['unit'], "Latest Value", "#4a90e2"),
        (f"{result['mean_value']}",   result['unit'], "Mean Value",   "#7b5ea7"),
        (result["trend"],             "",             "Trend",        tc),
        (str(result["anomaly_count"]),"readings",     "Anomalies",    "#f39c12"
         if result["anomaly_count"] > 0 else "#2ecc71"),
        (result["severity"].upper(),  "",             "Severity",
         "#e74c3c" if result["severity"]=="high"
         else "#f39c12" if result["severity"]=="medium" else "#2ecc71"),
    ]
    for col, (val, unit, lbl, color) in zip([c1,c2,c3,c4,c5], cards):
        with col:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-val' style='color:{color};'>{val}</div>"
                f"<div class='metric-unit'>{unit}</div>"
                f"<div class='metric-lbl'>{lbl}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("")

    # ── Normal range status ───────────────────────────────────
    if result["in_range"]:
        st.markdown(
            f"<div class='normal-box'>✅ Latest reading "
            f"<b>{result['latest_value']} {result['unit']}</b> is within normal range "
            f"({rng.get('min','?')}–{rng.get('max','?')} {result['unit']})</div>",
            unsafe_allow_html=True,
        )
    else:
        diff = result["latest_value"] - rng.get("max", result["latest_value"])
        direction = "above" if diff > 0 else "below"
        st.markdown(
            f"<div class='anomaly-box'>⚠️ Latest reading "
            f"<b>{result['latest_value']} {result['unit']}</b> is "
            f"<b>{abs(diff):.1f} {result['unit']} {direction}</b> normal range "
            f"({rng.get('min','?')}–{rng.get('max','?')} {result['unit']})</div>",
            unsafe_allow_html=True,
        )

    # ── Chart ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📉 Trend Chart")
    with st.spinner("Generating chart..."):
        fig = plot_vitals(result)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Forecast + Raw data side by side ─────────────────────
    st.markdown("---")
    col_fore, col_raw = st.columns([1, 1])

    with col_fore:
        st.markdown("### 🔮 Forecast (Next Readings)")
        now = datetime.now()
        for i, fval in enumerate(result["forecast"], 1):
            future_time = now + timedelta(hours=i)
            in_norm = rng.get("min", -np.inf) <= fval <= rng.get("max", np.inf)
            color   = "#2ecc71" if in_norm else "#e74c3c"
            icon    = "✅" if in_norm else "⚠️"
            st.markdown(
                f"<div class='forecast-item'>"
                f"<span style='color:#adb5bd;'>+{i}h  {future_time.strftime('%H:%M')}</span>"
                f"<span style='color:{color}; font-weight:700;'>"
                f"{icon} {fval:.1f} {result['unit']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Trend explanation
        trend_explain = {
            "Increasing": "📈 Values are rising. Monitor closely for continued increase.",
            "Stable":     "➡️ Values are stable and within expected range.",
            "Decreasing": "📉 Values are declining. Could indicate improvement or concern.",
        }
        st.markdown("")
        st.info(trend_explain.get(result["trend"], ""))

    with col_raw:
        st.markdown("### 📋 Raw Readings")
        display_df = df[["timestamp", "value"]].copy()
        display_df.columns = ["Time", metric_choice]
        display_df["Time"] = display_df["Time"].dt.strftime("%Y-%m-%d %H:%M")

        # Color code out-of-range
        def highlight_anomaly(row):
            val = row[metric_choice]
            if val < rng.get("min", -np.inf) or val > rng.get("max", np.inf):
                return ["background-color: #3a1a1a"] * len(row)
            return [""] * len(row)

        st.dataframe(
            display_df.style.apply(highlight_anomaly, axis=1),
            use_container_width=True,
            height=320,
        )

    # ── All vitals overview ───────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏥 All Vitals Overview")
    st.caption("Quick-check all vital signs against normal ranges using current demo scenario")

    cols = st.columns(3)
    for i, (vital, norm) in enumerate(NORMAL_RANGES.items()):
        demo_df = generate_demo_data(
            metric=vital, n_points=10,
            scenario=scenario if data_source=="Demo Data" else "stable"
        )
        res = analyze_vitals(demo_df, window=3, forecast_steps=3)
        color = "#2ecc71" if res["in_range"] else "#e74c3c"
        icon  = "✅" if res["in_range"] else "⚠️"
        with cols[i % 3]:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div style='color:#adb5bd; font-size:0.8rem;'>{vital}</div>"
                f"<div class='metric-val' style='color:{color}; font-size:1.4rem;'>"
                f"{icon} {res['latest_value']} {norm['unit']}</div>"
                f"<div style='color:#6c757d; font-size:0.75rem;'>"
                f"Normal: {norm['min']}–{norm['max']} | Trend: {res['trend']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Disclaimer ────────────────────────────────────────────
    st.markdown(
        "<div class='disclaimer'>"
        "⚠️ <b>DISCLAIMER:</b> This forecasting tool is for educational purposes only. "
        "Vital sign trends must always be interpreted by a qualified healthcare professional. "
        "Do not make clinical decisions based solely on this output."
        "</div>",
        unsafe_allow_html=True,
    )

elif df is not None:
    st.warning("⚠️ Need at least 5 readings to perform trend analysis. Please add more data.")
else:
    st.markdown("""
    <div style='text-align:center; background:#1e2130; border:2px dashed #2e3250;
                border-radius:12px; padding:40px; color:#6c757d;'>
        <div style='font-size:3rem;'>📈</div>
        <div style='font-size:1.1rem; margin-top:10px;'>
            Select a data source from the sidebar to begin analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#4a5568; font-size:0.8rem;'>"
    "Smart Healthcare Analytics · Time Series Module v1.0 · "
    "Moving Average + Trend Detection + Forecasting"
    "</div>",
    unsafe_allow_html=True,
)
