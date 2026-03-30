# =============================================================
#  SMART HEALTHCARE ANALYTICS — FULL INTEGRATION DASHBOARD
#  File: app.py  (Phase 4 — Final)
#  Run: python -m streamlit run app.py
#
#  Integrates:
#    ✅ NLP Module      — Disease prediction from symptoms
#    ✅ Image Module    — X-Ray classification (CNN)
#    ✅ Time Series     — Vital sign trend forecasting
#    ✅ Risk Engine     — Combined risk score (0–100)
#    ✅ Treatment       — Precautions + OTC medicines
# =============================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle, os, json, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from datetime import datetime, timedelta

# ── Local modules ─────────────────────────────────────────────
from treatment_engine  import get_treatment
from risk_engine       import NLPResult, ImageResult, TimeSeriesResult, compute_risk_score
from timeseries_model  import (analyze_vitals, plot_vitals,
                                generate_demo_data, NORMAL_RANGES)

# ── TensorFlow (optional — graceful fallback) ─────────────────
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Healthcare Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0a0d14; color: #e8eaed; }

    /* Gradient header */
    .main-header {
        background: linear-gradient(135deg, #0f1117 0%, #1a1f35 50%, #0f1117 100%);
        border: 1px solid #2e3250;
        border-radius: 20px;
        padding: 32px;
        text-align: center;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, #4a90e222 0%, transparent 60%);
        pointer-events: none;
    }

    /* Module cards */
    .module-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #2e3250;
        border-radius: 16px;
        padding: 20px;
        margin: 8px 0;
        transition: border-color 0.3s;
    }
    .module-card:hover { border-color: #4a90e2; }
    .module-title {
        font-size: 1rem; font-weight: 700;
        color: #4a90e2; margin-bottom: 10px;
        padding-bottom: 6px; border-bottom: 1px solid #2e3250;
    }

    /* Risk gauge */
    .risk-container {
        background: linear-gradient(135deg, #1e2130, #252840);
        border-radius: 20px;
        padding: 28px;
        text-align: center;
        border: 2px solid #2e3250;
    }
    .risk-score {
        font-size: 5rem; font-weight: 900;
        line-height: 1;
    }
    .risk-low    { color: #2ecc71; }
    .risk-medium { color: #f39c12; }
    .risk-high   { color: #e74c3c; }

    /* Progress bars */
    .prog-wrap { margin: 8px 0; }
    .prog-label {
        display: flex; justify-content: space-between;
        font-size: 0.82rem; color: #adb5bd; margin-bottom: 4px;
    }
    .prog-bg {
        background: #1e2130; border-radius: 6px;
        height: 10px; width: 100%; overflow: hidden;
    }
    .prog-fill { height: 100%; border-radius: 6px; }

    /* Prediction bars */
    .pred-bar-bg {
        background: #1e2130; border-radius: 6px;
        height: 24px; width: 100%;
        position: relative; overflow: hidden; margin: 5px 0;
    }
    .pred-bar-fill {
        height: 100%; border-radius: 6px;
        display: flex; align-items: center;
        justify-content: space-between;
        padding: 0 10px;
        font-size: 0.82rem; font-weight: 600; color: white;
    }

    /* Status badges */
    .badge {
        display: inline-block; border-radius: 20px;
        padding: 3px 12px; font-size: 0.8rem;
        font-weight: 600; margin: 2px;
    }
    .badge-green  { background:#0d2318; border:1px solid #2ecc71; color:#2ecc71; }
    .badge-yellow { background:#2a1f00; border:1px solid #f39c12; color:#f39c12; }
    .badge-red    { background:#2d0d0d; border:1px solid #e74c3c; color:#e74c3c; }
    .badge-blue   { background:#0d1a2d; border:1px solid #4a90e2; color:#4a90e2; }

    /* Treatment items */
    .pre-item {
        background:#0d2318; border-left:3px solid #2ecc71;
        padding:7px 14px; margin:4px 0; border-radius:0 8px 8px 0;
        font-size:0.88rem;
    }
    .med-item {
        background:#0d1a2d; border-left:3px solid #4a90e2;
        padding:7px 14px; margin:4px 0; border-radius:0 8px 8px 0;
        font-size:0.88rem;
    }

    /* Chip */
    .chip {
        display:inline-block; background:#1e2d4a;
        border:1px solid #4a90e2; border-radius:20px;
        padding:3px 10px; margin:3px;
        font-size:0.8rem; color:#7ab3f5;
    }

    /* Section divider */
    .section-header {
        font-size:1.1rem; font-weight:700; color:#4a90e2;
        border-bottom:1px solid #2e3250;
        padding-bottom:8px; margin:16px 0 10px 0;
    }

    .disclaimer {
        background:#2a1f00; border:1px solid #f39c12;
        border-radius:10px; padding:12px 16px;
        font-size:0.82rem; color:#f8c471; margin-top:16px;
    }

    /* Step indicator */
    .step-done   { color:#2ecc71; font-weight:700; }
    .step-active { color:#4a90e2; font-weight:700; }
    .step-wait   { color:#4a5568; }
</style>
""", unsafe_allow_html=True)


# ── Model loaders ────────────────────────────────────────────

@st.cache_resource
def load_nlp_model():
    path = "models/nlp_model.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_resource
def load_image_model():
    if not TF_AVAILABLE:
        return None, None
    for name in ["FastMobileNetV2", "MobileNetV2", "FastCNN", "SimpleCNN"]:
        path = f"models/{name}.keras"
        if os.path.exists(path):
            try:
                return tf.keras.models.load_model(path), name
            except Exception:
                continue
    return None, None


def predict_disease(symptoms: str, pipeline) -> dict:
    cleaned = re.sub(r"[^a-z\s]", " ", symptoms.lower()).strip()
    proba   = pipeline.predict_proba([cleaned])[0]
    classes = pipeline.named_steps["clf"].classes_
    idx     = np.argsort(proba)[::-1]
    top3    = [{"disease": classes[i],
                "probability": round(float(proba[i])*100, 2)}
               for i in idx[:3]]
    return {"predicted_disease": top3[0]["disease"],
            "confidence": top3[0]["probability"],
            "top_predictions": top3}

def predict_xray(pil_img, model) -> dict:
    arr = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    resized = cv2.resize(arr, (128, 128))
    prob    = float(model.predict(np.expand_dims(resized,0), verbose=0)[0][0])
    label   = "Abnormal" if prob >= 0.5 else "Normal"
    conf    = prob*100 if prob>=0.5 else (1-prob)*100
    sev     = "high" if prob>=0.85 else ("medium" if prob>=0.65 else "low")
    return {"label":label, "confidence":round(conf,2),
            "indication":"Pneumonia detected" if prob>=0.5 else "No abnormality",
            "severity":sev, "raw_prob":round(prob,4)}


# ── Sidebar ───────────────────────────────────────────────────

nlp_model   = load_nlp_model()
img_model, img_model_name = load_image_model()

with st.sidebar:
    st.markdown("## 🏥 Healthcare Analytics")
    st.markdown("---")

    st.markdown("### 📋 System Status")
    st.markdown(f"{'✅' if nlp_model   else '❌'} NLP Model")
    st.markdown(f"{'✅' if img_model   else '⚠️'} Image Model")
    st.markdown("✅ Time Series Engine")
    st.markdown("✅ Risk Engine")
    st.markdown("---")

    st.markdown("### 👤 Patient Info")
    patient_name = st.text_input("Patient Name", value="Patient")
    patient_age  = st.number_input("Age", min_value=1, max_value=120, value=35)
    patient_sex  = st.selectbox("Sex", ["Male", "Female", "Other"])
    st.markdown("---")
    st.caption("⚠️ For educational use only.")


# ── Header ────────────────────────────────────────────────────

st.markdown(f"""
<div class='main-header'>
    <div style='font-size:2.8rem; font-weight:900;
                background:linear-gradient(135deg,#4a90e2,#7b5ea7,#e74c3c);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        🏥 Smart Healthcare Analytics
    </div>
    <div style='color:#6c757d; margin-top:8px; font-size:1rem;'>
        NLP · Image Classification · Time Series · Risk Scoring · Treatment Guidance
    </div>
    <div style='margin-top:12px;'>
        <span class='badge badge-blue'>Patient: {patient_name}</span>
        <span class='badge badge-blue'>Age: {patient_age}</span>
        <span class='badge badge-blue'>{patient_sex}</span>
        <span class='badge badge-green'>{datetime.now().strftime("%d %b %Y")}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Step progress ─────────────────────────────────────────────
st.markdown("""
<div style='display:flex; justify-content:center; gap:8px;
            background:#1e2130; border-radius:12px; padding:12px; margin-bottom:16px;'>
    <span class='step-active'>① Symptoms</span>
    <span style='color:#2e3250;'>──────</span>
    <span class='step-active'>② X-Ray</span>
    <span style='color:#2e3250;'>──────</span>
    <span class='step-active'>③ Vitals</span>
    <span style='color:#2e3250;'>──────</span>
    <span class='step-active'>④ Risk Score</span>
    <span style='color:#2e3250;'>──────</span>
    <span class='step-active'>⑤ Treatment</span>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# MODULE 1 — NLP
# ═══════════════════════════════════════════════════════════════

st.markdown("## 🧠 Module 1 — Symptom Analysis (NLP)")
col_nlp_in, col_nlp_out = st.columns([1, 1])

with col_nlp_in:
    st.markdown("<div class='section-header'>Enter Symptoms</div>",
                unsafe_allow_html=True)
    symptoms_input = st.text_area(
        "Describe symptoms:",
        placeholder="e.g. fever, cough, headache, fatigue, body ache",
        height=100, label_visibility="collapsed",
    )
    quick_examples = {
        "Flu":         "fever cough body ache fatigue headache",
        "Dengue":      "high fever joint pain rash eye pain nausea",
        "Migraine":    "severe headache nausea light sensitivity blurred vision",
        "Pneumonia":   "cough fever chest pain difficulty breathing",
    }
    st.markdown("**Quick fill:**")
    ex_cols = st.columns(4)
    for col, (name, syms) in zip(ex_cols, quick_examples.items()):
        with col:
            if st.button(name, key=f"ex_{name}", use_container_width=True):
                symptoms_input = syms
                st.session_state["symptoms"] = syms

    if "symptoms" in st.session_state and not symptoms_input:
        symptoms_input = st.session_state["symptoms"]

with col_nlp_out:
    st.markdown("<div class='section-header'>NLP Result</div>",
                unsafe_allow_html=True)
    nlp_result_data = None

    if symptoms_input.strip() and nlp_model:
        nlp_result_data = predict_disease(symptoms_input, nlp_model)
        pred = nlp_result_data

        st.markdown(f"""
        <div style='background:#1e2130; border-radius:12px; padding:16px;'>
            <div style='color:#adb5bd; font-size:0.8rem;'>Top Prediction</div>
            <div style='font-size:1.5rem; font-weight:800; color:#4a90e2;'>
                {pred['predicted_disease']}
            </div>
            <div style='color:#2ecc71; font-size:0.9rem;'>
                Confidence: {pred['confidence']:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        for p in pred["top_predictions"]:
            color = "#4a90e2" if p == pred["top_predictions"][0] else "#7b5ea7"
            st.markdown(
                f"<div class='pred-bar-bg'>"
                f"<div class='pred-bar-fill' style='width:{min(p['probability'],100)}%;"
                f" background:linear-gradient(90deg,{color}88,{color});'>"
                f"<span>{p['disease']}</span>"
                f"<span>{p['probability']:.1f}%</span>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        chips = " ".join(
            f"<span class='chip'>{s.strip()}</span>"
            for s in re.sub(r"[,]", " ", symptoms_input).split() if s.strip()
        )
        st.markdown(chips, unsafe_allow_html=True)

    elif not nlp_model:
        st.warning("⚠️ NLP model not found. Run `python nlp_model.py` first.")
    else:
        st.info("👆 Enter symptoms above to see prediction")


# ═══════════════════════════════════════════════════════════════
# MODULE 2 — IMAGE
# ═══════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 🫁 Module 2 — X-Ray Analysis (CNN)")

col_img_in, col_img_out = st.columns([1, 1])
xray_result_data = None

with col_img_in:
    st.markdown("<div class='section-header'>Upload X-Ray</div>",
                unsafe_allow_html=True)
    uploaded_xray = st.file_uploader(
        "Upload chest X-ray",
        type=["jpg","jpeg","png"],
        label_visibility="collapsed",
    )
    if uploaded_xray:
        st.image(uploaded_xray, caption="Uploaded X-Ray",
                 use_container_width=True)

with col_img_out:
    st.markdown("<div class='section-header'>Image Result</div>",
                unsafe_allow_html=True)

    if uploaded_xray and img_model:
        pil_img = Image.open(uploaded_xray)
        xray_result_data = predict_xray(pil_img, img_model)
        xr = xray_result_data

        is_abn   = "Abnormal" in xr["label"]
        xr_color = "#e74c3c" if is_abn else "#2ecc71"
        xr_icon  = "🚨" if is_abn else "✅"

        st.markdown(f"""
        <div style='background:#1e2130; border-radius:12px; padding:16px; text-align:center;'>
            <div style='font-size:2.5rem;'>{xr_icon}</div>
            <div style='font-size:1.4rem; font-weight:800; color:{xr_color};'>
                {xr["label"]}
            </div>
            <div style='color:#adb5bd; font-size:0.9rem; margin-top:4px;'>
                {xr["indication"]}
            </div>
            <div style='margin-top:10px;'>
                <div style='color:#adb5bd; font-size:0.8rem;'>Confidence</div>
                <div style='background:#0a0d14; border-radius:8px; height:12px; margin:4px 0;'>
                    <div style='background:{xr_color}; width:{xr["confidence"]}%;
                                height:100%; border-radius:8px;'></div>
                </div>
                <div style='color:{xr_color}; font-weight:700;'>{xr["confidence"]:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"<span class='badge badge-blue'>Model: {img_model_name}</span>",
                    unsafe_allow_html=True)

    elif uploaded_xray and not img_model:
        st.warning("⚠️ Image model not found. Run `python image_model.py` first.")
        xray_result_data = {"label":"Unknown","confidence":50,
                            "indication":"Model unavailable","severity":"medium"}
    else:
        st.info("👆 Upload an X-ray image above")


# ═══════════════════════════════════════════════════════════════
# MODULE 3 — TIME SERIES
# ═══════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 📈 Module 3 — Vital Sign Trends (Time Series)")

col_ts_in, col_ts_out = st.columns([1, 1])
ts_result_data = None

with col_ts_in:
    st.markdown("<div class='section-header'>Vital Signs Input</div>",
                unsafe_allow_html=True)

    vital_metric  = st.selectbox("Select Vital Sign", list(NORMAL_RANGES.keys()))
    ts_input_mode = st.radio("Input Mode",
                             ["Demo Data", "Manual Entry"], horizontal=True)

    if ts_input_mode == "Demo Data":
        ts_scenario = st.select_slider(
            "Patient Scenario",
            options=["improving","stable","worsening","critical"],
            value="worsening",
        )
        ts_df = generate_demo_data(metric=vital_metric, n_points=24,
                                   scenario=ts_scenario)
        st.caption(f"Generated 24 hourly readings — scenario: **{ts_scenario}**")
    else:
        raw = st.text_area(
            "Enter comma-separated values:",
            placeholder="72, 75, 80, 78, 85, 90, 88, 92",
            height=70,
        )
        if raw.strip():
            try:
                vals = [float(v.strip()) for v in raw.split(",") if v.strip()]
                now  = datetime.now()
                ts_df = pd.DataFrame({
                    "timestamp": [now - timedelta(hours=len(vals)-i)
                                  for i in range(len(vals))],
                    "value":  vals,
                    "metric": vital_metric,
                })
            except Exception:
                st.error("❌ Invalid input — use comma-separated numbers")
                ts_df = generate_demo_data(metric=vital_metric, n_points=24)
        else:
            ts_df = generate_demo_data(metric=vital_metric, n_points=24)

with col_ts_out:
    st.markdown("<div class='section-header'>Trend Analysis</div>",
                unsafe_allow_html=True)
    ts_result = analyze_vitals(ts_df, window=5, forecast_steps=5)
    ts_result_data = ts_result

    trend_color = {"Increasing":"#e74c3c","Stable":"#2ecc71","Decreasing":"#3498db"}
    tc = trend_color.get(ts_result["trend"], "#adb5bd")
    in_range = ts_result["in_range"]

    m1, m2 = st.columns(2)
    with m1:
        st.markdown(
            f"<div style='background:#1e2130; border-radius:10px; padding:14px; text-align:center;'>"
            f"<div style='color:#adb5bd; font-size:0.78rem;'>Latest {vital_metric}</div>"
            f"<div style='font-size:1.8rem; font-weight:800; color:#4a90e2;'>"
            f"{ts_result['latest_value']}</div>"
            f"<div style='color:#adb5bd; font-size:0.78rem;'>{ts_result['unit']}</div>"
            f"</div>", unsafe_allow_html=True,
        )
    with m2:
        icon = "✅" if in_range else "⚠️"
        st.markdown(
            f"<div style='background:#1e2130; border-radius:10px; padding:14px; text-align:center;'>"
            f"<div style='color:#adb5bd; font-size:0.78rem;'>Trend</div>"
            f"<div style='font-size:1.3rem; font-weight:800; color:{tc};'>"
            f"{ts_result['trend']}</div>"
            f"<div style='font-size:0.8rem; color:{tc};'>{icon} "
            f"{'In range' if in_range else 'Out of range'}</div>"
            f"</div>", unsafe_allow_html=True,
        )

    # Mini trend chart
    fig_ts, ax = plt.subplots(figsize=(6, 2.5))
    fig_ts.patch.set_facecolor("#1e2130")
    ax.set_facecolor("#1e2130")
    ax.plot(ts_result["values"],   color="#4a90e266", linewidth=1.2)
    ax.plot(ts_result["smoothed"], color="#f39c12",   linewidth=2)
    n = len(ts_result["values"])
    ax.plot(range(n, n+5), ts_result["forecast"],
            color="#e74c3c", linewidth=2, linestyle="--")
    ax.axhspan(ts_result["normal_min"], ts_result["normal_max"],
               alpha=0.1, color="#2ecc71")
    ax.set_title(vital_metric, color="white", fontsize=9, pad=4)
    ax.tick_params(colors="#4a5568", labelsize=7)
    ax.spines[:].set_color("#2e3250")
    ax.grid(alpha=0.1, color="white")
    plt.tight_layout()
    st.pyplot(fig_ts, use_container_width=True)
    plt.close(fig_ts)

    if ts_result["anomaly_count"] > 0:
        st.markdown(
            f"<div style='background:#2a1f00; border:1px solid #f39c12; border-radius:8px;"
            f"padding:8px 14px; font-size:0.82rem; color:#f8c471;'>"
            f"⚠️ {ts_result['anomaly_count']} anomalous readings detected</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════
# MODULE 4 — RISK SCORE ENGINE
# ═══════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 🚨 Module 4 — Risk Score Engine")

# Build inputs for risk engine
nlp_input = None
img_input = None
ts_input  = None

if nlp_result_data:
    nlp_input = NLPResult(
        predicted_disease=nlp_result_data["predicted_disease"],
        confidence=nlp_result_data["confidence"],
        top_predictions=nlp_result_data["top_predictions"],
    )
if xray_result_data:
    img_input = ImageResult(
        label=xray_result_data["label"],
        confidence=xray_result_data["confidence"],
        indication=xray_result_data.get("indication",""),
    )
if ts_result_data:
    ts_input = TimeSeriesResult(
        trend=ts_result_data["trend"],
        latest_value=ts_result_data["latest_value"],
        metric_name=vital_metric,
    )

# Fallback if NLP not run yet
if nlp_input is None:
    nlp_input = NLPResult(predicted_disease="Unknown", confidence=50.0)

risk = compute_risk_score(nlp_input, img_input, ts_input)

risk_color = {"Low":"#2ecc71","Medium":"#f39c12","High":"#e74c3c"}
rc = risk_color.get(risk["level"], "#adb5bd")
risk_emoji = {"Low":"🟢","Medium":"🟡","High":"🔴"}
risk_score_val  = risk["score"]
risk_level_val  = risk["level"]
risk_action_val = risk["action"]

col_gauge, col_breakdown = st.columns([1, 2])

with col_gauge:
    st.markdown(
        f"<div class='risk-container'>"
        f"<div style='color:#adb5bd; font-size:0.85rem; margin-bottom:8px;'>"
        f"PATIENT RISK SCORE</div>"
        f"<div class='risk-score risk-{risk['level'].lower()}'>"
        f"{risk['score']}</div>"
        f"<div style='color:#6c757d; font-size:0.8rem;'>out of 100</div>"
        f"<div style='background:#0a0d14; border-radius:12px; height:16px;"
        f" width:100%; margin:14px 0;'>"
        f"<div style='background:linear-gradient(90deg,{rc}88,{rc});"
        f" width:{risk_score_val}%; height:100%; border-radius:12px;'></div></div>"
        f"<div style='font-size:1.5rem; font-weight:800; color:{rc};'>"
        f"{risk_emoji.get(risk_level_val,'')} {risk_level_val} Risk</div>"
        f"<div style='color:#adb5bd; font-size:0.85rem; margin-top:10px;"
        f" background:#0a0d14; padding:10px; border-radius:8px;'>"
        f"{risk_action_val}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with col_breakdown:
    st.markdown("<div class='section-header'>Score Breakdown</div>",
                unsafe_allow_html=True)

    comp = risk["component_scores"]
    breakdown_items = [
        ("🧠 NLP Score",        comp["nlp_score"],  "Disease prediction confidence × severity", "#4a90e2"),
        ("🫁 Image Score",      comp["image_score"],"X-ray abnormality level",                  "#7b5ea7"),
        ("📈 Time Series Score",comp["ts_score"],   "Vital sign trend severity",                "#f39c12"),
    ]
    weights = [50, 30, 20]

    for (label, score, desc, color), weight in zip(breakdown_items, weights):
        st.markdown(
            f"<div class='prog-wrap'>"
            f"<div class='prog-label'>"
            f"<span>{label} <span style='color:#4a5568;'>({weight}% weight)</span></span>"
            f"<span style='color:{color}; font-weight:700;'>{score:.1f}/100</span>"
            f"</div>"
            f"<div class='prog-bg'>"
            f"<div class='prog-fill' "
            f"style='width:{min(score,100)}%; background:linear-gradient(90deg,{color}66,{color});'>"
            f"</div></div>"
            f"<div style='color:#4a5568; font-size:0.75rem; margin-top:2px;'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")
    # Summary table
    summary_data = {
        "Module":  ["NLP",
                    "Image",
                    "Time Series"],
        "Finding": [
            nlp_result_data["predicted_disease"] if nlp_result_data else "—",
            xray_result_data["label"]             if xray_result_data else "Not provided",
            ts_result_data["trend"]               if ts_result_data  else "—",
        ],
        "Score": [
            f"{comp['nlp_score']:.1f}",
            f"{comp['image_score']:.1f}",
            f"{comp['ts_score']:.1f}",
        ],
    }
    st.dataframe(
        pd.DataFrame(summary_data),
        use_container_width=True, hide_index=True,
    )


# ═══════════════════════════════════════════════════════════════
# MODULE 5 — TREATMENT RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 💊 Module 5 — Treatment Recommendations")

disease_for_treatment = (nlp_result_data["predicted_disease"]
                          if nlp_result_data else "Flu")
treatment = get_treatment(disease_for_treatment)

col_pre, col_med, col_doc = st.columns(3)

with col_pre:
    st.markdown("<div class='section-header'>✅ Precautions</div>",
                unsafe_allow_html=True)
    for p in treatment["precautions"]:
        st.markdown(f"<div class='pre-item'>✔ {p}</div>",
                    unsafe_allow_html=True)

with col_med:
    st.markdown("<div class='section-header'>💊 OTC Medicines</div>",
                unsafe_allow_html=True)
    for m in treatment["medicines"]:
        st.markdown(f"<div class='med-item'>💊 {m}</div>",
                    unsafe_allow_html=True)

with col_doc:
    st.markdown("<div class='section-header'>🏥 Medical Advice</div>",
                unsafe_allow_html=True)

    sev_map   = {"low":"#2ecc71","medium":"#f39c12","high":"#e74c3c"}
    sev_color = sev_maps = sev_map.get(treatment["severity"], "#adb5bd")
    st.markdown(
        f"<div style='background:#1e2130; border-radius:10px; padding:14px;'>"
        f"<div style='color:#adb5bd; font-size:0.8rem;'>Disease</div>"
        f"<div style='font-size:1.2rem; font-weight:700; color:#4a90e2;'>"
        f"{disease_for_treatment}</div>"
        f"<div style='margin-top:8px;'>"
        f"<span style='background:{sev_color}22; border:1px solid {sev_color};"
        f" border-radius:6px; padding:3px 10px; color:{sev_color}; font-size:0.82rem;'>"
        f"Severity: {treatment['severity'].upper()}</span></div>"
        f"<div style='color:#adb5bd; font-size:0.85rem; margin-top:12px;"
        f" background:#0a0d14; padding:10px; border-radius:8px;'>"
        f"🏥 {treatment['when_to_see_doctor']}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ── Final Report Summary ──────────────────────────────────────
st.markdown("---")
st.markdown("## 📋 Final Patient Report")

report_col1, report_col2 = st.columns([1,1])
with report_col1:
    st.markdown(f"""
    <div style='background:#1e2130; border-radius:14px; padding:20px;
                border:1px solid #2e3250;'>
        <div style='color:#4a90e2; font-weight:700; font-size:1rem;
                    border-bottom:1px solid #2e3250; padding-bottom:8px; margin-bottom:12px;'>
            👤 {patient_name} | Age: {patient_age} | {patient_sex}
        </div>
        <div style='color:#adb5bd; font-size:0.85rem; line-height:1.8;'>
            <b style='color:white;'>Predicted Disease:</b>
            {nlp_result_data['predicted_disease'] if nlp_result_data else '—'}<br>
            <b style='color:white;'>NLP Confidence:</b>
            {nlp_result_data['confidence'] if nlp_result_data else '—'}%<br>
            <b style='color:white;'>X-Ray Finding:</b>
            {xray_result_data['label'] if xray_result_data else 'Not provided'}<br>
            <b style='color:white;'>Vital Trend:</b>
            {ts_result_data['trend'] if ts_result_data else '—'}
            ({vital_metric}: {ts_result_data['latest_value'] if ts_result_data else '—'}
            {ts_result_data['unit'] if ts_result_data else ''})<br>
            <b style='color:white;'>Anomalies:</b>
            {ts_result_data['anomaly_count'] if ts_result_data else '—'} readings<br>
        </div>
    </div>
    """, unsafe_allow_html=True)

with report_col2:
    st.markdown(
        f"<div style='background:#1e2130; border-radius:14px; padding:20px;"
        f" border:2px solid {rc}; text-align:center;'>"
        f"<div style='color:#adb5bd; font-size:0.85rem;'>FINAL RISK ASSESSMENT</div>"
        f"<div style='font-size:4rem; font-weight:900; color:{rc};'>"
        f"{risk['score']}</div>"
        f"<div style='font-size:1.2rem; font-weight:700; color:{rc};'>"
        f"{risk_emoji.get(risk['level'],'')} {risk['level']} Risk</div>"
        f"<div style='color:#adb5bd; font-size:0.85rem; margin-top:10px;'>"
        f"{risk['action']}</div>"
        f"<div style='margin-top:12px; color:#6c757d; font-size:0.75rem;'>"
        f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ── Disclaimer ────────────────────────────────────────────────
st.markdown(
    "<div class='disclaimer'>"
    "⚠️ <b>MEDICAL DISCLAIMER:</b> This system is built for <b>educational and "
    "research purposes only</b>. It does NOT constitute medical advice, diagnosis, "
    "or treatment. All outputs must be reviewed by a qualified healthcare professional. "
    "Never make clinical decisions based solely on AI predictions."
    "</div>",
    unsafe_allow_html=True,
)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#4a5568; font-size:0.8rem;'>"
    "Smart Healthcare Analytics System v1.0 · "
    "NLP + CNN + Time Series + Risk Engine · Built with TensorFlow & Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
