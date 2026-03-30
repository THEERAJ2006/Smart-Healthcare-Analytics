# =============================================================
#  STREAMLIT DASHBOARD — NLP Module (Phase 1)
#  File: app_nlp.py
#  Run: streamlit run app_nlp.py
# =============================================================

import streamlit as st
import pickle
import os
import numpy as np

# ── Import local modules ──────────────────────────────────────
from treatment_engine import get_treatment
from risk_engine import NLPResult, compute_risk_score

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Healthcare Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; color: #e8eaed; }

    /* Cards */
    .card {
        background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
        border: 1px solid #2e3250;
        border-radius: 16px;
        padding: 24px;
        margin: 8px 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    /* Risk badge */
    .risk-low    { background: #1a3a2a; border: 1px solid #2ecc71; color: #2ecc71; }
    .risk-medium { background: #3a2e1a; border: 1px solid #f39c12; color: #f39c12; }
    .risk-high   { background: #3a1a1a; border: 1px solid #e74c3c; color: #e74c3c; }
    .risk-badge {
        border-radius: 8px; padding: 8px 20px;
        font-size: 1.1rem; font-weight: 700;
        display: inline-block; margin: 4px 0;
    }

    /* Prediction bars */
    .pred-bar-wrap { margin: 6px 0; }
    .pred-label { font-size: 0.9rem; margin-bottom: 3px; color: #adb5bd; }
    .pred-bar-bg {
        background: #1e2130; border-radius: 6px;
        height: 22px; width: 100%;
        position: relative; overflow: hidden;
    }
    .pred-bar-fill {
        height: 100%; border-radius: 6px;
        background: linear-gradient(90deg, #4a90e2, #7b5ea7);
        transition: width 0.6s ease;
        display: flex; align-items: center; justify-content: flex-end;
        padding-right: 8px;
    }
    .pred-bar-text { font-size: 0.8rem; font-weight: 600; color: white; }

    /* Section headers */
    .section-title {
        font-size: 1.2rem; font-weight: 700;
        color: #4a90e2; margin: 16px 0 8px 0;
        border-bottom: 1px solid #2e3250; padding-bottom: 6px;
    }

    /* Symptom chips */
    .chip {
        display: inline-block; background: #1e2d4a;
        border: 1px solid #4a90e2; border-radius: 20px;
        padding: 4px 12px; margin: 4px;
        font-size: 0.85rem; color: #7ab3f5;
    }

    /* Medicine item */
    .med-item {
        background: #1a2535; border-left: 3px solid #4a90e2;
        padding: 8px 14px; margin: 5px 0; border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
    }

    /* Precaution item */
    .pre-item {
        background: #1a2535; border-left: 3px solid #2ecc71;
        padding: 8px 14px; margin: 5px 0; border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
    }

    /* Disclaimer box */
    .disclaimer {
        background: #2a1f00; border: 1px solid #f39c12;
        border-radius: 10px; padding: 12px 16px;
        font-size: 0.85rem; color: #f8c471; margin-top: 16px;
    }

    /* Gauge */
    .gauge-wrapper { text-align: center; padding: 10px; }
    .gauge-score {
        font-size: 3rem; font-weight: 900;
        background: linear-gradient(135deg, #4a90e2, #7b5ea7);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)


# ── Model loader ─────────────────────────────────────────────

@st.cache_resource
def load_nlp_model():
    model_path = "models/nlp_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None


def predict_disease_from_pipeline(symptoms_input: str, pipeline, top_n: int = 3) -> dict:
    import re
    cleaned = re.sub(r"[^a-z\s]", " ", symptoms_input.lower()).strip()
    proba = pipeline.predict_proba([cleaned])[0]
    classes = pipeline.named_steps["clf"].classes_
    sorted_idx = np.argsort(proba)[::-1]
    top_preds = [
        {"disease": classes[i], "probability": round(float(proba[i]) * 100, 2)}
        for i in sorted_idx[:top_n]
    ]
    return {
        "predicted_disease": top_preds[0]["disease"],
        "confidence": top_preds[0]["probability"],
        "top_predictions": top_preds,
    }


# ── Sidebar ───────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏥 Healthcare Analytics")
    st.markdown("---")
    st.markdown("### 📋 Module Status")
    st.success("✅ NLP Module — Active")
    st.info("🔄 Image Module — Phase 2")
    st.info("🔄 Time Series — Phase 2")
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "This system uses **NLP + ML** to predict diseases from symptoms, "
        "generate a risk score, and suggest basic precautions."
    )
    st.markdown("---")
    st.caption("⚠️ Not a substitute for professional medical advice.")


# ── Header ────────────────────────────────────────────────────

st.markdown("""
<div style='text-align:center; padding: 20px 0 10px 0;'>
    <h1 style='font-size:2.4rem; font-weight:900; 
               background: linear-gradient(135deg, #4a90e2, #7b5ea7);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        🏥 Smart Healthcare Analytics
    </h1>
    <p style='color:#6c757d; font-size:1rem;'>
        AI-powered disease prediction · Risk assessment · Treatment guidance
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Input Section ─────────────────────────────────────────────

col_in, col_guide = st.columns([2, 1])

with col_in:
    st.markdown("### 🔍 Enter Patient Symptoms")
    symptoms_input = st.text_area(
        label="Describe symptoms (comma or space separated):",
        placeholder="e.g. fever, cough, headache, fatigue, body ache",
        height=120,
        help="Enter all symptoms the patient is experiencing"
    )
    analyze_btn = st.button("🧠 Analyze Symptoms", type="primary", use_container_width=True)

with col_guide:
    st.markdown("### 💡 Example Symptoms")
    examples = {
        "Flu":          "fever, cough, body ache, fatigue, headache",
        "Dengue":       "high fever, joint pain, rash, eye pain, nausea",
        "Migraine":     "severe headache, nausea, light sensitivity, blurred vision",
        "Gastro":       "abdominal pain, nausea, vomiting, diarrhea",
        "Common Cold":  "sneezing, runny nose, sore throat, mild fever",
    }
    for disease, syms in examples.items():
        with st.expander(f"📌 {disease}"):
            st.code(syms)


# ── Results ───────────────────────────────────────────────────

if analyze_btn:
    if not symptoms_input.strip():
        st.warning("⚠️ Please enter at least one symptom.")
    else:
        pipeline = load_nlp_model()

        if pipeline is None:
            st.error(
                "❌ Model not found! Please train the model first:\n\n"
                "```bash\npython nlp_model.py\n```"
            )
        else:
            with st.spinner("🔍 Analyzing symptoms..."):
                result = predict_disease_from_pipeline(symptoms_input, pipeline, top_n=3)
                treatment = get_treatment(result["predicted_disease"])
                nlp_result = NLPResult(
                    predicted_disease=result["predicted_disease"],
                    confidence=result["confidence"],
                    top_predictions=result["top_predictions"],
                )
                risk = compute_risk_score(nlp_result)

            st.markdown("---")
            st.markdown("## 📊 Analysis Results")

            # ── Top metrics row ──────────────────────────────
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(
                    label="🦠 Predicted Disease",
                    value=result["predicted_disease"],
                    delta=f"{result['confidence']:.1f}% confidence"
                )
            with m2:
                level_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                st.metric(
                    label="⚠️ Risk Level",
                    value=f"{level_emoji.get(risk['level'], '⚪')} {risk['level']}",
                    delta=f"Score: {risk['score']}/100"
                )
            with m3:
                severity_map = {"low": "🟢 Low", "medium": "🟡 Medium", "high": "🔴 High"}
                st.metric(
                    label="📋 Disease Severity",
                    value=severity_map.get(treatment["severity"], "⚪ Unknown"),
                )

            st.markdown("")

            # ── Two columns: predictions + risk ─────────────
            col_pred, col_risk = st.columns([1, 1])

            with col_pred:
                st.markdown("#### 🎯 Top Predictions")
                for pred in result["top_predictions"]:
                    pct = pred["probability"]
                    color = "#4a90e2" if pct == result["confidence"] else "#7b5ea7"
                    st.markdown(f"""
                    <div class='pred-bar-wrap'>
                        <div class='pred-label'>{pred['disease']}</div>
                        <div class='pred-bar-bg'>
                            <div class='pred-bar-fill'
                                 style='width:{min(pct,100)}%; background: linear-gradient(90deg, {color}, #a29bfe);'>
                                <span class='pred-bar-text'>{pct:.1f}%</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Display detected symptoms as chips
                st.markdown("#### 🏷️ Detected Symptoms")
                symptom_chips = " ".join(
                    f"<span class='chip'>{s.strip()}</span>"
                    for s in symptoms_input.replace(",", " ").split()
                    if s.strip()
                )
                st.markdown(symptom_chips, unsafe_allow_html=True)

            with col_risk:
                st.markdown("#### 🚨 Risk Score")
                risk_color = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
                color = risk_color.get(risk["level"], "#adb5bd")
                st.markdown(f"""
                <div class='gauge-wrapper'>
                    <div class='gauge-score'>{risk['score']}</div>
                    <div style='color:{color}; font-size:1.3rem; font-weight:700;'>
                        {risk['level']} Risk
                    </div>
                    <div style='color:#6c757d; font-size:0.85rem; margin-top:6px;'>
                        out of 100
                    </div>
                    <div style='background:#1e2130; border-radius:10px; height:12px; 
                                width:100%; margin:12px auto;'>
                        <div style='background:{color}; width:{risk["score"]}%; 
                                    height:100%; border-radius:10px;'></div>
                    </div>
                    <div style='color:#adb5bd; font-size:0.9rem; margin-top:10px;
                                background:#1e2130; padding:10px; border-radius:8px;'>
                        {risk["action"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # ── Treatment Section ────────────────────────────
            st.markdown("## 💊 Treatment Recommendations")
            t_col1, t_col2 = st.columns(2)

            with t_col1:
                st.markdown("#### ✅ Precautions")
                for p in treatment["precautions"]:
                    st.markdown(f"<div class='pre-item'>✔ {p}</div>", unsafe_allow_html=True)

            with t_col2:
                st.markdown("#### 💊 OTC Medicines")
                for m in treatment["medicines"]:
                    st.markdown(f"<div class='med-item'>💊 {m}</div>", unsafe_allow_html=True)

            st.markdown("")
            st.markdown(
                f"<div class='disclaimer'>"
                f"🏥 <b>When to see a doctor:</b> {treatment['when_to_see_doctor']}<br><br>"
                f"⚠️ <b>DISCLAIMER:</b> This system is for informational purposes only and does NOT replace "
                f"professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider."
                f"</div>",
                unsafe_allow_html=True
            )

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#4a5568; font-size:0.8rem;'>"
    "Smart Healthcare Analytics System · NLP Module v1.0 · "
    "Built with Scikit-learn + Streamlit"
    "</div>",
    unsafe_allow_html=True
)
