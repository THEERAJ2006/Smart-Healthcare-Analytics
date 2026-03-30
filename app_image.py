# =============================================================
#  STREAMLIT DASHBOARD — Image Module (Phase 2)
#  File: app_image.py
#  Run: streamlit run app_image.py
# =============================================================

import streamlit as st
import numpy as np
import json
import os
import io
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from gradcam import generate_gradcam_figure

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="X-Ray Analysis | Healthcare AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e8eaed; }

    .card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #2e3250;
        border-radius: 16px;
        padding: 24px;
        margin: 8px 0;
    }

    .result-normal {
        background: #0d2318;
        border: 2px solid #2ecc71;
        border-radius: 12px;
        padding: 18px 24px;
        text-align: center;
    }
    .result-abnormal {
        background: #2d0d0d;
        border: 2px solid #e74c3c;
        border-radius: 12px;
        padding: 18px 24px;
        text-align: center;
    }

    .conf-bar-bg {
        background: #1e2130;
        border-radius: 8px;
        height: 28px;
        width: 100%;
        position: relative;
        overflow: hidden;
        margin: 8px 0;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.9rem;
        color: white;
    }

    .model-badge {
        display: inline-block;
        background: #1e2d4a;
        border: 1px solid #4a90e2;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.85rem;
        color: #7ab3f5;
        margin: 4px;
    }

    .metric-card {
        background: #1a1f2e;
        border: 1px solid #2e3250;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    .metric-val { font-size: 1.6rem; font-weight: 800; color: #4a90e2; }
    .metric-lbl { font-size: 0.78rem; color: #6c757d; margin-top: 2px; }

    .disclaimer {
        background: #2a1f00;
        border: 1px solid #f39c12;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #f8c471;
        margin-top: 16px;
    }

    .upload-hint {
        background: #1e2130;
        border: 2px dashed #2e3250;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)


# ── Model loader ─────────────────────────────────────────────

@st.cache_resource
def load_best_model():
    """Load the best model based on training comparison results."""
    meta_path = "models/best_image_model.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            info = json.load(f)
        best_name = info["best_model"]
    else:
        # Fallback: try MobileNetV2 first, then SimpleCNN
        best_name = None
        for name in ["MobileNetV2", "SimpleCNN"]:
            if os.path.exists(f"models/{name}.keras"):
                best_name = name
                break

    if best_name is None:
        return None, None, None

    model_path = f"models/{best_name}.keras"
    if not os.path.exists(model_path):
        return None, None, None

    model = tf.keras.models.load_model(model_path)

    # Load metrics if available
    metrics = {}
    metrics_path = f"models/{best_name}_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    return model, best_name, metrics


@st.cache_resource
def load_all_models():
    """Load both models for comparison display."""
    models_loaded = {}
    for name in ["SimpleCNN", "MobileNetV2"]:
        path = f"models/{name}.keras"
        if os.path.exists(path):
            try:
                models_loaded[name] = tf.keras.models.load_model(path)
            except Exception:
                pass
    return models_loaded


def predict_from_pil(pil_image: Image.Image, model, threshold: float = 0.5) -> dict:
    """Run prediction on a PIL image."""
    img_rgb = np.array(pil_image.convert("RGB"))
    resized  = cv2.resize(img_rgb, (224, 224)).astype(np.float32) / 255.0
    batch    = np.expand_dims(resized, 0)
    prob     = float(model.predict(batch, verbose=0)[0][0])
    label    = "Abnormal (Pneumonia)" if prob >= threshold else "Normal"
    conf     = prob * 100 if prob >= threshold else (1 - prob) * 100
    indication = "Pneumonia detected" if prob >= threshold else "No abnormality detected"
    severity = "high" if prob >= 0.85 else ("medium" if prob >= 0.65 else "low")
    return {
        "label": label, "confidence": round(conf, 2),
        "indication": indication, "severity": severity,
        "raw_prob": round(prob, 4),
    }


# ── Sidebar ───────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏥 Healthcare Analytics")
    st.markdown("---")
    st.markdown("### 📋 Module Status")
    st.success("✅ NLP Module — Active")
    st.success("✅ Image Module — Active")
    st.info("🔄 Time Series — Phase 3")
    st.markdown("---")

    # Model selector
    st.markdown("### ⚙️ Model Selection")
    model_choice = st.radio(
        "Choose model for analysis:",
        ["Best (Auto)", "SimpleCNN", "MobileNetV2"],
        index=0,
    )

    st.markdown("---")
    threshold = st.slider(
        "🎚️ Decision Threshold",
        min_value=0.30, max_value=0.80, value=0.50, step=0.05,
        help="Lower = more sensitive (catches more pneumonia but more false positives)"
    )

    show_gradcam = st.toggle("🔥 Show Grad-CAM Heatmap", value=True,
                              help="Visualise which regions influenced the decision")

    st.markdown("---")
    st.caption("⚠️ Not a substitute for radiologist review.")


# ── Header ────────────────────────────────────────────────────

st.markdown("""
<div style='text-align:center; padding: 20px 0 10px 0;'>
    <h1 style='font-size:2.4rem; font-weight:900;
               background: linear-gradient(135deg, #4a90e2, #7b5ea7);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        🫁 X-Ray Analysis Module
    </h1>
    <p style='color:#6c757d; font-size:1rem;'>
        CNN-powered chest X-ray classification · Pneumonia detection · Grad-CAM explanation
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Load model ────────────────────────────────────────────────

best_model, best_name, best_metrics = load_best_model()
all_models = load_all_models()

if best_model is None:
    st.error(
        "❌ No trained model found. Please train first:\n\n"
        "```bash\npython image_model.py\n```"
    )
    st.stop()

# Apply model choice
if model_choice == "SimpleCNN" and "SimpleCNN" in all_models:
    active_model = all_models["SimpleCNN"]
    active_name  = "SimpleCNN"
elif model_choice == "MobileNetV2" and "MobileNetV2" in all_models:
    active_model = all_models["MobileNetV2"]
    active_name  = "MobileNetV2"
else:
    active_model = best_model
    active_name  = best_name

st.markdown(f"<span class='model-badge'>🤖 Active Model: {active_name}</span>", unsafe_allow_html=True)
if active_name == best_name:
    st.markdown("<span class='model-badge'>⭐ Best Performer</span>", unsafe_allow_html=True)

# ── Model metrics banner ──────────────────────────────────────

if best_metrics:
    st.markdown("### 📊 Model Performance (Test Set)")
    m1, m2, m3, m4 = st.columns(4)
    metrics_items = [
        ("Accuracy",    f"{best_metrics.get('accuracy', 0)*100:.1f}%"),
        ("AUC Score",   f"{best_metrics.get('auc', 0):.3f}"),
        ("Sensitivity", f"{best_metrics.get('sensitivity', 0)*100:.1f}%"),
        ("Specificity", f"{best_metrics.get('specificity', 0)*100:.1f}%"),
    ]
    for col, (lbl, val) in zip([m1, m2, m3, m4], metrics_items):
        with col:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-val'>{val}</div>"
                f"<div class='metric-lbl'>{lbl}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
    st.markdown("")

st.markdown("---")

# ── Upload Section ────────────────────────────────────────────

st.markdown("### 📤 Upload Chest X-Ray")
col_upload, col_info = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload a chest X-ray image",
        type=["jpg", "jpeg", "png"],
        help="Upload a frontal chest X-ray (PA or AP view) in JPEG or PNG format"
    )

with col_info:
    st.markdown("""
    **📋 Supported Inputs:**
    - Frontal chest X-ray (PA/AP view)
    - JPEG or PNG format
    - Any resolution (auto-resized)

    **🎯 Model Detects:**
    - Normal lung tissue
    - Pneumonia (bacterial/viral)
    """)

# ── Analysis ─────────────────────────────────────────────────

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)
    img_rgb = np.array(pil_img.convert("RGB"))

    st.markdown("---")
    st.markdown("## 🔬 Analysis Results")

    # ── Side by side: original + result ──────────────────────
    col_img, col_res = st.columns([1, 1])

    with col_img:
        st.markdown("#### 🖼️ Uploaded X-Ray")
        st.image(pil_img, use_container_width=True, caption="Input Image")
        st.caption(f"Resolution: {pil_img.size[0]}×{pil_img.size[1]} px")

    with col_res:
        st.markdown("#### 🧠 Prediction")
        with st.spinner("Analyzing X-ray..."):
            result = predict_from_pil(pil_img, active_model, threshold=threshold)

        is_abnormal = "Abnormal" in result["label"]
        css_class   = "result-abnormal" if is_abnormal else "result-normal"
        icon        = "🚨" if is_abnormal else "✅"
        color       = "#e74c3c" if is_abnormal else "#2ecc71"

        st.markdown(
            f"<div class='{css_class}'>"
            f"<div style='font-size:2.5rem;'>{icon}</div>"
            f"<div style='font-size:1.4rem; font-weight:800; color:{color};'>"
            f"{result['label']}</div>"
            f"<div style='color:#adb5bd; margin-top:6px;'>{result['indication']}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown("")

        # Confidence bar
        conf_color = color
        st.markdown(
            f"<div style='margin-top:12px;'>"
            f"<div style='color:#adb5bd; font-size:0.85rem; margin-bottom:4px;'>"
            f"Confidence</div>"
            f"<div class='conf-bar-bg'>"
            f"<div class='conf-bar-fill' style='width:{result['confidence']}%;"
            f" background:linear-gradient(90deg,{conf_color}88,{conf_color});'>"
            f"{result['confidence']:.1f}%</div></div></div>",
            unsafe_allow_html=True
        )

        # Severity badge
        sev_colors = {"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"}
        sev_color  = sev_colors.get(result["severity"], "#adb5bd")
        st.markdown(
            f"<div style='margin-top:12px;'>"
            f"<span style='background:{sev_color}22; border:1px solid {sev_color};"
            f" border-radius:8px; padding:5px 14px; color:{sev_color}; font-weight:600;'>"
            f"Severity: {result['severity'].upper()}</span></div>",
            unsafe_allow_html=True
        )

        # Raw probability
        st.markdown(
            f"<div style='color:#4a5568; font-size:0.8rem; margin-top:10px;'>"
            f"Raw probability (pneumonia): {result['raw_prob']:.4f} | "
            f"Threshold: {threshold:.2f}</div>",
            unsafe_allow_html=True
        )

    # ── Model Comparison (if both trained) ───────────────────
    if len(all_models) == 2:
        st.markdown("---")
        st.markdown("### 🔄 Model Comparison")
        comp_cols = st.columns(2)
        for col, (name, mdl) in zip(comp_cols, all_models.items()):
            with col:
                r = predict_from_pil(pil_img, mdl, threshold=threshold)
                icon = "🚨" if "Abnormal" in r["label"] else "✅"
                col_c = "#e74c3c" if "Abnormal" in r["label"] else "#2ecc71"
                st.markdown(
                    f"<div style='background:#1e2130; border:1px solid #2e3250;"
                    f" border-radius:12px; padding:16px; text-align:center;'>"
                    f"<div style='color:#7ab3f5; font-weight:700; margin-bottom:8px;'>"
                    f"🤖 {name}</div>"
                    f"<div style='font-size:1.3rem; font-weight:800; color:{col_c};'>"
                    f"{icon} {r['label']}</div>"
                    f"<div style='color:#adb5bd; font-size:0.9rem; margin-top:4px;'>"
                    f"Confidence: {r['confidence']:.1f}%</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    # ── Grad-CAM Section ──────────────────────────────────────
    if show_gradcam:
        st.markdown("---")
        st.markdown("### 🔥 Grad-CAM — Explainability")
        st.caption("Heatmap shows which regions of the X-ray most influenced the model's decision.")
        with st.spinner("Generating Grad-CAM heatmap..."):
            try:
                fig = generate_gradcam_figure(
                    img_rgb, active_model,
                    prediction_label=result["label"],
                    confidence=result["confidence"],
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as e:
                st.warning(f"Grad-CAM could not be generated for this model: {e}")

    # ── Disclaimer ────────────────────────────────────────────
    st.markdown(
        "<div class='disclaimer'>"
        "⚠️ <b>DISCLAIMER:</b> This AI analysis is for <b>educational and research purposes only</b>. "
        "It does not replace a qualified radiologist's interpretation. "
        "All clinical decisions must be made by licensed medical professionals."
        "</div>",
        unsafe_allow_html=True
    )

else:
    st.markdown(
        "<div class='upload-hint'>"
        "<div style='font-size:3rem;'>🫁</div>"
        "<div style='font-size:1.1rem; margin-top:10px;'>Upload a chest X-ray above to begin analysis</div>"
        "<div style='font-size:0.85rem; margin-top:6px;'>Supports JPG, PNG formats</div>"
        "</div>",
        unsafe_allow_html=True
    )

# ── Training plots (if available) ─────────────────────────────
st.markdown("---")
with st.expander("📈 View Training History Plots"):
    for name in ["SimpleCNN", "MobileNetV2"]:
        plot_path = f"plots/{name}_training_history.png"
        cm_path   = f"plots/{name}_confusion_matrix.png"
        if os.path.exists(plot_path):
            st.markdown(f"**{name}**")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                if os.path.exists(plot_path):
                    st.image(plot_path, caption=f"{name} — Training History")
            with col_p2:
                if os.path.exists(cm_path):
                    st.image(cm_path, caption=f"{name} — Confusion Matrix")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#4a5568; font-size:0.8rem;'>"
    "Smart Healthcare Analytics System · Image Module v1.0 · "
    "Built with TensorFlow/Keras + Streamlit"
    "</div>",
    unsafe_allow_html=True
)
