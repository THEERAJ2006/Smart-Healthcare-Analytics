# 🫁 Smart Healthcare Analytics — Image Module (Phase 2)

---

## 📁 Project Structure

```
healthcare_image/
│
├── image_model.py      ← Train SimpleCNN + MobileNetV2, compare, save best
├── gradcam.py          ← Grad-CAM heatmap generation
├── app_image.py        ← Streamlit dashboard (Image module)
│
├── data/
│   └── chest_xray/     ← Kaggle dataset (place here)
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       ├── val/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       └── test/
│           ├── NORMAL/
│           └── PNEUMONIA/
│
├── models/
│   ├── SimpleCNN.keras              ← Auto-created
│   ├── MobileNetV2.keras            ← Auto-created
│   ├── SimpleCNN_metrics.json       ← Test metrics
│   ├── MobileNetV2_metrics.json     ← Test metrics
│   └── best_image_model.json        ← Which model won
│
├── plots/
│   ├── SimpleCNN_training_history.png
│   ├── MobileNetV2_training_history.png
│   ├── SimpleCNN_confusion_matrix.png
│   └── MobileNetV2_confusion_matrix.png
│
└── requirements.txt
```

---

## ⚡ Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Download dataset from Kaggle
URL: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Extract so the structure is:
```
data/chest_xray/train/NORMAL/     ← ~1341 images
data/chest_xray/train/PNEUMONIA/  ← ~3875 images
data/chest_xray/val/NORMAL/       ← 8 images
data/chest_xray/val/PNEUMONIA/    ← 8 images
data/chest_xray/test/NORMAL/      ← 234 images
data/chest_xray/test/PNEUMONIA/   ← 390 images
```

### Step 3 — Train both models
```bash
python image_model.py
```

Expected output (approx):
```
SimpleCNN     Accuracy: 90.5%   AUC: 0.961   Sensitivity: 95.1%
MobileNetV2   Accuracy: 93.8%   AUC: 0.978   Sensitivity: 97.4%

[★] Best model (by AUC): MobileNetV2
```
Training time: ~15–30 min on CPU | ~5 min on GPU

### Step 4 — Launch dashboard
```bash
streamlit run app_image.py
```
Open → http://localhost:8501

---

## 🔍 What the Dashboard Does

| Feature | Description |
|---|---|
| Upload X-Ray | JPG / PNG chest X-ray |
| Prediction | Normal vs Abnormal (Pneumonia) |
| Confidence bar | Visual confidence score |
| Model comparison | Side-by-side SimpleCNN vs MobileNetV2 |
| Grad-CAM heatmap | Highlights which lung region influenced decision |
| Threshold slider | Adjust sensitivity vs specificity |
| Training plots | View accuracy/loss curves + confusion matrix |

---

## 🧠 Model Architecture Summary

### SimpleCNN
```
Input (224×224×3)
→ Conv2D(32) + BN + MaxPool + Dropout   ← Block 1
→ Conv2D(64) + BN + MaxPool + Dropout   ← Block 2
→ Conv2D(128) + BN + MaxPool + Dropout  ← Block 3
→ Conv2D(128) + BN + MaxPool + Dropout  ← Block 4
→ GlobalAveragePooling
→ Dense(256) + BN + Dropout(0.5)
→ Dense(1, sigmoid)
```

### MobileNetV2 (Transfer Learning)
```
Input (224×224×3)
→ MobileNetV2 base (ImageNet weights, frozen)  ← Phase A
→ GlobalAveragePooling
→ Dense(256) + BN + Dropout(0.5)
→ Dense(1, sigmoid)
→ [Fine-tune top 30 base layers]               ← Phase B
```

---

## 🔜 What's Coming Next

| Phase | Module | Status |
|---|---|---|
| 1 | NLP — Disease Prediction | ✅ Done |
| 2 | Image — X-Ray CNN | ✅ Done |
| 3 | Time Series — Vital Forecasting | 🔜 Next |
| 4 | Full Integration Dashboard | 🔜 Phase 4 |

---

## ⚠️ Disclaimer
For educational purposes only. Not a substitute for qualified radiologist review.
