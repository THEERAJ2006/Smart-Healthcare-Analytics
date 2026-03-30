# 🏥 Smart Healthcare Analytics System
## Phase 1 — NLP Module

---

## 📁 Project Structure

```
healthcare_nlp/
│
├── nlp_model.py          ← Train disease prediction model
├── treatment_engine.py   ← OTC treatment knowledge base
├── risk_engine.py        ← Risk score computation
├── app_nlp.py            ← Streamlit dashboard (NLP only)
│
├── data/
│   └── symptoms_disease.csv   ← Your Kaggle dataset (add here)
│
├── models/
│   └── nlp_model.pkl     ← Auto-created after training
│
└── requirements.txt
```

---

## ⚡ Quick Start (5 steps)

### Step 1 — Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Add your dataset (optional for first test)
Download from Kaggle:
- https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset

Place CSV as: `data/symptoms_disease.csv`

**If you skip this step**, the code uses a built-in demo dataset (15 diseases) so you can test immediately.

### Step 4 — Train the model
```bash
python nlp_model.py
```
This will:
- Load and preprocess data
- Train Naive Bayes + Logistic Regression
- Pick the best model
- Save to `models/nlp_model.pkl`
- Print accuracy report + demo prediction

Expected output:
```
[✔] Preprocessed 4920 records | 41 unique diseases
Model: Naive Bayes        Accuracy: 91.2%
Model: Logistic Regression  Accuracy: 93.7%
[★] Best model selected: Logistic Regression (accuracy=93.7%)
[✔] Model saved → models/nlp_model.pkl
```

### Step 5 — Launch the dashboard
```bash
streamlit run app_nlp.py
```
Open browser → http://localhost:8501

---

## 🧪 Test the risk engine standalone
```bash
python risk_engine.py
```

## 🧪 Test the treatment engine standalone
```bash
python treatment_engine.py
```

---

## 📊 Dataset Format

Your CSV should have these columns (any extra columns are ignored):

| symptoms | disease |
|---|---|
| fever cough headache | Flu |
| chest pain sweating nausea | Heart Attack |

The code also accepts:
- `symptom` → renamed to `symptoms`
- `prognosis` → renamed to `disease`

---

## 🔜 What's Coming Next

| Phase | Module | Status |
|---|---|---|
| 1 | NLP — Disease Prediction | ✅ Done |
| 2 | Image — X-Ray CNN | 🔜 Next |
| 3 | Time Series — Vital Forecasting | 🔜 Phase 3 |
| 4 | Full Integration Dashboard | 🔜 Phase 4 |

---

## ⚠️ Disclaimer
This system is for **educational and informational purposes only**.
It does NOT replace professional medical diagnosis or treatment.
Always consult a qualified healthcare provider.
