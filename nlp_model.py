# =============================================================
#  NLP MODULE — Disease Prediction from Symptoms
#  File: nlp_model.py
#  Usage: python nlp_model.py
# =============================================================

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import warnings
import random
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------
# STEP 1 — Load Dataset
# ---------------------------------------------------------------
# Dataset format expected: CSV with two columns:
#   'symptoms'  → comma-separated symptom text  e.g. "fever,cough,headache"
#   'disease'   → label                          e.g. "Flu"
#
# Download from Kaggle:
#   https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
#
# If you don't have it yet, a small SAMPLE dataset is generated below
# so you can test the pipeline immediately.
# ---------------------------------------------------------------

DATASET_PATH = r"data\symptoms_disease.csv\dataset.csv"   # ← Primary Kaggle dataset path
DATA_FOLDER = r"data"   # ← Folder to scan for additional datasets

# Symptom synonyms for data augmentation
SYMPTOM_SYNONYMS = {
    "fever": ["high temperature", "elevated temperature", "heat", "feverish"],
    "cough": ["coughing", "coughed", "persistent cough", "dry cough"],
    "headache": ["head pain", "migraine", "severe headache", "throbbing headache"],
    "fatigue": ["tiredness", "exhaustion", "weakness", "lethargy", "tired"],
    "body ache": ["muscle pain", "myalgia", "body pain", "muscular pain"],
    "nausea": ["feeling sick", "queasiness", "sickness", "vomiting urge"],
    "chest pain": ["chest discomfort", "thoracic pain", "cardiac pain"],
    "shortness of breath": ["dyspnea", "difficulty breathing", "breathlessness", "panting"],
    "rash": ["skin rash", "skin eruption", "skin lesion", "hives"],
    "diarrhea": ["loose stools", "watery stools", "bowel movement", "loose motion"],
    "vomiting": ["throwing up", "regurgitation", "emesis", "retching"],
    "sore throat": ["throat pain", "pharyngitis", "throat discomfort"],
    "runny nose": ["nasal discharge", "rhinorrhea", "stuffy nose"],
    "chills": ["shivering", "shivering with cold", "cold shivers"],
    "sweating": ["perspiration", "diaphoresis", "heavy sweating"],
}


def augment_symptoms(symptoms_text: str, num_variations: int = 1) -> list:
    """
    Generate variations of symptom text by replacing symptoms with synonyms.
    This creates multiple training samples from each record.
    """
    variations = [symptoms_text]
    words = symptoms_text.lower().split()
    
    for _ in range(num_variations):
        words_copy = words.copy()
        # Randomly replace some words with synonyms
        for i, word in enumerate(words_copy):
            for symptom, synonyms in SYMPTOM_SYNONYMS.items():
                if symptom.lower() in word.lower():
                    words_copy[i] = random.choice(synonyms)
                    break
        variations.append(" ".join(words_copy))
    
    return variations


def load_all_datasets(primary_path: str, data_folder: str) -> pd.DataFrame:
    """
    Load primary dataset + merge all compatible CSVs from data folder.
    Applies augmentation to increase training data diversity.
    """
    all_data = []
    
    # Load primary dataset
    if os.path.exists(primary_path):
        print(f"[✔] Loading primary dataset from: {primary_path}")
        df = pd.read_csv(primary_path)
        df.columns = df.columns.str.strip().str.lower()
        
        # Handle Kaggle format (Disease + Symptom_1, Symptom_2, ... columns)
        if "disease" in df.columns and any(col.startswith("symptom_") for col in df.columns):
            symptom_cols = [col for col in df.columns if col.startswith("symptom_")]
            df["symptoms"] = df[symptom_cols].fillna("").agg(" ".join, axis=1)
            df = df[["symptoms", "disease"]]
        
        df = df.dropna()
        all_data.append(df)
        print(f"   Loaded {len(df)} records")
    
    # Scan for additional datasets
    if os.path.exists(data_folder):
        for file in os.listdir(data_folder):
            if file.endswith(".csv") and file != "dataset.csv":
                filepath = os.path.join(data_folder, file)
                try:
                    print(f"[✔] Found additional dataset: {file}")
                    df_extra = pd.read_csv(filepath)
                    df_extra.columns = df_extra.columns.str.strip().str.lower()
                    
                    # Try to standardize columns
                    if "disease" in df_extra.columns:
                        if "symptoms" in df_extra.columns:
                            df_extra = df_extra[["symptoms", "disease"]]
                        elif "symptom" in df_extra.columns:
                            df_extra = df_extra.rename(columns={"symptom": "symptoms"})
                            df_extra = df_extra[["symptoms", "disease"]]
                        elif "prognosis" in df_extra.columns:
                            df_extra = df_extra.rename(columns={"prognosis": "disease"})
                            if "symptom" in df_extra.columns:
                                df_extra = df_extra.rename(columns={"symptom": "symptoms"})
                                df_extra = df_extra[["symptoms", "disease"]]
                        else:
                            continue
                        
                        df_extra = df_extra.dropna()
                        all_data.append(df_extra)
                        print(f"   Loaded {len(df_extra)} records")
                except Exception as e:
                    print(f"   ⚠ Could not load {file}: {e}")
    
    # Combine all datasets
    if all_data:
        df_combined = pd.concat(all_data, ignore_index=True)
        df_combined = df_combined.drop_duplicates()
        print(f"\n[✔] Combined dataset: {len(df_combined)} unique records")
        return df_combined
    
    # Fallback to demo data
    print("[!] No datasets found — using built-in demo data for testing.")
    demo = {
        "symptoms": [
            "fever cough headache fatigue",
            "fever body ache chills sweating",
            "sneezing runny nose sore throat mild fever",
            "chest pain shortness of breath sweating nausea",
            "abdominal pain nausea vomiting diarrhea",
            "headache nausea sensitivity to light blurred vision",
            "frequent urination excessive thirst fatigue blurred vision",
            "joint pain swelling redness stiffness",
            "cough shortness of breath wheezing chest tightness",
            "rash itching fever swollen lymph nodes",
            "high fever severe headache neck stiffness sensitivity to light",
            "burning urination frequent urination lower back pain",
            "yellowing skin dark urine fatigue abdominal pain",
            "fatigue pallor dizziness shortness of breath",
            "cough blood in sputum night sweats weight loss fever",
            "fever cough fatigue body ache",
            "chest pain difficulty breathing rapid heartbeat",
            "abdominal pain bloating gas indigestion",
            "headache fatigue dizziness blurred vision",
            "skin rash fever joint pain red eyes",
        ],
        "disease": [
            "Flu", "Malaria", "Common Cold", "Heart Attack",
            "Gastroenteritis", "Migraine", "Diabetes",
            "Arthritis", "Asthma", "Dengue",
            "Meningitis", "UTI", "Jaundice",
            "Anemia", "Tuberculosis", "Flu",
            "Heart Attack", "Gastroenteritis", "Migraine", "Dengue",
        ]
    }
    df = pd.DataFrame(demo)
    return df


# ---------------------------------------------------------------
# STEP 2 — Preprocessing
# ---------------------------------------------------------------

def preprocess_symptoms(text: str) -> str:
    """
    Clean and normalise a symptom string.
    - Lowercase
    - Remove commas / special chars
    - Strip extra whitespace
    """
    import re
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)   # keep only letters + spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["symptoms_clean"] = df["symptoms"].apply(preprocess_symptoms)
    df["disease"] = df["disease"].str.strip()
    print(f"[✔] Preprocessed {len(df)} records | {df['disease'].nunique()} unique diseases")
    return df


# ---------------------------------------------------------------
# STEP 3 — Train Models
# ---------------------------------------------------------------

def build_naive_bayes_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams
            max_features=5000,
            sublinear_tf=True
        )),
        ("clf", MultinomialNB(alpha=0.5))
    ])


def build_logistic_regression_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs"
        ))
    ])


def train_and_evaluate(df: pd.DataFrame):
    X = df["symptoms_clean"]
    y = df["disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(df) > 40 else None
    )

    results = {}

    for name, pipeline in [
        ("Naive Bayes",           build_naive_bayes_pipeline()),
        ("Logistic Regression",   build_logistic_regression_pipeline()),
    ]:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {"pipeline": pipeline, "accuracy": acc}
        print(f"\n{'='*50}")
        print(f"  Model : {name}")
        print(f"  Accuracy : {acc:.4f} ({acc*100:.1f}%)")
        print(f"{'='*50}")
        print(classification_report(y_test, y_pred, zero_division=0))

    # Choose best model
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_pipeline = results[best_name]["pipeline"]
    print(f"\n[★] Best model selected: {best_name}  "
          f"(accuracy={results[best_name]['accuracy']*100:.1f}%)")

    return best_pipeline, results


# ---------------------------------------------------------------
# STEP 4 — Save / Load
# ---------------------------------------------------------------

def save_model(pipeline: Pipeline, path: str = "models/nlp_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"[✔] Model saved → {path}")


def load_model(path: str = "models/nlp_model.pkl") -> Pipeline:
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    print(f"[✔] Model loaded from {path}")
    return pipeline


# ---------------------------------------------------------------
# STEP 5 — Prediction Function (used by app)
# ---------------------------------------------------------------

def predict_disease(symptoms_input: str, pipeline: Pipeline, top_n: int = 3) -> dict:
    """
    Given a raw symptoms string, return:
      - predicted_disease  : top predicted label
      - confidence         : probability of top prediction (%)
      - top_predictions    : list of (disease, probability%) — top_n entries
    """
    cleaned = preprocess_symptoms(symptoms_input)
    clf = pipeline.named_steps["clf"]

    # Probability scores
    proba = pipeline.predict_proba([cleaned])[0]
    classes = clf.classes_

    # Sort descending
    sorted_idx = np.argsort(proba)[::-1]
    top_predictions = [
        {"disease": classes[i], "probability": round(proba[i] * 100, 2)}
        for i in sorted_idx[:top_n]
    ]

    return {
        "predicted_disease": top_predictions[0]["disease"],
        "confidence": top_predictions[0]["probability"],
        "top_predictions": top_predictions,
    }


# ---------------------------------------------------------------
# MAIN — Train + Demo Prediction
# ---------------------------------------------------------------

if __name__ == "__main__":
    # 1. Load all datasets (primary + additional + augmented)
    df = load_all_datasets(DATASET_PATH, DATA_FOLDER)
    
    # 2. Apply data augmentation to increase dataset diversity
    print("\n[★] Applying data augmentation to increase training samples...")
    augmented_rows = []
    for _, row in df.iterrows():
        original = {
            "symptoms": row["symptoms"],
            "disease": row["disease"]
        }
        augmented_rows.append(original)
        
        # Generate 2 augmented variations per original sample
        variations = augment_symptoms(row["symptoms"], num_variations=2)
        for variation in variations[1:]:  # Skip original (already added)
            augmented_rows.append({
                "symptoms": variation,
                "disease": row["disease"]
            })
    
    df = pd.DataFrame(augmented_rows)
    print(f"[✔] After augmentation: {len(df)} total training samples")

    # 3. Preprocess
    df = preprocess_dataframe(df)

    # 4. Train
    best_model, all_results = train_and_evaluate(df)

    # 5. Save
    save_model(best_model)

    # 6. Demo prediction
    print("\n" + "="*60)
    print("  DEMO PREDICTION")
    print("="*60)
    test_symptoms = "fever cough headache body ache fatigue"
    result = predict_disease(test_symptoms, best_model)

    print(f"  Input Symptoms  : {test_symptoms}")
    print(f"  Predicted Disease : {result['predicted_disease']}")
    print(f"  Confidence       : {result['confidence']}%")
    print(f"\n  Top {len(result['top_predictions'])} Predictions:")
    for i, pred in enumerate(result["top_predictions"], 1):
        bar = "█" * int(pred["probability"] / 5)
        print(f"    {i}. {pred['disease']:<20} {pred['probability']:>6.1f}%  {bar}")
    print("="*60)
