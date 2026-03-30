# =============================================================
#  RISK SCORING ENGINE
#  File: risk_engine.py
#  Combines NLP + Image + Time Series outputs → final risk score
# =============================================================

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------
# Data Classes for module outputs
# ---------------------------------------------------------------

@dataclass
class NLPResult:
    predicted_disease: str
    confidence: float           # 0–100
    top_predictions: list = field(default_factory=list)


@dataclass
class ImageResult:
    label: str                  # "Normal" | "Abnormal"
    confidence: float           # 0–100
    indication: str = ""        # e.g. "Pneumonia"


@dataclass
class TimeSeriesResult:
    trend: str                  # "Increasing" | "Stable" | "Decreasing"
    latest_value: Optional[float] = None
    metric_name: str = "Heart Rate"


# ---------------------------------------------------------------
# Severity map — how risky each predicted disease is
# ---------------------------------------------------------------

DISEASE_SEVERITY: dict = {
    # High severity (score contribution: 60–80)
    "heart attack": 80, "meningitis": 80, "tuberculosis": 75,
    "malaria": 70, "dengue": 70, "jaundice": 65, "asthma": 65,
    "diabetes": 60,
    # Medium severity (score contribution: 35–55)
    "anemia": 50, "arthritis": 45, "uti": 45, "migraine": 40,
    "gastroenteritis": 40,
    # Low severity (score contribution: 10–30)
    "flu": 30, "common cold": 15,
}
DEFAULT_SEVERITY = 40  # unknown diseases


def _get_disease_base_score(disease: str) -> float:
    return DISEASE_SEVERITY.get(disease.strip().lower(), DEFAULT_SEVERITY)


# ---------------------------------------------------------------
# Individual component scorers (each returns 0–100)
# ---------------------------------------------------------------

def score_nlp(nlp: NLPResult) -> float:
    """
    NLP score = disease base severity × (confidence / 100)
    High confidence + severe disease = high score.
    """
    base = _get_disease_base_score(nlp.predicted_disease)
    score = base * (nlp.confidence / 100)
    return min(score, 100.0)


def score_image(img: Optional[ImageResult]) -> float:
    """
    Image score:
      Normal   → 10
      Abnormal → scales with confidence (50–95)
    """
    if img is None:
        return 0.0
    if img.label.lower() == "normal":
        return 10.0
    # Abnormal: scale from 50 to 95 based on confidence
    return 50 + (img.confidence / 100) * 45


def score_timeseries(ts: Optional[TimeSeriesResult]) -> float:
    """
    Time series score based on trend direction:
      Increasing  → 80
      Stable      → 40
      Decreasing  → 15
    """
    if ts is None:
        return 0.0
    mapping = {"increasing": 80, "stable": 40, "decreasing": 15}
    return mapping.get(ts.trend.lower(), 40)


# ---------------------------------------------------------------
# Combined Risk Score
# ---------------------------------------------------------------

WEIGHTS = {
    "nlp":         0.50,   # NLP (disease prediction) has most weight
    "image":       0.30,   # Image (visual evidence)
    "timeseries":  0.20,   # Trend (trajectory)
}


def compute_risk_score(
    nlp: NLPResult,
    image: Optional[ImageResult] = None,
    timeseries: Optional[TimeSeriesResult] = None,
) -> dict:
    """
    Compute a composite risk score in 0–100.

    Returns:
        score            : float (0–100)
        level            : "Low" | "Medium" | "High"
        component_scores : breakdown dict
        recommendation   : short action string
    """
    nlp_score = score_nlp(nlp)
    img_score  = score_image(image)
    ts_score   = score_timeseries(timeseries)

    # Adjust weights if modules are absent
    active_weight = WEIGHTS["nlp"]
    if image is not None:
        active_weight += WEIGHTS["image"]
    if timeseries is not None:
        active_weight += WEIGHTS["timeseries"]

    composite = (
        WEIGHTS["nlp"] * nlp_score
        + (WEIGHTS["image"] * img_score if image else 0)
        + (WEIGHTS["timeseries"] * ts_score if timeseries else 0)
    ) / active_weight

    composite = round(min(composite, 100.0), 1)

    # Risk level
    if composite < 30:
        level = "Low"
        action = "Monitor symptoms at home. Rest and stay hydrated."
    elif composite < 70:
        level = "Medium"
        action = "Consult a doctor within 24–48 hours. Follow precautions."
    else:
        level = "High"
        action = "⚠️ Seek medical attention immediately!"

    return {
        "score": composite,
        "level": level,
        "action": action,
        "component_scores": {
            "nlp_score":    round(nlp_score, 1),
            "image_score":  round(img_score, 1),
            "ts_score":     round(ts_score, 1),
        },
    }


# ---------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  RISK SCORING DEMO")
    print("=" * 55)

    # Scenario 1: High risk
    nlp1 = NLPResult(predicted_disease="Dengue", confidence=87.5)
    img1 = ImageResult(label="Abnormal", confidence=78.0, indication="Inflammation")
    ts1  = TimeSeriesResult(trend="Increasing", latest_value=105, metric_name="Heart Rate")
    r1 = compute_risk_score(nlp1, img1, ts1)
    print(f"\nScenario 1 (Dengue, Abnormal X-ray, Increasing HR)")
    print(f"  Score : {r1['score']} → {r1['level']} Risk")
    print(f"  Action: {r1['action']}")
    print(f"  Breakdown: {r1['component_scores']}")

    # Scenario 2: Low risk
    nlp2 = NLPResult(predicted_disease="Common Cold", confidence=92.0)
    img2 = ImageResult(label="Normal", confidence=95.0)
    ts2  = TimeSeriesResult(trend="Stable", latest_value=72, metric_name="Heart Rate")
    r2 = compute_risk_score(nlp2, img2, ts2)
    print(f"\nScenario 2 (Common Cold, Normal X-ray, Stable HR)")
    print(f"  Score : {r2['score']} → {r2['level']} Risk")
    print(f"  Action: {r2['action']}")

    # Scenario 3: NLP only (no image/timeseries)
    nlp3 = NLPResult(predicted_disease="Flu", confidence=65.0)
    r3 = compute_risk_score(nlp3)
    print(f"\nScenario 3 (NLP only — Flu)")
    print(f"  Score : {r3['score']} → {r3['level']} Risk")
    print(f"  Action: {r3['action']}")
