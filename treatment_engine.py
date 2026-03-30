# =============================================================
#  TREATMENT RECOMMENDATION MODULE
#  File: treatment_engine.py
#  Used by: app.py (Streamlit dashboard)
# =============================================================

# ---------------------------------------------------------------
# Knowledge Base
# Format:
#   disease_name (lowercase) → {
#       precautions    : list[str]
#       medicines      : list[str]   (OTC only)
#       when_to_see_doctor : str
#       severity       : "low" | "medium" | "high"
#   }
# ---------------------------------------------------------------

TREATMENT_DB: dict = {
    "flu": {
        "precautions": ["Rest at home", "Stay hydrated (water, soups)", "Avoid contact with others", "Wear a mask"],
        "medicines": ["Paracetamol 500mg (for fever/pain)", "Cetirizine (for runny nose)", "Vitamin C supplements"],
        "when_to_see_doctor": "If fever exceeds 103°F or lasts > 3 days",
        "severity": "low",
    },
    "common cold": {
        "precautions": ["Rest well", "Drink warm fluids", "Gargle with warm salt water", "Steam inhalation"],
        "medicines": ["Paracetamol (for mild fever)", "Lozenges (for sore throat)", "Saline nasal drops"],
        "when_to_see_doctor": "If symptoms worsen after 7 days",
        "severity": "low",
    },
    "malaria": {
        "precautions": ["Use mosquito nets", "Apply insect repellent", "Avoid stagnant water areas", "Rest and hydrate"],
        "medicines": ["Consult doctor immediately — prescription antimalarials required"],
        "when_to_see_doctor": "IMMEDIATELY — malaria requires urgent medical treatment",
        "severity": "high",
    },
    "dengue": {
        "precautions": ["Complete bed rest", "Stay well hydrated", "Use mosquito repellent", "Monitor platelet count"],
        "medicines": ["Paracetamol (for fever — avoid Ibuprofen/Aspirin)", "ORS for hydration"],
        "when_to_see_doctor": "Immediately if bleeding, severe vomiting, or rapid breathing occurs",
        "severity": "high",
    },
    "gastroenteritis": {
        "precautions": ["Stay hydrated with ORS", "Eat bland foods (rice, toast)", "Avoid dairy and spicy food", "Wash hands frequently"],
        "medicines": ["ORS sachets", "Probiotics", "Loperamide (for diarrhea, adults only)"],
        "when_to_see_doctor": "If vomiting persists > 24 hrs or blood in stool",
        "severity": "medium",
    },
    "migraine": {
        "precautions": ["Rest in a dark quiet room", "Apply cold compress", "Avoid screen time", "Stay hydrated"],
        "medicines": ["Ibuprofen 400mg", "Paracetamol 1000mg", "Caffeine (in moderation)"],
        "when_to_see_doctor": "If headache is 'thunderclap' or worst in life — seek ER",
        "severity": "medium",
    },
    "diabetes": {
        "precautions": ["Monitor blood sugar regularly", "Follow a low-sugar diet", "Exercise daily (30 min)", "Avoid alcohol"],
        "medicines": ["As prescribed by doctor only — do not self-medicate"],
        "when_to_see_doctor": "Regularly — diabetes requires ongoing medical supervision",
        "severity": "high",
    },
    "arthritis": {
        "precautions": ["Gentle exercise (yoga, swimming)", "Apply warm/cold compress", "Maintain healthy weight", "Avoid joint strain"],
        "medicines": ["Ibuprofen 200-400mg (for pain)", "Topical diclofenac gel"],
        "when_to_see_doctor": "If joint swelling worsens or mobility is affected",
        "severity": "medium",
    },
    "asthma": {
        "precautions": ["Avoid known triggers (dust, pollen)", "Keep rescue inhaler handy", "Avoid cold air exposure", "Practice breathing exercises"],
        "medicines": ["Use prescribed inhaler", "Avoid OTC medicines without doctor advice"],
        "when_to_see_doctor": "Immediately if inhaler doesn't help or lips turn blue",
        "severity": "high",
    },
    "heart attack": {
        "precautions": ["CALL EMERGENCY SERVICES (112/911) IMMEDIATELY", "Sit or lie down calmly", "Chew aspirin if available and not allergic"],
        "medicines": ["Aspirin 325mg (if not allergic) — while awaiting emergency help ONLY"],
        "when_to_see_doctor": "EMERGENCY — Call ambulance immediately",
        "severity": "high",
    },
    "meningitis": {
        "precautions": ["SEEK EMERGENCY CARE IMMEDIATELY", "Avoid contact with others until cleared by doctor"],
        "medicines": ["Requires IV antibiotics — hospital treatment only"],
        "when_to_see_doctor": "EMERGENCY — This is life-threatening",
        "severity": "high",
    },
    "uti": {
        "precautions": ["Drink plenty of water", "Urinate frequently — don't hold it", "Avoid caffeinated drinks", "Maintain hygiene"],
        "medicines": ["Consult doctor for antibiotics", "Cranberry supplements (supportive)"],
        "when_to_see_doctor": "If fever develops or pain spreads to back/kidneys",
        "severity": "medium",
    },
    "jaundice": {
        "precautions": ["Rest completely", "Drink plenty of water and coconut water", "Avoid oily/fatty foods", "No alcohol"],
        "medicines": ["As prescribed — depends on cause (viral vs. other)"],
        "when_to_see_doctor": "Immediately — jaundice requires blood tests to determine cause",
        "severity": "high",
    },
    "anemia": {
        "precautions": ["Eat iron-rich foods (spinach, lentils, meat)", "Take Vitamin C with iron foods", "Avoid tea/coffee with meals", "Rest adequately"],
        "medicines": ["Iron supplements (ferrous sulfate) — after confirming with doctor", "Folic acid supplements"],
        "when_to_see_doctor": "If fatigue is severe or breathlessness on rest",
        "severity": "medium",
    },
    "tuberculosis": {
        "precautions": ["MUST follow complete prescribed TB treatment course", "Wear mask", "Ensure good ventilation", "Adequate nutrition"],
        "medicines": ["Requires prescription DOTS therapy — do NOT self-medicate"],
        "when_to_see_doctor": "Immediately — TB is infectious and needs proper treatment",
        "severity": "high",
    },
}

DEFAULT_RECOMMENDATION = {
    "precautions": ["Rest and stay hydrated", "Monitor your symptoms", "Avoid self-medication"],
    "medicines": ["Consult a doctor before taking any medicine"],
    "when_to_see_doctor": "If symptoms persist more than 2–3 days",
    "severity": "medium",
}


def get_treatment(disease_name: str) -> dict:
    """
    Return treatment recommendation for a given disease.
    Falls back to default if disease not in knowledge base.
    """
    key = disease_name.strip().lower()
    rec = TREATMENT_DB.get(key, DEFAULT_RECOMMENDATION).copy()
    rec["disease"] = disease_name
    rec["found_in_db"] = key in TREATMENT_DB
    return rec


def format_treatment_text(rec: dict) -> str:
    """Format treatment dict into a readable text block."""
    lines = []
    lines.append(f"Disease: {rec['disease']}")
    lines.append(f"Severity Level: {rec['severity'].upper()}")
    lines.append("\n✅ Precautions:")
    for p in rec["precautions"]:
        lines.append(f"  • {p}")
    lines.append("\n💊 Medicines (OTC):")
    for m in rec["medicines"]:
        lines.append(f"  • {m}")
    lines.append(f"\n🏥 When to See a Doctor:\n  {rec['when_to_see_doctor']}")
    lines.append("\n⚠️  DISCLAIMER: This is for informational purposes only.")
    lines.append("    Always consult a qualified doctor for proper diagnosis and treatment.")
    return "\n".join(lines)


# ---------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------
if __name__ == "__main__":
    for disease in ["Flu", "Dengue", "Heart Attack", "Unknown Disease XYZ"]:
        rec = get_treatment(disease)
        print("\n" + "="*60)
        print(format_treatment_text(rec))
