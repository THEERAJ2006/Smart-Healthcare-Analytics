# =============================================================
#  TIME SERIES MODULE — Health Trend Forecasting
#  File: timeseries_model.py
#  Usage: python timeseries_model.py
#
#  Supports: Heart Rate, Blood Pressure, SpO2, Temperature
#  Methods:  Moving Average, Trend Detection, Anomaly Flagging
# =============================================================

import numpy as np
import pandas as pd
import json
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)


# ---------------------------------------------------------------
# NORMAL RANGES — used for anomaly detection
# ---------------------------------------------------------------

NORMAL_RANGES = {
    "Heart Rate":        {"min": 60,  "max": 100, "unit": "bpm"},
    "Systolic BP":       {"min": 90,  "max": 120, "unit": "mmHg"},
    "Diastolic BP":      {"min": 60,  "max": 80,  "unit": "mmHg"},
    "SpO2":              {"min": 95,  "max": 100, "unit": "%"},
    "Temperature":       {"min": 36.1,"max": 37.2,"unit": "°C"},
    "Respiratory Rate":  {"min": 12,  "max": 20,  "unit": "breaths/min"},
}


# ---------------------------------------------------------------
# STEP 1 — Generate / Load Data
# ---------------------------------------------------------------

def generate_demo_data(metric: str = "Heart Rate",
                       n_points: int = 30,
                       scenario: str = "worsening") -> pd.DataFrame:
    """
    Generate realistic demo vital sign data.

    Scenarios:
      'worsening'  → values gradually increase/worsen
      'stable'     → values stay within normal range
      'improving'  → values gradually return to normal
      'critical'   → spike into danger zone
    """
    np.random.seed(42)
    rng    = NORMAL_RANGES.get(metric, {"min": 60, "max": 100, "unit": ""})
    mid    = (rng["min"] + rng["max"]) / 2
    spread = (rng["max"] - rng["min"]) / 4

    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=n_points, freq="h")

    if scenario == "worsening":
        trend  = np.linspace(0, spread * 2.5, n_points)
        noise  = np.random.normal(0, spread * 0.3, n_points)
        values = mid + trend + noise
    elif scenario == "stable":
        noise  = np.random.normal(0, spread * 0.3, n_points)
        values = mid + noise
    elif scenario == "improving":
        trend  = np.linspace(spread * 2, 0, n_points)
        noise  = np.random.normal(0, spread * 0.2, n_points)
        values = (rng["max"] + spread) - trend + noise
    elif scenario == "critical":
        values = np.random.normal(mid, spread * 0.3, n_points)
        # Spike in last 5 readings
        values[-5:] = rng["max"] + spread * 3 + np.random.normal(0, spread * 0.2, 5)
    else:
        values = np.random.normal(mid, spread * 0.3, n_points)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "value":     np.round(values, 1),
        "metric":    metric,
    })
    return df


def load_or_generate(csv_path: str = None,
                     metric: str = "Heart Rate",
                     scenario: str = "worsening") -> pd.DataFrame:
    """Load CSV if available, else generate demo data."""
    if csv_path and os.path.exists(csv_path):
        print(f"[✔] Loading data from: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df.columns = df.columns.str.lower().str.strip()
        df = df.rename(columns={"time": "timestamp", "date": "timestamp"})
        df = df[["timestamp", "value"]].dropna()
        df["metric"] = metric
    else:
        print(f"[!] No CSV found — generating demo '{scenario}' data for {metric}")
        df = generate_demo_data(metric=metric, n_points=30, scenario=scenario)
    return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------
# STEP 2 — Moving Average Smoothing
# ---------------------------------------------------------------

def moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average with edge padding."""
    if len(values) < window:
        window = max(2, len(values) // 2)
    kernel = np.ones(window) / window
    # Pad edges to avoid shrinking
    padded = np.pad(values, (window//2, window//2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


def exponential_moving_average(values: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """EMA — more weight on recent values."""
    ema = np.zeros_like(values, dtype=float)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i-1]
    return ema


# ---------------------------------------------------------------
# STEP 3 — Trend Detection
# ---------------------------------------------------------------

def detect_trend(values: np.ndarray,
                 smoothed: np.ndarray,
                 metric: str) -> dict:
    """
    Detect overall trend using linear regression slope on smoothed values.
    Also checks if latest readings are in danger zone.

    Returns:
        trend        : "Increasing" | "Stable" | "Decreasing"
        direction    : "up" | "flat" | "down"
        slope        : float (change per reading)
        latest_value : float
        in_range     : bool
        anomalies    : list of indices where value is out of range
        severity     : "low" | "medium" | "high"
    """
    n      = len(smoothed)
    x      = np.arange(n)
    slope  = np.polyfit(x, smoothed, 1)[0]   # linear regression slope

    rng    = NORMAL_RANGES.get(metric, {"min": -np.inf, "max": np.inf})
    latest = float(values[-1])
    recent = values[-5:]   # last 5 readings

    # Trend classification based on slope magnitude
    threshold = (rng["max"] - rng["min"]) * 0.01  # 1% of range per step
    if slope > threshold:
        trend, direction = "Increasing", "up"
    elif slope < -threshold:
        trend, direction = "Decreasing", "down"
    else:
        trend, direction = "Stable", "flat"

    # Anomaly detection — readings outside normal range
    anomalies = [
        i for i, v in enumerate(values)
        if v < rng["min"] or v > rng["max"]
    ]

    # In range check for latest value
    in_range = rng["min"] <= latest <= rng["max"]

    # Recent anomaly ratio
    recent_anomaly_ratio = sum(
        1 for v in recent if v < rng["min"] or v > rng["max"]
    ) / len(recent)

    # Severity
    if not in_range and recent_anomaly_ratio >= 0.6:
        severity = "high"
    elif not in_range or recent_anomaly_ratio >= 0.4:
        severity = "medium"
    else:
        severity = "low"

    return {
        "trend":         trend,
        "direction":     direction,
        "slope":         round(float(slope), 4),
        "latest_value":  round(latest, 1),
        "mean_value":    round(float(np.mean(values)), 1),
        "min_value":     round(float(np.min(values)), 1),
        "max_value":     round(float(np.max(values)), 1),
        "in_range":      in_range,
        "anomalies":     anomalies,
        "anomaly_count": len(anomalies),
        "severity":      severity,
        "normal_min":    rng["min"],
        "normal_max":    rng["max"],
        "unit":          rng.get("unit", ""),
    }


# ---------------------------------------------------------------
# STEP 4 — Forecast (simple linear projection)
# ---------------------------------------------------------------

def forecast_next(smoothed: np.ndarray,
                  steps: int = 5) -> np.ndarray:
    """
    Project next N values using linear regression on recent trend.
    Simple but interpretable — appropriate for this project.
    """
    n = len(smoothed)
    x = np.arange(n)
    coeffs  = np.polyfit(x, smoothed, 1)
    x_future= np.arange(n, n + steps)
    return np.polyval(coeffs, x_future)


# ---------------------------------------------------------------
# STEP 5 — Full Analysis Pipeline
# ---------------------------------------------------------------

def analyze_vitals(df: pd.DataFrame,
                   window: int = 5,
                   forecast_steps: int = 5) -> dict:
    """
    Complete analysis: smooth → trend → forecast → summary.
    Returns a results dict used by both CLI and Streamlit app.
    """
    values    = df["value"].values.astype(float)
    metric    = df["metric"].iloc[0]
    timestamps= df["timestamp"].values

    smoothed   = moving_average(values, window=window)
    ema        = exponential_moving_average(values, alpha=0.3)
    trend_info = detect_trend(values, smoothed, metric)
    forecast   = forecast_next(smoothed, steps=forecast_steps)

    return {
        "metric":        metric,
        "timestamps":    timestamps,
        "values":        values,
        "smoothed":      smoothed,
        "ema":           ema,
        "forecast":      forecast,
        "forecast_steps":forecast_steps,
        "window":        window,
        **trend_info,
    }


# ---------------------------------------------------------------
# STEP 6 — Plot
# ---------------------------------------------------------------

def plot_vitals(result: dict, save_path: str = None) -> plt.Figure:
    """Generate a clean multi-panel vital signs chart."""
    metric   = result["metric"]
    values   = result["values"]
    smoothed = result["smoothed"]
    forecast = result["forecast"]
    n        = len(values)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(f"{metric} — Trend Analysis", color="white",
                 fontsize=14, fontweight="bold")

    # ── Panel 1: Raw + Smoothed + Forecast ───────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#1e2130")

    x = np.arange(n)
    x_fore = np.arange(n, n + result["forecast_steps"])

    # Normal range band
    ax1.axhspan(result["normal_min"], result["normal_max"],
                alpha=0.12, color="#2ecc71", label="Normal Range")
    ax1.axhline(result["normal_min"], color="#2ecc71", linewidth=0.8,
                linestyle="--", alpha=0.5)
    ax1.axhline(result["normal_max"], color="#2ecc71", linewidth=0.8,
                linestyle="--", alpha=0.5)

    # Raw values
    ax1.plot(x, values, color="#4a90e2", linewidth=1.2,
             alpha=0.5, label="Raw readings")
    ax1.scatter(x, values, color="#4a90e2", s=18, alpha=0.6, zorder=3)

    # Anomaly points
    if result["anomalies"]:
        ax1.scatter(result["anomalies"],
                    values[result["anomalies"]],
                    color="#e74c3c", s=60, zorder=5,
                    label=f"Anomalies ({result['anomaly_count']})", marker="x")

    # Smoothed line
    ax1.plot(x, smoothed, color="#f39c12", linewidth=2.5,
             label=f"Moving Avg (w={result['window']})")

    # Forecast
    ax1.plot(x_fore, forecast, color="#e74c3c", linewidth=2,
             linestyle="--", label="Forecast")
    ax1.fill_between(x_fore,
                     forecast * 0.97, forecast * 1.03,
                     alpha=0.15, color="#e74c3c")

    # Trend arrow annotation
    trend_colors = {"Increasing": "#e74c3c", "Stable": "#2ecc71", "Decreasing": "#3498db"}
    tc = trend_colors.get(result["trend"], "white")
    ax1.annotate(f"Trend: {result['trend']}",
                 xy=(0.02, 0.92), xycoords="axes fraction",
                 color=tc, fontsize=11, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e2130",
                           edgecolor=tc, alpha=0.8))

    ax1.set_ylabel(f"{metric} ({result['unit']})", color="#adb5bd")
    ax1.tick_params(colors="#adb5bd")
    ax1.spines[:].set_color("#2e3250")
    ax1.legend(loc="upper right", fontsize=8,
               facecolor="#1e2130", labelcolor="white")
    ax1.grid(alpha=0.15, color="white")

    # ── Panel 2: EMA + Stats bar ──────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#1e2130")

    ax2.plot(x, result["ema"], color="#7b5ea7", linewidth=2,
             label="EMA (α=0.3)")
    ax2.axhline(result["mean_value"], color="#adb5bd", linewidth=1,
                linestyle=":", label=f"Mean: {result['mean_value']} {result['unit']}")
    ax2.fill_between(x, result["ema"], result["mean_value"],
                     alpha=0.15, color="#7b5ea7")

    ax2.set_xlabel("Reading Number", color="#adb5bd")
    ax2.set_ylabel(f"EMA ({result['unit']})", color="#adb5bd")
    ax2.tick_params(colors="#adb5bd")
    ax2.spines[:].set_color("#2e3250")
    ax2.legend(loc="upper right", fontsize=8,
               facecolor="#1e2130", labelcolor="white")
    ax2.grid(alpha=0.15, color="white")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight",
                    facecolor="#0f1117")
        print(f"[✔] Plot saved → {save_path}")

    return fig


# ---------------------------------------------------------------
# STEP 7 — Risk contribution (used by risk_engine.py)
# ---------------------------------------------------------------

def get_timeseries_risk_input(result: dict) -> dict:
    """
    Returns a simplified dict compatible with risk_engine.TimeSeriesResult.
    """
    return {
        "trend":        result["trend"],
        "latest_value": result["latest_value"],
        "metric_name":  result["metric"],
        "severity":     result["severity"],
        "anomaly_count":result["anomaly_count"],
    }


# ---------------------------------------------------------------
# MAIN — Demo run
# ---------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  TIME SERIES MODULE — DEMO")
    print("="*60)

    scenarios = [
        ("Heart Rate",   "worsening"),
        ("Systolic BP",  "critical"),
        ("SpO2",         "improving"),
        ("Temperature",  "stable"),
    ]

    for metric, scenario in scenarios:
        print(f"\n[{metric}] Scenario: {scenario}")
        df     = load_or_generate(metric=metric, scenario=scenario)
        result = analyze_vitals(df)

        print(f"  Latest value : {result['latest_value']} {result['unit']}")
        print(f"  Trend        : {result['trend']}  (slope={result['slope']})")
        print(f"  In range     : {result['in_range']}")
        print(f"  Anomalies    : {result['anomaly_count']}")
        print(f"  Severity     : {result['severity'].upper()}")
        print(f"  Forecast +5  : {[round(v,1) for v in result['forecast']]}")

        plot_vitals(result,
                    save_path=f"plots/timeseries_{metric.replace(' ','_')}.png")

    print("\n[✔] All plots saved to plots/")
    print("    Run: python -m streamlit run app_timeseries.py")
