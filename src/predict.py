
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = joblib.load(ROOT / "models" / "best_failure_model.pkl")

def predict(machine_type, air_temp, process_temp, rpm, torque, tool_wear):
    row = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }])
    probability = float(MODEL.predict_proba(row)[0, 1])
    prediction = int(probability >= 0.50)
    if probability >= 0.70:
        alert = "HIGH RISK - Immediate maintenance recommended"
    elif probability >= 0.40:
        alert = "MEDIUM RISK - Schedule inspection"
    else:
        alert = "LOW RISK - Continue monitoring"
    return prediction, probability, alert
