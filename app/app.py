
import streamlit as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from predict import predict

st.set_page_config(page_title="Predictive Maintenance", page_icon="⚙️", layout="centered")

st.title("Industrial Machine Failure Prediction")
st.write("Enter machine operating parameters to estimate failure risk.")

machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
air = st.number_input("Air Temperature [K]", 295.0, 310.0, 300.0)
process = st.number_input("Process Temperature [K]", 300.0, 320.0, 310.0)
rpm = st.number_input("Rotational Speed [rpm]", 1000, 3000, 1500)
torque = st.number_input("Torque [Nm]", 3.0, 80.0, 40.0)
wear = st.number_input("Tool Wear [min]", 0, 250, 100)

if st.button("Predict Failure Risk"):
    pred, probability, alert = predict(machine_type, air, process, rpm, torque, wear)
    st.metric("Failure Probability", f"{probability*100:.1f}%")
    if pred:
        st.error(alert)
    else:
        st.success(alert)
    st.caption("This is a decision-support prototype, not a safety-certified industrial control system.")
