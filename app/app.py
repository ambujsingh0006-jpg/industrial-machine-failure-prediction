

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict import predict

st.set_page_config(
    page_title="MachineGuard | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
)

st.markdown("""
<style>

.stApp {
    background: #f4f7fb;
    color: #172033;
}
.block-container {
    padding-top: 1.8rem;
    max-width: 1400px;
}
.hero {
    background: linear-gradient(120deg, #10233f, #155e75);
    padding: 28px;
    border-radius: 18px;
    color: white;
    margin-bottom: 22px;
}
.hero h1, .hero p {
    color: #ffffff !important;
}
h1, h2, h3, p, label,
[data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"] p {
    color: #172033 !important;
}
div[data-testid="stMetric"] {
    background: #ffffff;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #e3e9f1;
}
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: #172033 !important;
}
div[data-testid="stAlert"] p {
    color: #172033 !important;
}

</style>
""", unsafe_allow_html=True)

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

st.markdown("""
<div class="hero">
    <h1>⚙️ MachineGuard</h1>
    <p>Industrial Machine Failure Prediction & Predictive Maintenance</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("MachineGuard")
    page = st.radio(
        "Navigation",
        ["Live Prediction", "Batch Analysis", "Model Dashboard", "About Project"],
    )
    st.divider()
    st.caption("Machine Learning • Predictive Maintenance")
    st.caption("Model: trained classification pipeline")

if page == "Live Prediction":
    st.header("Machine Health Assessment")
    st.write("Enter machine operating readings to estimate failure risk.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
            air = st.number_input(
                "Air Temperature (K)", 250.0, 350.0, 300.0, step=0.5
            )
            process = st.number_input(
                "Process Temperature (K)", 250.0, 400.0, 310.0, step=0.5
            )

        with col2:
            rpm = st.number_input(
                "Rotational Speed (rpm)", 1, 10000, 1500, step=10
            )
            torque = st.number_input(
                "Torque (Nm)", 0.0, 200.0, 40.0, step=0.5
            )
            wear = st.number_input(
                "Tool Wear (minutes)", 0, 1000, 100, step=5
            )

        threshold = st.slider(
            "Inspection alert threshold",
            min_value=0.05,
            max_value=0.90,
            value=0.30,
            step=0.05,
            help="A lower threshold flags more machines for inspection.",
        )

        submitted = st.form_submit_button(
            "Assess Machine Risk", type="primary", use_container_width=True
        )

    if submitted:
        if process < air:
            st.warning(
                "Process temperature is below air temperature. "
                "Please verify the sensor readings."
            )

        try:
            prediction, probability, alert = predict(
                machine_type, air, process, rpm, torque, wear
            )

            if probability >= max(threshold, 0.70):
                risk = "High"
                st.error("HIGH RISK — Inspection and maintenance review recommended.")
            elif probability >= threshold:
                risk = "Attention"
                st.warning("ATTENTION — Consider inspecting this machine.")
            else:
                risk = "Lower"
                st.success("LOWER PREDICTED RISK — Continue routine monitoring.")

            a, b, c = st.columns(3)
            a.metric("Estimated Failure Probability", f"{probability:.1%}")
            b.metric("Predicted Class", "Failure" if prediction else "No Failure")
            c.metric("Risk Category", risk)

            st.progress(
                min(max(float(probability), 0.0), 1.0),
                text="Model-estimated failure probability",
            )

            st.info(
                "Model message: " + str(alert)
            )

            result = pd.DataFrame([{
                "Type": machine_type,
                "Air temperature [K]": air,
                "Process temperature [K]": process,
                "Rotational speed [rpm]": rpm,
                "Torque [Nm]": torque,
                "Tool wear [min]": wear,
                "Predicted failure": int(prediction),
                "Estimated failure probability": round(float(probability), 5),
                "Risk category": risk,
            }])

            st.download_button(
                "Download Prediction Report (CSV)",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="machineguard_prediction.csv",
                mime="text/csv",
            )

        except Exception as exc:
            st.error(
                "Prediction could not be completed. Check that the trained model "
                "exists in the models folder and that its feature columns match."
            )
            st.caption(f"Technical detail: {exc}")

elif page == "Batch Analysis":
    st.header("Batch Machine Analysis")
    st.write(
        "Upload a CSV containing machine readings to assess multiple machines."
    )

    template = pd.DataFrame([
        ["L", 300.0, 310.0, 1500, 40.0, 100],
        ["M", 302.0, 312.0, 1400, 45.0, 200],
    ], columns=FEATURES)

    st.download_button(
        "Download Sample CSV Template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="machineguard_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload machine readings CSV", type=["csv"])
    threshold = st.slider(
        "Flag machines above this probability",
        0.05, 0.90, 0.30, 0.05, key="batch_threshold",
    )

    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
            missing = [col for col in FEATURES if col not in data.columns]

            if missing:
                st.error("Missing columns: " + ", ".join(missing))
            elif len(data) > 10000:
                st.error("Please upload no more than 10,000 rows at a time.")
            else:
                data = data.copy()
                data["Type"] = data["Type"].astype(str).str.strip().str.upper()

                if not data["Type"].isin(["L", "M", "H"]).all():
                    st.error("Type values must be L, M or H.")
                else:
                    predictions = []
                    probabilities = []
                    alerts = []

                    with st.spinner("Assessing machines..."):
                        for row in data[FEATURES].itertuples(index=False, name=None):
                            pred, prob, alert = predict(*row)
                            predictions.append(int(pred))
                            probabilities.append(float(prob))
                            alerts.append(str(alert))

                    data["Predicted failure"] = predictions
                    data["Estimated failure probability"] = probabilities
                    data["Risk category"] = [
                        "High" if p >= max(threshold, 0.70)
                        else "Attention" if p >= threshold
                        else "Lower"
                        for p in probabilities
                    ]
                    data["Model recommendation"] = alerts

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Machines Assessed", len(data))
                    c2.metric(
                        "High Risk",
                        int((data["Risk category"] == "High").sum()),
                    )
                    c3.metric(
                        "Needs Attention",
                        int((data["Risk category"] == "Attention").sum()),
                    )

                    st.dataframe(data, use_container_width=True, hide_index=True)

                    st.subheader("Risk Distribution")
                    counts = data["Risk category"].value_counts()
                    st.bar_chart(counts)

                    st.download_button(
                        "Download All Results (CSV)",
                        data=data.to_csv(index=False).encode("utf-8"),
                        file_name="machineguard_batch_results.csv",
                        mime="text/csv",
                    )

        except Exception as exc:
            st.error(f"Could not process this CSV: {exc}")

elif page == "Model Dashboard":
    st.header("Model Performance Dashboard")
    metrics_path = ROOT / "results" / "classification_model_comparison.csv"

    if metrics_path.exists():
        try:
            metrics = pd.read_csv(metrics_path)
            st.subheader("Classification Model Comparison")
            st.dataframe(metrics, use_container_width=True, hide_index=True)

            score_columns = [
                col for col in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
                if col in metrics.columns
            ]

            if "Model" in metrics.columns and score_columns:
                chart_data = metrics.set_index("Model")[score_columns]
                st.subheader("Performance Metrics")
                st.bar_chart(chart_data)

        except Exception as exc:
            st.warning(f"Could not load model metrics: {exc}")
    else:
        st.info(
            "Model comparison results are not present yet. "
            "Run src/train_models.py to generate them."
        )

    st.caption(
        "Metrics describe performance on the project's evaluation split. "
        "Check precision and recall, especially because machine failures are rare."
    )

elif page == "About Project":
    st.header("About MachineGuard")
    st.write("""
    MachineGuard is an industrial predictive-maintenance project that uses
    machine operating parameters to estimate the likelihood of machine failure.

    **Inputs**
    - Machine type
    - Air and process temperature
    - Rotational speed
    - Torque
    - Tool wear

    **How it works**
    1. The user enters readings or uploads a CSV.
    2. The existing trained machine-learning pipeline processes the readings.
    3. The website displays the predicted class, estimated probability and risk category.
    4. The user can download the results as a CSV report.
    """)

    st.warning("""
    Important: This is a prototype for decision support, not a certified safety
    system. The model's probability is not a guarantee of real-world failure risk.
    Validate it on representative factory data and follow approved safety and
    maintenance procedures.
    """)

st.divider()
st.caption(
    "MachineGuard • Industrial Machine Failure Prediction • "
    "Educational prototype — verify alerts with qualified maintenance personnel."
)
