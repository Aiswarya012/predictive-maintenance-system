import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.features.engineering import engineer_features


@st.cache_resource
def load_model():
    model_artifact = joblib.load(Path("models/model_latest.joblib"))
    scaler = joblib.load(Path("models/preprocessor_latest.joblib"))
    return model_artifact["model"], model_artifact["feature_names"], scaler


def prepare_input(
    air_temp: float,
    process_temp: float,
    rpm: float,
    torque: float,
    tool_wear: float,
    machine_type: str,
    feature_names: list[str],
) -> pd.DataFrame:
    data = {
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "temp_diff": process_temp - air_temp,
        "power": torque * rpm * (2 * 3.14159 / 60),
        "wear_torque_interaction": tool_wear * torque,
        "Type_L": 1 if machine_type == "L" else 0,
        "Type_M": 1 if machine_type == "M" else 0,
    }
    df = pd.DataFrame([data])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]


def main() -> None:
    st.set_page_config(
        page_title="Predictive Maintenance",
        page_icon="⚙️",
        layout="centered",
    )

    st.title("⚙️ Predictive Maintenance System")
    st.markdown("Predict machine failure from sensor readings using XGBoost.")

    model, feature_names, scaler = load_model()

    st.sidebar.header("Sensor Inputs")

    air_temp = st.sidebar.slider(
        "Air Temperature (K)", min_value=290.0, max_value=310.0, value=298.1, step=0.1
    )
    process_temp = st.sidebar.slider(
        "Process Temperature (K)", min_value=300.0, max_value=320.0, value=308.6, step=0.1
    )
    rpm = st.sidebar.slider(
        "Rotational Speed (RPM)", min_value=1000, max_value=3000, value=1551, step=10
    )
    torque = st.sidebar.slider(
        "Torque (Nm)", min_value=3.0, max_value=80.0, value=42.8, step=0.1
    )
    tool_wear = st.sidebar.slider(
        "Tool Wear (min)", min_value=0, max_value=250, value=0, step=1
    )
    machine_type = st.sidebar.selectbox("Machine Type", options=["L", "M", "H"], index=1)

    if st.sidebar.button("Predict", type="primary", use_container_width=True):
        df = prepare_input(air_temp, process_temp, float(rpm), torque, float(tool_wear), machine_type, feature_names)
        scaled = scaler.transform(df)
        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error("🔴 FAILURE PREDICTED")
            else:
                st.success("🟢 MACHINE HEALTHY")

        with col2:
            st.metric("Failure Probability", f"{probability:.2%}")

        st.divider()

        st.subheader("Input Summary")
        input_data = {
            "Air Temperature (K)": air_temp,
            "Process Temperature (K)": process_temp,
            "Rotational Speed (RPM)": rpm,
            "Torque (Nm)": torque,
            "Tool Wear (min)": tool_wear,
            "Machine Type": machine_type,
        }
        st.dataframe(pd.DataFrame([input_data]), use_container_width=True, hide_index=True)

        st.subheader("Engineered Features")
        eng_data = {
            "Temp Difference (K)": process_temp - air_temp,
            "Power (W)": round(torque * rpm * (2 * 3.14159 / 60), 2),
            "Wear × Torque": round(tool_wear * torque, 2),
        }
        st.dataframe(pd.DataFrame([eng_data]), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
