import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Patient Vitals Dashboard",
    page_icon="💓",
    layout="wide"
)

st.title("💓 Patient Vital Signs Dashboard")
st.caption("Smart Healthcare Analytics | Demo Dashboard")

st.sidebar.header("Patient Details")
patient_name = st.sidebar.text_input("Patient Name", "Demo Patient")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=35)

st.subheader(f"Vitals Overview — {patient_name}")
st.write(f"Patient age: {age}")

col1, col2, col3, col4 = st.columns(4)

with col1:
    heart_rate = st.number_input("Heart Rate (BPM)", min_value=30, max_value=220, value=72)
with col2:
    systolic = st.number_input("Systolic Blood Pressure (mmHg)", min_value=50, max_value=250, value=120)
with col3:
    spo2 = st.number_input("SpO2 (%)", min_value=50, max_value=100, value=98)
with col4:
    temperature = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=36.8, step=0.1)

st.divider()
st.subheader("Vital Signs Summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Heart Rate", f"{heart_rate} BPM")
c2.metric("Systolic BP", f"{systolic} mmHg")
c3.metric("Oxygen Saturation", f"{spo2}%")
c4.metric("Temperature", f"{temperature:.1f} °C")

st.subheader("Sample Vital Trends")
data = pd.DataFrame({
    "Time": range(1, 8),
    "Heart Rate (BPM)": [70, 72, 75, 73, 76, 72, heart_rate],
    "SpO2 (%)": [98, 97, 98, 99, 97, 98, spo2],
})

chart = px.line(
    data,
    x="Time",
    y=["Heart Rate (BPM)", "SpO2 (%)"],
    markers=True,
    title="Illustrative Vital Signs Trends"
)
st.plotly_chart(chart, use_container_width=True)

st.info(
    "Demo data only. Values and trends are illustrative, not real patient readings. "
    "This dashboard is not intended for medical diagnosis."
)
