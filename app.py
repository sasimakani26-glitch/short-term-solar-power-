import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import config
import json
import datetime


from config import fetch_live_weather
from modules.preprocessing import split_and_scale
from modules.preprocessing import validate_data
from modules.solar_engine import solar_system_engine
from modules.analysis import full_analysis

# ==============================================================
# PAGE CONFIG
# ==============================================================
st.set_page_config(
    page_title="Ultra Short Term Solar Power Prediction System at 15 minutes Resolution",
    layout="wide"
)

# ==============================================================
# UI
# ==============================================================
st.markdown("""
<style>
.big-title {font-size:36px;font-weight:bold;color:#FDB813;}
.prediction-box {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    padding:25px;border-radius:12px;
    text-align:center;font-size:30px;
    font-weight:bold;color:white;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">Ultra Short Term Solar Power Prediction at 15-Minute Resolution</div>', unsafe_allow_html=True)
st.divider()

# ==============================================================
# LOAD MODEL
# ==============================================================
@st.cache_resource
def load_model():
    return joblib.load(config.MODEL_PATH)

model = load_model()

# Load feature columns
feat_path = getattr(config, "FEATURE_COLUMNS_PATH", "models/feature_columns.json")
try:
    with open(feat_path, "r") as fh:
        feature_cols = json.load(fh)
except:
    feature_cols = ["Temperature","Humidity","Cloud_Cover","Wind_Speed","GHI","Hour","DayOfWeek","Month"]

# ==============================================================
# SIDEBAR
# ==============================================================
LIVE_MODE = "Live Prediction"
COMPARE_MODE = "Model Comparison"
ACTUAL_MODE = "Actual vs Predicted"
FORECAST_MODE = "Forecast (15-60 min)"

mode = st.sidebar.radio("Select Mode", [LIVE_MODE, COMPARE_MODE, ACTUAL_MODE, FORECAST_MODE])

# ==============================================================
# 1. LIVE PREDICTION (UPDATED)
# ==============================================================
if mode == LIVE_MODE:
    st.subheader("Live Prediction")

    previous_power = st.session_state.get("previous_power", None)

    if st.button("Fetch Weather & Predict"):

        weather = fetch_live_weather() or {}
        now = datetime.datetime.now()

        ghi = now.minute  # still weak, but keeping your logic

        row = {
            "Temperature": weather.get("Temperature", 0),
            "Humidity": weather.get("Humidity", 0),
            "Cloud_Cover": weather.get("Cloud_Cover", 0),
            "Wind_Speed": weather.get("Wind_Speed", 0),
            "Hour": now.hour,
            "DayOfWeek": now.weekday(),
            "Month": now.month,
            "GHI": ghi
        }

        for col in feature_cols:
            if col.startswith("Power_lag_"):
                row[col] = 0.0

        input_df = pd.DataFrame([row]).reindex(columns=feature_cols, fill_value=0)
        prediction = model.predict(input_df)[0]

        # ⚠️ Fake actual (since real-time actual not available)
        actual = prediction * (1 - (row["Cloud_Cover"] / 200))

        # 🔥 UPDATED ENGINE CALL
        result = solar_system_engine(prediction, actual, previous_power, row)

        alerts = result["alerts"]
        metrics = result["metrics"]
        energy = result["energy"]

        st.session_state["previous_power"] = prediction

        # ---------------- WEATHER ----------------
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Temp (°C)", round(row["Temperature"], 2))
        c2.metric("Humidity (%)", round(row["Humidity"], 2))
        c3.metric("Cloud (%)", round(row["Cloud_Cover"], 2))
        c4.metric("Wind (m/s)", round(row["Wind_Speed"], 2))
        c5.metric("GHI", round(row["GHI"], 2))

        st.divider()

        # ---------------- PREDICTION ----------------
        st.markdown(f'<div class="prediction-box">{round(prediction,2)} kW</div>', unsafe_allow_html=True)

        # ---------------- METRICS ----------------
        st.subheader("Model Performance (Live)")

        m1, m2, m3 = st.columns(3)
        m1.metric("Error", metrics["Error"])
        m2.metric("Error (%)", metrics["Error_%"])
        m3.metric("Confidence (%)", metrics["Confidence_%"])

        st.divider()

        # ---------------- ALERTS ----------------
        st.subheader("System Alerts")

        for alert in alerts:
            if "High" in alert or "Error" in alert or "Drop" in alert:
                st.error(alert)
            elif "Cloud" in alert:
                st.warning(alert)
            else:
                st.success(alert)

        st.divider()

        # ---------------- ENERGY ----------------
        st.subheader("Energy & Cloud Impact")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Sunlight (W/m²)", energy["Sunlight_Wm2"])
        e2.metric("Panel Output (W)", energy["Panel_Output_W"])
        e3.metric("Cloud Impact (%)", energy["Cloud_Impact_%"])
        e4.metric("Efficiency", energy.get("Efficiency_Used", 0))
        
# ==============================================================
# 2. MODEL COMPARISON
# ==============================================================
elif mode == COMPARE_MODE:
    st.subheader("Model Performance")

    try:
        performance = pd.read_csv(config.PERFORMANCE_PATH)
        st.dataframe(performance)

        fig = px.bar(performance, x="Model", y="RMSE")
        st.plotly_chart(fig)

        best = performance.loc[performance["RMSE"].idxmin()]
        st.success(f"Best Model: {best['Model']}")

    except Exception as e:
        st.warning(f"Run training first: {e}")
       

# ==============================================================
# 3. ACTUAL VS PREDICTED
# ==============================================================
elif mode == ACTUAL_MODE:
    st.subheader("Actual vs Predicted")

    try:
        df = pd.read_csv("solar dataset.csv")
        df = validate_data(df)

        X_train, X_test, y_train, y_test = split_and_scale(df)
        y_pred = model.predict(X_test)

        comp = pd.DataFrame({
            "Actual": y_test,
            "Predicted": y_pred
        }).reset_index(drop=True)

        # Step 1: Reduce noise (sampling)
        comp_sampled = comp.iloc[::10]  

        # Step 2: Smooth curves
        comp_sampled["Actual"] = comp_sampled["Actual"].rolling(window=5).mean()
        comp_sampled["Predicted"] = comp_sampled["Predicted"].rolling(window=5).mean()

        st.dataframe(comp_sampled.head(50))

        # Step 3: Clean Plot
        fig = px.line(
            comp_sampled,
            y=["Actual", "Predicted"],
            title="Actual vs Predicted"
        )

        fig.update_layout(
            xaxis_title="Time Step",
            yaxis_title="Power Output",
            legend_title="Legend"
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(e)
# ==============================================================
# 4. FORECAST (15,30,45,60 MIN)
# ==============================================================
elif mode == FORECAST_MODE:
    st.subheader("Multi-Resolution Forecast")

    if st.button("Generate Forecast"):

        now = datetime.datetime.now()
        weather = fetch_live_weather() or {}

        base_ghi = now.hour * 50  # ⚠️ fake

        steps = [15, 30, 45, 60]
        forecast_data = []

        for step in steps:
            future = now + datetime.timedelta(minutes=step)

            row = {
                "Temperature": weather.get("Temperature", 30),
                "Humidity": weather.get("Humidity", 60),
                "Cloud_Cover": weather.get("Cloud_Cover", 20),
                "Wind_Speed": weather.get("Wind_Speed", 5),
                "Hour": future.hour,
                "DayOfWeek": future.weekday(),
                "Month": future.month,
                "GHI": base_ghi
            }

            for col in feature_cols:
                if col.startswith("Power_lag_"):
                    row[col] = 0.0

            input_df = pd.DataFrame([row]).reindex(columns=feature_cols, fill_value=0)
            pred = model.predict(input_df)[0]

            forecast_data.append({
                "Time": future.strftime("%H:%M"),
                "Resolution": f"{step} min",
                "Power (kW)": round(pred,2)
            })

        forecast_df = pd.DataFrame(forecast_data)

        st.dataframe(forecast_df, use_container_width=True)

        fig = px.line(
            forecast_df,
            x="Time",
            y="Power (kW)",
            markers=True,
            title="Forecast (15–60 mins)"
        )
        st.plotly_chart(fig) 