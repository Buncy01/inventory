import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from scipy.stats import norm

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Inventory Optimization System", layout="wide")
st.title("📦 AI-Based Inventory Optimization System")

st.markdown(
    """
    This application forecasts product demand using **Auto-ARIMA**
    and calculates **EOQ, Safety Stock, and Reorder Point**.
    """
)

# --------------------------------------------------
# FILE UPLOAD UI
# --------------------------------------------------
st.subheader("📤 Upload Inventory Data")

uploaded_file = st.file_uploader(
    "Upload Excel or CSV file (must contain Date & Demand columns)",
    type=["xlsx", "csv"]
)

# --------------------------------------------------
# INVENTORY FORMULAS
# --------------------------------------------------
def calculate_eoq(annual_demand, ordering_cost, holding_cost):
    return np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)

def calculate_safety_stock(ts, lead_time, service_level):
    z = norm.ppf(service_level)
    sigma = ts.std()
    return z * sigma * np.sqrt(lead_time)

def calculate_reorder_point(ts, lead_time, safety_stock):
    avg_daily_demand = ts.mean()
    return avg_daily_demand * lead_time + safety_stock

# --------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------
if uploaded_file:

    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    date_col = st.sidebar.selectbox("Select Date Column", df.columns)
    demand_col = st.sidebar.selectbox("Select Demand Column", df.columns)

    forecast_horizon = st.sidebar.number_input(
        "Forecast Horizon (days)", 30, 180, 90
    )
    lead_time = st.sidebar.number_input(
        "Lead Time (days)", 1, 60, 7
    )
    ordering_cost = st.sidebar.number_input(
        "Ordering Cost (₹)", value=500
    )
    holding_cost = st.sidebar.number_input(
        "Holding Cost (₹ per unit/year)", value=50
    )
    service_level = st.sidebar.slider(
        "Service Level", 0.80, 0.99, 0.95
    )

    # Validate columns
    if date_col == demand_col:
        st.error("Date and Demand columns must be different.")
        st.stop()

    # --------------------------------------------------
    # TIME SERIES PREPARATION
    # --------------------------------------------------
    df[date_col] = pd.to_datetime(df[date_col])
    ts = df.set_index(date_col)[demand_col]
    ts = ts.asfreq("D").fillna(method="ffill")

    # Weekly aggregation for stable forecasting
    ts_weekly = ts.resample("W").sum()

    # --------------------------------------------------
    # AUTO-ARIMA FORECAST
    # --------------------------------------------------
    with st.spinner("🔮 Forecasting demand using Auto-ARIMA..."):
        model = auto_arima(
            ts_weekly,
            seasonal=True,
            m=52,
            stepwise=True,
            suppress_warnings=True
        )
        forecast_weeks = max(1, forecast_horizon // 7)
        forecast = model.predict(n_periods=forecast_weeks)

    forecast_index = pd.date_range(
        start=ts_weekly.index[-1] + pd.Timedelta(weeks=1),
        periods=len(forecast),
        freq="W"
    )

    forecast_series = pd.Series(forecast, index=forecast_index)

    # --------------------------------------------------
    # INVENTORY CALCULATIONS
    # --------------------------------------------------
    annual_demand = forecast_series.sum() * (52 / len(forecast_series))

    eoq = calculate_eoq(annual_demand, ordering_cost, holding_cost)
    safety_stock = calculate_safety_stock(ts, lead_time, service_level)
    reorder_point = calculate_reorder_point(ts, lead_time, safety_stock)

    # --------------------------------------------------
    # KPI DISPLAY
    # --------------------------------------------------
    st.subheader("📊 Inventory KPIs")

    k1, k2, k3 = st.columns(3)
    k1.metric("EOQ (Units)", f"{eoq:.0f}")
    k2.metric("Safety Stock (Units)", f"{safety_stock:.0f}")
    k3.metric("Reorder Point (Units)", f"{reorder_point:.0f}")

    # --------------------------------------------------
    # FORECAST PLOT
    # --------------------------------------------------
    st.subheader("📈 Demand Forecast")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts[-180:], label="Historical Demand")
    ax.plot(forecast_series, label="Forecast", color="orange")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units")
    ax.legend()
    st.pyplot(fig)

    # --------------------------------------------------
    # DOWNLOAD RESULTS
    # --------------------------------------------------
    result_df = pd.DataFrame({
        "Date": forecast_series.index,
        "Forecasted_Demand": forecast_series.values
    })

    st.download_button(
        "📥 Download Forecast Results",
        result_df.to_csv(index=False),
        file_name="inventory_forecast_results.csv",
        mime="text/csv"
    )

else:
    st.warning("⬆️ Please upload an Excel or CSV file to continue.")


