import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Production Capacity Dashboard", page_icon="🏭", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .perf-card {
        background: linear-gradient(135deg, #1e2130, #2a2f45);
        border: 1px solid #3d4466;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .perf-value { font-size: 1.8rem; font-weight: 700; }
    .perf-label { font-size: 0.75rem; color: #8a8fa8; margin-top: 4px; }
    .perf-sub   { font-size: 0.7rem;  color: #5a6080; margin-top: 2px; }
    h1 { color: #e0e4f7 !important; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = r"D:\Zia\final_ml_ready.csv"

@st.cache_data
def load_csv(path):
    return pd.read_csv(path, parse_dates=["date"])

def count_extra_shifts(df_all, target_dates, capacity=1320000):
    """Sums forecast across all SKUs and checks against monthly capacity."""
    df_target = df_all[df_all['date'].isin(target_dates)].copy()
    if not df_target.empty and 'forecast' in df_target.columns:
        monthly_totals = df_target.groupby('date')['forecast'].sum().reset_index()
        monthly_totals.rename(columns={'forecast': 'total_forecast'}, inplace=True)
    else:
        monthly_totals = pd.DataFrame({'date': target_dates, 'total_forecast': 0})
        
    monthly_totals['capacity'] = capacity
    monthly_totals['extra_shift_needed'] = monthly_totals['total_forecast'] > capacity
    return monthly_totals

st.title("🏭 Main Dashboard: Production Capacity Planning")

if not os.path.exists(DATA_PATH):
    st.error(f"Data file not found: `{DATA_PATH}`.")
    st.stop()

df_raw = load_csv(DATA_PATH)

with st.sidebar:
    st.title("⚙️ Settings")
    capacity_limit = st.number_input("Monthly Capacity Threshold", min_value=0, value=1320000, step=10000)
    num_months = st.slider("Months to Predict", min_value=1, max_value=12, value=4)

st.markdown("Calculates the total forecasted volume across **all products** to determine if an extra production shift is needed.")

last_date = pd.to_datetime('2026-03-31')
future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, num_months + 1)]

capacity_df = count_extra_shifts(df_raw, future_dates, capacity=capacity_limit)

if not capacity_df.empty:
    cols = st.columns(len(capacity_df))
    for idx, row in capacity_df.iterrows():
        with cols[idx]:
            needed = "Yes ⚠️" if row['extra_shift_needed'] else "No ✅"
            color = "#f5687c" if row['extra_shift_needed'] else "#7cf5a5"
            st.markdown(f"""
            <div class="perf-card">
                <div class="perf-label">{row['date'].strftime('%B %Y')}</div>
                <div class="perf-value" style="color:{color}">{needed}</div>
                <div class="perf-sub">Total Forecast: {row['total_forecast']:,.0f}</div>
                <div class="perf-sub">Capacity: {row['capacity']:,.0f}</div>
            </div>""", unsafe_allow_html=True)