import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import pmdarima as pm
from statsmodels.tsa.statespace.varmax import VARMAX

st.set_page_config(page_title="Statistical Forecast Dashboard", page_icon="📈", layout="wide")

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

# ── Fixed paths — edit here ───────────────────────────────────────────────────
DATA_PATH   = r"D:\Zia\final_ml_ready.csv"
FEATURE_EXOG = ['is_lebaran_window', 'is_year_end', 'fc_error_1', 'fc_acc_1', 'usd_change_rate']

# ── Helpers ───────────────────────────────────────────────────────────────────
def mape(actual, pred):
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100

def mae(actual, pred):
    return np.mean(np.abs(actual - pred))

def rmse(actual, pred):
    return np.sqrt(np.mean((actual - pred) ** 2))

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

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_csv(path):
    return pd.read_csv(path, parse_dates=["date"])

@st.cache_resource(show_spinner="Training ARIMAX & VARX on Train Set...")
def train_and_evaluate_models(train_df, test_df, exog_cols):
    train_df = train_df.sort_values("date").set_index("date")
    test_df = test_df.sort_values("date").set_index("date")
    
    y_train = train_df['actual']
    y_test = test_df['actual']
    X_train_exog = train_df[exog_cols]
    X_test_exog = test_df[exog_cols]
    
    results = {}
    
    # 1. ARIMAX
    try:
        model_arimax = pm.auto_arima(
            y_train, X=X_train_exog, seasonal=True, m=12,
            suppress_warnings=True, stepwise=True
        )
        preds_arimax = model_arimax.predict(n_periods=len(y_test), X=X_test_exog)
        mape_arimax = mape(y_test.values, preds_arimax.values)
        results['ARIMAX'] = {'preds': preds_arimax.values, 'mape': mape_arimax}
    except Exception as e:
        results['ARIMAX'] = {'preds': None, 'mape': float('inf'), 'error': str(e)}

    # 2. VARX
    try:
        var_endog = ['actual', 'fc_error_1']
        var_exog = [c for c in exog_cols if c != 'fc_error_1']
        
        train_var = train_df[var_endog]
        exog_train_var = train_df[var_exog]
        exog_test_var = test_df[var_exog]
        
        model_varx = VARMAX(train_var, exog=exog_train_var, order=(1, 0), enforce_stationarity=False)
        varx_res = model_varx.fit(disp=False)
        forecast_varx = varx_res.predict(start=len(train_df), end=len(train_df)+len(test_df)-1, exog=exog_test_var)
        preds_varx = forecast_varx['actual']
        mape_varx = mape(y_test.values, preds_varx.values)
        results['VARX'] = {'preds': preds_varx.values, 'mape': mape_varx}
    except Exception as e:
        results['VARX'] = {'preds': None, 'mape': float('inf'), 'error': str(e)}

    return results

@st.cache_resource(show_spinner="Refitting Best Model on Full Data for 3-Month Forecast...")
def forecast_future(full_df, future_exog_df, exog_cols, best_model_type):
    full_df = full_df.sort_values("date").set_index("date")
    
    if best_model_type == 'ARIMAX':
        full_y = full_df['actual']
        full_X_exog = full_df[exog_cols]
        model = pm.auto_arima(full_y, X=full_X_exog, seasonal=True, m=12, suppress_warnings=True)
        forecast_values = model.predict(n_periods=len(future_exog_df), X=future_exog_df[exog_cols])
        return forecast_values.values
    else:
        var_endog = ['actual', 'fc_error_1']
        var_exog = [c for c in exog_cols if c != 'fc_error_1']
        endog = full_df[var_endog]
        exog = full_df[var_exog]
        model_var = VARMAX(endog, exog=exog, order=(1, 0), enforce_stationarity=False)
        res_var = model_var.fit(disp=False)
        forecast_res = res_var.predict(start=len(full_df), end=len(full_df)+len(future_exog_df)-1, exog=future_exog_df[var_exog])
        return forecast_res['actual'].values

# ── Load data ─────────────────────────────────────────────────────────────────
st.title("📈 Statistical Sales Forecast Dashboard")

if not os.path.exists(DATA_PATH):
    st.error(f"Data file not found: `{DATA_PATH}`.")
    st.stop()

df_raw = load_csv(DATA_PATH)
products = sorted(df_raw["product"].unique())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")
    selected_product = st.selectbox("Select product", products)
    st.divider()
    st.caption(f"✅ Data Loaded: `{DATA_PATH}`")
    st.caption("Exogenous Variables Used:")
    for f in FEATURE_EXOG:
        st.caption(f"• {f}")

df = df_raw[df_raw["product"] == selected_product].copy().sort_values("date").reset_index(drop=True)
df = df.dropna(subset=['actual'] + FEATURE_EXOG)

# Filter to use newest data only until March 2026 so the model learns up to March
df = df[df['date'] <= '2026-03-31']

df['split'] = '' # Clear existing split and enforce object/string dtype
df.loc[(df['date'] > '2021-12-01') & (df['date'] < '2025-03-01'), 'split'] = 'train'
df.loc[(df['date'] >= '2025-03-01') & (df['date'] <= '2025-12-01'), 'split'] = 'test'

df_train = df[df["split"] == "train"].copy()
df_test = df[df["split"] == "test"].copy()

if df_train.empty or df_test.empty:
    st.warning("Insufficient train or test data for this product.")
    st.stop()

# 1. Train and Evaluate on Test Set
results = train_and_evaluate_models(df_train, df_test, FEATURE_EXOG)

m_arimax = results['ARIMAX']['mape']
m_varx = results['VARX']['mape']

best_model_type = 'ARIMAX' if m_arimax <= m_varx else 'VARX'
best_preds = results[best_model_type]['preds']

st.markdown(f"### Best Model Selected: **{best_model_type}**")
st.write(f"Test MAPE Comparison - ARIMAX: {m_arimax:.2f}% | VARX: {m_varx:.2f}%")

df_test["model_forecast"] = best_preds
df.loc[df["split"] == "test", "model_forecast"] = best_preds

# 2. Performance Cards
def perf_card(col, label, model_val, baseline_val, unit="", lower_better=True):
    better = (model_val < baseline_val) if lower_better else (model_val > baseline_val)
    color  = "#7cf5a5" if better else "#f5687c"
    delta  = model_val - baseline_val
    sign   = "▼" if delta < 0 else "▲"
    col.markdown(f"""
    <div class="perf-card">
        <div class="perf-value" style="color:{color}">{model_val:.2f}{unit}</div>
        <div class="perf-label">{label} — Model (test set)</div>
        <div class="perf-sub">Baseline: {baseline_val:.2f}{unit} &nbsp;|&nbsp; {sign} {abs(delta):.2f}{unit}</div>
    </div>""", unsafe_allow_html=True)

act = df_test["actual"].values
pred = df_test["model_forecast"].values
base = df_test["forecast"].values

m_mape = mape(act, pred)
b_mape = mape(act, base)
m_mae = mae(act, pred)
b_mae = mae(act, base)
m_rmse = rmse(act, pred)
b_rmse = rmse(act, base)

c1, c2, c3 = st.columns(3)
perf_card(c1, "MAPE", m_mape, b_mape, unit="%")
perf_card(c2, "MAE", m_mae, b_mae)
perf_card(c3, "RMSE", m_rmse, b_rmse)
st.markdown("<br>", unsafe_allow_html=True)

if "show_4m" not in st.session_state:
    st.session_state["show_4m"] = False

if st.button("Show 4th Month Prediction" if not st.session_state["show_4m"] else "Hide 4th Month Prediction"):
    st.session_state["show_4m"] = not st.session_state["show_4m"]

num_months = 4 if st.session_state["show_4m"] else 3

# 3. Predict Future
last_date = df["date"].max()
future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, num_months + 1)]
future_rows = []

df_raw_prod = df_raw[df_raw["product"] == selected_product]

# Define specific usd_change_rate floats for the future months
future_usd_rates = [0.0066, -0.0012, 0.0137, 0.0186]  # Replace the 4th value with your specific rate

for i, dt in enumerate(future_dates):
    raw_match = df_raw_prod[df_raw_prod["date"] == dt]
    new_row = {
        "date": dt,
        "split": "future",
        "quarter": dt.quarter,
        "is_lebaran_window": 0, # adjust logic as needed
        "is_year_end": 1 if dt.month == 12 else 0,
        "fc_acc_1": df["fc_acc_1"].iloc[-1] if "fc_acc_1" in df.columns else 0.8,
        "fc_error_1": df["fc_error_1"].iloc[-1] if "fc_error_1" in df.columns else 0.2,
        "usd_change_rate": future_usd_rates[i] if i < len(future_usd_rates) else 0.0,
    }
    
    # Pull actual exogenous variables from the data if they exist
    if not raw_match.empty:
        for col in FEATURE_EXOG:
            if col in raw_match.columns and not pd.isna(raw_match[col].iloc[0]):
                new_row[col] = raw_match[col].iloc[0]
                
    future_rows.append(new_row)

future_exog_df = pd.DataFrame(future_rows).set_index("date")

# Hardcode the specific start date for future forecast training here
df_future_train = df[(df['date'] >= '2022-01-01') & (df['date'] <= '2025-12-01')].copy()

future_preds = forecast_future(df_future_train, future_exog_df, FEATURE_EXOG, best_model_type)

for i, row in enumerate(future_rows):
    dt = row["date"]
    raw_match = df_raw_prod[df_raw_prod["date"] == dt]
    
    row["model_forecast"] = future_preds[i]
    
    # Import baseline forecast from raw data if present
    if not raw_match.empty and "forecast" in raw_match.columns and not pd.isna(raw_match["forecast"].iloc[0]):
        row["forecast"] = raw_match["forecast"].iloc[0]
    else:
        row["forecast"] = future_preds[i]  # fallback visualization
        
    # Import actual from raw data if present
    if not raw_match.empty and "actual" in raw_match.columns and not pd.isna(raw_match["actual"].iloc[0]):
        row["actual"] = raw_match["actual"].iloc[0]
    else:
        row["actual"] = np.nan

df_future = pd.DataFrame(future_rows)
df_full_plot = pd.concat([df, df_future], ignore_index=True)

# 4. Main Chart
fig = go.Figure()

# Actual
fig.add_trace(go.Scatter(x=df_full_plot["date"], y=df_full_plot["actual"],
    name="Actual", mode="lines+markers",
    line=dict(color="#7c9ef5", width=2), marker=dict(size=4)))

# Baseline forecast
fig.add_trace(go.Scatter(x=df_full_plot["date"], y=df_full_plot["forecast"],
    name="Baseline Forecast", mode="lines",
    line=dict(color="#f5a97c", width=1.5, dash="dot")))

# Model Forecast on Test
df_test_plot = df_full_plot[df_full_plot["split"] == "test"]
fig.add_trace(go.Scatter(
    x=df_test_plot["date"],
    y=df_test_plot["model_forecast"],
    name=f"Test Forecast ({best_model_type})", mode="lines+markers",
    line=dict(color="#7cf5a5", width=2.5), marker=dict(size=5)))

# Model Forecast Future
df_future_plot = df_full_plot[df_full_plot["split"] == "future"]
last_known = df_full_plot[df_full_plot["split"] != "future"].iloc[[-1]].copy()
last_known["model_forecast"] = last_known["actual"]
plot_future = pd.concat([last_known, df_future_plot])

fig.add_trace(go.Scatter(
    x=plot_future["date"],
    y=plot_future["model_forecast"],
    name=f"Future Forecast ({num_months}M)", mode="lines+markers",
    line=dict(color="#f5d17c", width=2.5, dash="dot"), marker=dict(size=7, symbol="star")))

fig.add_vrect(x0=df_test["date"].min(), x1=df_test["date"].max(),
    fillcolor="rgba(124,158,245,0.06)", line_width=0,
    annotation_text="Test set", annotation_position="top left",
    annotation_font_color="#8a8fa8")

fig.add_vrect(x0=df_future_plot["date"].min(), x1=df_future_plot["date"].max(),
    fillcolor="rgba(245,209,124,0.06)", line_width=0,
    annotation_text=f"Future {num_months}M", annotation_position="top right",
    annotation_font_color="#f5d17c")

fig.update_layout(
    title=f"Forecast vs Actual — {selected_product}",
    xaxis_title="Date", yaxis_title="Sales", template="plotly_dark",
    hovermode="x unified", legend=dict(orientation="h", y=1.08),
    height=440, margin=dict(t=60, b=40))

st.plotly_chart(fig, use_container_width=True)

# 5. Raw Data Table
with st.expander("📋 Future Forecast Table"):
    st.dataframe(df_future[["date", "model_forecast"]].style.format({"model_forecast": "{:,.0f}"}), use_container_width=True)

with st.expander("📋 Raw Data Table"):
    show_cols = ["date", "split", "actual", "forecast", "model_forecast"] + FEATURE_EXOG
    st.dataframe(df_full_plot[show_cols].style.format(na_rep="-"), use_container_width=True)

# 6. Production Capacity Check
st.subheader("🏭 Production Capacity Planning (All SKUs)")
st.markdown("Calculates the total forecasted volume across **all products** to determine if an extra production shift is needed (Capacity: 1,320,000 / month).")

capacity_df = count_extra_shifts(df_raw, future_dates, capacity=1320000*(5/60))

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