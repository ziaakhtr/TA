import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import pmdarima as pm
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from prophet import Prophet
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="Sistem Pendukung Keputusan Produksi", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .perf-card {
        background: linear-gradient(135deg, #1e2130, #2a2f45);
        border: 1px solid #3d4466;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        text-align: center;
    }
    .perf-value { font-size: 1.8rem; font-weight: 700; }
    .perf-label { font-size: 0.75rem; color: #8a8fa8; margin-top: 4px; }
    .perf-sub   { font-size: 0.7rem;  color: #5a6080; margin-top: 2px; }
    h1 { color: #e0e4f7 !important; text-align: center !important; }

    .fc-card {
        background-color: #131626;
        border-radius: 14px;
        padding: 12px 18px;
        border: 1px solid #2a2d45;
        border-left: 5px solid #f5d17c;
        display: flex;
        flex-direction: column;
        gap: 1px;
    }
    .fc-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2px;
    }
    .fc-card-month {
        font-size: 0.9rem;
        color: #f0f4ff;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .fc-card-status { font-size: 1.1rem; }
    .fc-card-forecast-label {
        font-size: 0.65rem;
        color: #5a5f7a;
        margin-top: 1px;
    }
    .fc-card-forecast-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f0f4ff;
        line-height: 1.1;
        margin: 0 0 2px;
    }
    .fc-card-divider {
        border: none;
        border-top: 1px solid #23263d;
        margin: 3px 0;
    }
    .fc-card-target-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .fc-card-target-label {
        font-size: 0.7rem;
        color: #6a6f8a;
    }
    .fc-card-target-value {
        font-size: 1rem;
        font-weight: 600;
        color: #c8cce8;
    }
    .fc-card-variance {
        font-size: 0.85rem;
        font-weight: 700;
        padding: 2px 10px;
        border-radius: 6px;
    }
    .fc-card-variance.above { background: #1a3328; color: #7cf5a5; }
    .fc-card-variance.below { background: #3d3a1a; color: #f5d17c; }

    .fc-mape-card {
        background-color: #131626;
        border-radius: 14px;
        padding: 20px 22px;
        border: 1px solid #2a2d45;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .fc-mape-label {
        font-size: 0.7rem;
        color: #6a6f8a;
    }
    .fc-mape-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f0f4ff;
    }

    .hero-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 80vh;
        text-align: center;
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        color: #f0f4ff;
        line-height: 1.2;
        margin-bottom: 12px;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #8a8fa8;
        font-weight: 400;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 40px;
    }
    .hero-footer {
        font-size: 0.75rem;
        color: #3d4466;
        margin-top: 60px;
    }
    .hero-btn {
        background: linear-gradient(135deg, #f5d17c, #e8b84b) !important;
        color: #0e1117 !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 8px 48px !important;
        border-radius: 40px !important;
        border: none !important;
        transition: transform 0.15s, box-shadow 0.15s !important;
    }
    .hero-btn:hover {
        transform: scale(1.04) !important;
        box-shadow: 0 0 24px rgba(245,209,124,0.35) !important;
    }

    button[kind="primary"] {
        background: linear-gradient(135deg, #f5d17c, #e8b84b) !important;
        color: #0e1117 !important;
        font-weight: 700 !important;
        border: none !important;
    }
    button[kind="primary"]:hover {
        box-shadow: 0 0 20px rgba(245,209,124,0.3) !important;
    }
    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #f5d17c, #e8b84b) !important;
    }
    div[data-testid="stSelectbox"]:focus-within {
        border-color: #f5d17c !important;
        box-shadow: 0 0 0 1px #f5d17c !important;
    }
    div[data-testid="stTabs"] div[role="tablist"] {
        display: flex;
        justify-content: center;
        width: 500px;
        margin: 0 auto;
    }
    div[data-testid="stTabs"] button {
        flex: 1;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Fixed paths ─────────────────────────────────────────────────────────────────
DATA_PATH   = r"D:\Zia\final_ml_ready.csv"
FEATURE_EXOG = ['is_lebaran_window', 'is_year_end', 'fc_error_1', 'fc_acc_1', 'usd_change_rate']
FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "roll_mean_3", "roll_mean_6", "roll_std_3",
    "trend_slope_12", "quarter",
    "is_lebaran_window", "is_year_end", "year_index",
    "fc_error_1", "fc_acc_1", "usd_change_rate"
]
MODELS = ["ARIMAX", "Lasso", "Random Forest", "XGBoost", "LightGBM", "Prophet"]

# ── Helpers ─────────────────────────────────────────────────────────────────────
def mape(actual, pred):
    mask = (actual != 0) & (~np.isnan(actual))
    if not mask.any():
        return float('inf')
    return np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100

def mae(actual, pred):
    return np.mean(np.abs(actual - pred))

def mse(actual, pred):
    mask = ~np.isnan(actual) & ~np.isnan(pred)
    if not mask.any():
        return float('inf')
    return np.mean((actual[mask] - pred[mask]) ** 2)

def rmse(actual, pred):
    return np.sqrt(mse(actual, pred))

def build_features(actual_series, forecast_series, known_row):
    n = len(actual_series)
    def _safe(v):
        return v if np.isfinite(v) else 0.0

    f = {}
    f['lag_1'] = _safe(actual_series[-1]) if n >= 1 else 0.0
    f['lag_2'] = _safe(actual_series[-2]) if n >= 2 else 0.0
    f['lag_3'] = _safe(actual_series[-3]) if n >= 3 else 0.0
    f['lag_6'] = _safe(actual_series[-6]) if n >= 6 else 0.0
    f['lag_12'] = _safe(actual_series[-12]) if n >= 12 else 0.0
    f['roll_mean_3'] = _safe(np.mean(actual_series[-3:])) if n >= 3 else 0.0
    f['roll_mean_6'] = _safe(np.mean(actual_series[-6:])) if n >= 6 else 0.0
    f['roll_std_3'] = _safe(np.std(actual_series[-3:])) if n >= 3 else 0.0
    f['trend_slope_12'] = _safe(np.polyfit(np.arange(12), actual_series[-12:], 1)[0]) if n >= 12 else 0.0

    if n >= 1:
        a, fc = actual_series[-1], forecast_series[-1]
        fc_err = (a - _safe(fc)) / a if a != 0 else 0.0
    else:
        fc_err = 0.0
    f['fc_error_1'] = fc_err
    f['fc_acc_1'] = 1 - abs(fc_err)

    for col in ['quarter', 'is_lebaran_window', 'is_year_end', 'year_index', 'usd_change_rate']:
        f[col] = _safe(known_row[col])
    return pd.Series(f)

def count_extra_shifts(df_all, target_dates, capacity=1320000):
    df_target = df_all[df_all['date'].isin(target_dates)].copy()
    if not df_target.empty and 'forecast' in df_target.columns:
        monthly_totals = df_target.groupby('date')['forecast'].sum().reset_index()
        monthly_totals.rename(columns={'forecast': 'total_forecast'}, inplace=True)
    else:
        monthly_totals = pd.DataFrame({'date': target_dates, 'total_forecast': 0})
    monthly_totals['capacity'] = capacity
    monthly_totals['extra_shift_needed'] = monthly_totals['total_forecast'] > capacity
    return monthly_totals

# ── Loaders ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_csv(path):
    return pd.read_csv(path, parse_dates=["date"])

# ── Simulation step ─────────────────────────────────────────────────────────────
def run_simulation_step(product, step_idx):
    df_all = load_csv(DATA_PATH)
    df_p = df_all[df_all['product'] == product].copy().sort_values("date").reset_index(drop=True)
    df_p = df_p[df_p['date'] >= '2022-01-01']

    train_end = pd.Timestamp('2025-02-01') + pd.DateOffset(months=step_idx - 1)
    train_df = df_p[(df_p['date'] >= '2022-01-01') & (df_p['date'] <= train_end)]

    pred_dates = [train_end + pd.DateOffset(months=i) for i in range(1, 4)]

    pred_rows = []
    for dt in pred_dates:
        raw_match = df_p[df_p['date'] == dt]
        row = {
            'date': dt, 'quarter': dt.quarter,
            'is_lebaran_window': 0,
            'is_year_end': 1 if dt.month == 12 else 0,
            'fc_acc_1': train_df['fc_acc_1'].iloc[-1],
            'fc_error_1': train_df['fc_error_1'].iloc[-1],
            'usd_change_rate': 0.0,
        }
        if not raw_match.empty:
            for col in FEATURE_EXOG + FEATURE_COLS:
                if col in raw_match.columns and not pd.isna(raw_match[col].iloc[0]):
                    row[col] = raw_match[col].iloc[0]
        pred_rows.append(row)

    pred_exog_df = pd.DataFrame(pred_rows).set_index('date')

    actual_vals = []
    baseline_vals = []
    for i, dt in enumerate(pred_dates):
        match = df_p[df_p['date'] == dt]
        if not match.empty:
            actual_vals.append(match['actual'].iloc[0])
            bl = match['forecast'].iloc[0] if 'forecast' in match.columns and not pd.isna(match['forecast'].iloc[0]) else 0
            baseline_vals.append(bl)
        else:
            actual_vals.append(np.nan)
            baseline_vals.append(np.nan)

    actual_arr = np.array(actual_vals)
    base_arr = np.array(baseline_vals)
    valid = (~np.isnan(actual_arr)) & (actual_arr != 0)

    y_train = train_df['actual'].values
    models_out = {}

    # ── ARIMAX ──
    pred_vals_arimax = np.full(3, np.nan)
    try:
        train_ari = train_df[np.isfinite(train_df['actual'])]
        if len(train_ari) >= 12:
            y_ari = train_ari['actual']
            X_ari = train_ari[FEATURE_EXOG]
            try:
                model_arimax = pm.auto_arima(
                    y_ari, X=X_ari, seasonal=True, m=12,
                    suppress_warnings=True, stepwise=True
                )
            except Exception:
                model_arimax = pm.auto_arima(
                    y_ari, X=X_ari, seasonal=False,
                    suppress_warnings=True, stepwise=True
                )
            pred_vals_arimax = model_arimax.predict(
                n_periods=3, X=pred_exog_df[FEATURE_EXOG]
            ).values
    except Exception:
        pass

    pred_arr = np.array(pred_vals_arimax)
    ape_arr = np.full(3, np.nan)
    for i in range(3):
        if valid[i]:
            ape_arr[i] = abs((actual_arr[i] - pred_arr[i]) / actual_arr[i]) * 100

    valid_any = valid.any()
    models_out['ARIMAX'] = {
        'pred_vals': pred_arr,
        'mape': mape(actual_arr[valid], pred_arr[valid]) if valid_any else float('inf'),
        'mae': mae(actual_arr[valid], pred_arr[valid]) if valid_any else float('inf'),
        'mse': mse(actual_arr[valid], pred_arr[valid]) if valid_any else float('inf'),
        'rmse': rmse(actual_arr[valid], pred_arr[valid]) if valid_any else float('inf'),
        'ape': ape_arr,
    }

    # ── ML models: Lasso, RF, XGBoost, LightGBM ──
    ml_configs = [
        ('Random Forest', RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1)),
        ('XGBoost', XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0)),
        ('Lasso', make_pipeline(StandardScaler(), Lasso(alpha=0.01, random_state=42, max_iter=10000))),
        ('LightGBM', LGBMRegressor(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42, verbose=-1)),
    ]

    train_feat_ml = train_df[FEATURE_COLS].dropna()
    if len(train_feat_ml) >= 10:
        y_train_ml = train_df.loc[train_feat_ml.index, 'actual'].values
        eff_forecasts_init = list(train_df.loc[train_feat_ml.index, 'forecast'].values)
        for ml_name, ml_model in ml_configs:
            try:
                X_train_ml = train_feat_ml.values
                ml_model.fit(X_train_ml, y_train_ml)

                eff_actuals = list(y_train_ml)
                eff_forecasts = list(eff_forecasts_init)
                ml_preds = []
                for mi in range(3):
                    raw_match = df_p[df_p['date'] == pred_dates[mi]]
                    if not raw_match.empty:
                        known_row = raw_match.iloc[0]
                    else:
                        known_row = pd.Series({
                            'quarter': pred_dates[mi].quarter,
                            'is_lebaran_window': 0,
                            'is_year_end': 1 if pred_dates[mi].month == 12 else 0,
                            'year_index': pred_dates[mi].year - 2022,
                            'usd_change_rate': 0.0,
                        })
                    feat = build_features(eff_actuals, eff_forecasts, known_row)
                    X = feat[FEATURE_COLS].values.reshape(1, -1)
                    X = np.nan_to_num(X, nan=0.0)
                    pred = ml_model.predict(X)[0]
                    ml_preds.append(pred)
                    eff_actuals.append(pred)
                    fc_val = known_row.get('forecast', np.nan)
                    eff_forecasts.append(fc_val if np.isfinite(fc_val) else pred)

                ml_pred_arr = np.array(ml_preds)
                ml_ape = np.full(3, np.nan)
                for i in range(3):
                    if valid[i]:
                        ml_ape[i] = abs((actual_arr[i] - ml_pred_arr[i]) / actual_arr[i]) * 100
                models_out[ml_name] = {
                    'pred_vals': ml_pred_arr,
                    'mape': mape(actual_arr[valid], ml_pred_arr[valid]) if valid_any else float('inf'),
                    'mae': mae(actual_arr[valid], ml_pred_arr[valid]) if valid_any else float('inf'),
                    'mse': mse(actual_arr[valid], ml_pred_arr[valid]) if valid_any else float('inf'),
                    'rmse': rmse(actual_arr[valid], ml_pred_arr[valid]) if valid_any else float('inf'),
                    'ape': ml_ape,
                }
            except Exception:
                models_out[ml_name] = {
                    'pred_vals': np.full(3, np.nan),
                    'mape': float('inf'),
                    'mae': float('inf'),
                    'mse': float('inf'),
                    'rmse': float('inf'),
                    'ape': np.full(3, np.nan),
                }
    else:
        for ml_name, _ in ml_configs:
            models_out[ml_name] = {
                'pred_vals': np.full(3, np.nan),
                'mape': float('inf'),
                'mae': float('inf'),
                'mse': float('inf'),
                'rmse': float('inf'),
                'ape': np.full(3, np.nan),
            }

    # ── Prophet ──
    try:
        prophet_df = train_df[['date', 'actual'] + FEATURE_COLS].rename(columns={'date': 'ds', 'actual': 'y'})
        prophet_df = prophet_df.dropna(subset=FEATURE_COLS)
        if len(prophet_df) >= 10:
            model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            for col in FEATURE_COLS:
                model_prophet.add_regressor(col)
            model_prophet.fit(prophet_df)

            eff_actuals = list(prophet_df['y'].values)
            eff_forecasts = list(train_df.loc[prophet_df.index, 'forecast'].values)
            prophet_preds = []
            for mi in range(3):
                raw_match = df_p[df_p['date'] == pred_dates[mi]]
                if not raw_match.empty:
                    known_row = raw_match.iloc[0]
                else:
                    known_row = pd.Series({
                        'quarter': pred_dates[mi].quarter,
                        'is_lebaran_window': 0,
                        'is_year_end': 1 if pred_dates[mi].month == 12 else 0,
                        'year_index': pred_dates[mi].year - 2022,
                        'usd_change_rate': 0.0,
                    })
                feat = build_features(eff_actuals, eff_forecasts, known_row)
                future = pd.DataFrame({'ds': [pred_dates[mi]]})
                for col in FEATURE_COLS:
                    future[col] = feat[col]
                forecast = model_prophet.predict(future)
                pred = forecast['yhat'].values[0]
                prophet_preds.append(pred)
                eff_actuals.append(pred)
                fc_val = known_row.get('forecast', np.nan)
                eff_forecasts.append(fc_val if np.isfinite(fc_val) else pred)

            pred_vals_prophet = np.array(prophet_preds)
            prophet_ape = np.full(3, np.nan)
            for i in range(3):
                if valid[i]:
                    prophet_ape[i] = abs((actual_arr[i] - pred_vals_prophet[i]) / actual_arr[i]) * 100
            models_out['Prophet'] = {
                'pred_vals': pred_vals_prophet,
                'mape': mape(actual_arr[valid], pred_vals_prophet[valid]) if valid_any else float('inf'),
                'mae': mae(actual_arr[valid], pred_vals_prophet[valid]) if valid_any else float('inf'),
                'mse': mse(actual_arr[valid], pred_vals_prophet[valid]) if valid_any else float('inf'),
                'rmse': rmse(actual_arr[valid], pred_vals_prophet[valid]) if valid_any else float('inf'),
                'ape': prophet_ape,
            }
        else:
            raise ValueError(f"Not enough Prophet training data ({len(prophet_df)} rows)")
    except Exception:
        models_out['Prophet'] = {
            'pred_vals': np.full(3, np.nan),
            'mape': float('inf'),
            'mae': float('inf'),
            'mse': float('inf'),
            'rmse': float('inf'),
            'ape': np.full(3, np.nan),
        }

    return {
        'actual_vals': actual_arr,
        'baseline_vals': base_arr,
        'valid_mask': valid,
        'models': models_out,
    }


# ── Landing Page ────────────────────────────────────────────────────────────────
if st.session_state.get('app_page', 'home') == 'home':
    st.markdown("""<style>section[data-testid="stSidebar"]{display:none} header{display:none} button[kind="primary"]{background:linear-gradient(135deg,#f5d17c,#e8b84b)!important;color:#0e1117!important;font-weight:700!important;font-size:1.1rem!important;padding:8px 48px!important;border-radius:40px!important;border:none!important} button[kind="primary"]:hover{transform:scale(1.04);box-shadow:0 0 24px rgba(245,209,124,0.35)}</style>""", unsafe_allow_html=True)
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Sistem Pendukung<br>Keputusan Produksi</div>
        <div class="hero-subtitle">PT CIPTA MORTAR UTAMA</div>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Login", use_container_width=True, type="primary"):
            st.session_state.app_page = 'dashboard'
            st.rerun()
    st.markdown("""<div class="hero-footer">© 2025 — All Rights Reserved</div>""", unsafe_allow_html=True)
    st.stop()

# Undo landing page CSS that may persist in browser
st.markdown("""<style>section[data-testid="stSidebar"]{display:flex!important} header{display:block!important}</style>""", unsafe_allow_html=True)

# ── Load data ───────────────────────────────────────────────────────────────────
st.title("Demand Forecast Dashboard")

if not os.path.exists(DATA_PATH):
    st.error(f"Data file not found: `{DATA_PATH}`.")
    st.stop()

df_raw = load_csv(DATA_PATH)
products = sorted(df_raw["product"].unique())

# ── Session state: simulation (shared across all products) ─────────────────────
if 'app_page' not in st.session_state:
    st.session_state.app_page = 'home'
if 'sim_step' not in st.session_state:
    st.session_state.sim_step = 0
    st.session_state.sim_history = []
if 'model_sel' not in st.session_state:
    st.session_state.model_sel = MODELS[0]

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🏠 Beranda", use_container_width=True, type="secondary"):
        st.session_state.app_page = 'home'
        st.rerun()
    selected_product = st.selectbox("Select product", products)
    st.divider()

    # Simulation controls
    if st.session_state.sim_step == 0:
        st.markdown("### Simulation ready")
        st.caption("Press Run to start")
    else:
        st.markdown(f"### Step {st.session_state.sim_step} / 11")
        st.caption(f"Train: Jan 2022 → {pd.Timestamp('2025-02-01') + pd.DateOffset(months=st.session_state.sim_step - 1):%b %Y}")

    if st.session_state.sim_step == 0:
        run_label = "▶️ Start Simulation"
    elif st.session_state.sim_step < 11:
        run_label = "▶️ Run Next Step"
    else:
        run_label = "✅ Complete"

    if st.button(run_label, use_container_width=True, disabled=st.session_state.sim_step >= 11):
        new_step = st.session_state.sim_step + 1
        train_end = pd.Timestamp('2025-02-01') + pd.DateOffset(months=new_step - 1)
        pred_dates = [train_end + pd.DateOffset(months=i) for i in range(1, 4)]

        all_prods = sorted(df_raw["product"].unique())
        step_products = {}
        progress = st.progress(0, text=f"Training all models for step {new_step}/11...")
        for idx, prod in enumerate(all_prods):
            progress.progress((idx + 1) / len(all_prods), text=f"Training {prod} (6 models)...")
            step_products[prod] = run_simulation_step(prod, new_step)
        progress.empty()

        step_data = {
            'train_end': train_end,
            'pred_dates': pred_dates,
            'products': step_products,
        }
        st.session_state.sim_step = new_step
        st.session_state.sim_history.append(step_data)
        st.rerun()

    if st.session_state.sim_step > 0:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.sim_step = 0
            st.session_state.sim_history = []
            st.rerun()

    # Run All button
    if st.session_state.sim_step < 11:
        if st.button("▶️▶️ Run All 11 Steps", use_container_width=True, type="primary"):
            all_prods = sorted(df_raw["product"].unique())
            n_prods = len(all_prods)
            steps_remaining = 11 - st.session_state.sim_step
            total_tasks = steps_remaining * n_prods
            current_task = 0
            progress = st.progress(0, text="Running all steps...")
            for si in range(st.session_state.sim_step + 1, 12):
                train_end = pd.Timestamp('2025-02-01') + pd.DateOffset(months=si - 1)
                pred_dates = [train_end + pd.DateOffset(months=i) for i in range(1, 4)]
                step_products = {}
                for pi, prod in enumerate(all_prods):
                    current_task += 1
                    progress.progress(current_task / total_tasks, text=f"Step {si}/11 — {prod} (6 models)...")
                    step_products[prod] = run_simulation_step(prod, si)
                step_data = {
                    'train_end': train_end,
                    'pred_dates': pred_dates,
                    'products': step_products,
                }
                st.session_state.sim_history.append(step_data)
            st.session_state.sim_step = 11
            progress.empty()
            st.rerun()

    st.divider()
    st.selectbox("Show model forecast:", MODELS, key="model_sel")

    st.divider()
    with st.expander("📂 Data Source", expanded=False):
        st.caption(f"`{DATA_PATH}`")
    with st.expander("📐 Exogenous Features (ARIMAX)", expanded=False):
        for f in FEATURE_EXOG:
            st.caption(f"• {f}")
    with st.expander("📊 ML Features (Lasso/RF/XGB/LGBM)", expanded=False):
        for f in FEATURE_COLS:
            st.caption(f"• {f}")

df_prod = df_raw[df_raw["product"] == selected_product].copy().sort_values("date").reset_index(drop=True)
df_prod = df_prod.dropna(subset=['actual'] + FEATURE_EXOG)
df_prod = df_prod[df_prod['date'] <= '2026-03-31']

def get_best_model(product):
    if st.session_state.sim_step < 2:
        return 'ARIMAX'
    model_mapes = {m: [] for m in MODELS}
    for i in range(st.session_state.sim_step - 1):
        hp = st.session_state.sim_history[i]['products'][product]
        for m in MODELS:
            md = hp['models'].get(m)
            if md is not None:
                for mi in range(3):
                    if hp['valid_mask'][mi] and np.isfinite(md['ape'][mi]):
                        model_mapes[m].append(md['ape'][mi])
    best = 'ARIMAX'
    best_mean = float('inf')
    for m, apes in model_mapes.items():
        if len(apes) >= 1:
            mean_ape = np.mean(apes)
            if mean_ape < best_mean:
                best_mean = mean_ape
                best = m
    return best

# ── Tabs ────────────────────────────────────────────────────────────────────────

# ── Tabs ────────────────────────────────────────────────────────────────────────
tab_forecast, tab_capacity, tab_performance = st.tabs(["📊 Forecast", "🏭 Capacity Planning", "📈 Model Performance"])

# ── FORECAST TAB ────────────────────────────────────────────────────────────────
with tab_forecast:
    selected_model = st.session_state.model_sel
    if selected_model not in MODELS:
        selected_model = MODELS[0]

    # Metrics bar (current step + resolved accuracy)
    if st.session_state.sim_step > 0 and st.session_state.sim_history:
        step_data = st.session_state.sim_history[-1]
        prod_data = step_data['products'][selected_product]
        model_data = prod_data['models'].get(selected_model)
        if model_data is None:
            model_data = prod_data['models'].get('ARIMAX', next(iter(prod_data['models'].values())))
        step = st.session_state.sim_step

        # Resolved accuracy from previous steps' first-month predictions
        if step > 1:
            res_preds = []
            res_actuals = []
            for i in range(step - 1):
                hp = st.session_state.sim_history[i]['products'][selected_product]
                res_preds.append(hp['models'].get(selected_model))
                if res_preds[-1] is None:
                    res_preds[-1] = hp['models'].get('ARIMAX', next(iter(hp['models'].values())))
                res_preds[-1] = res_preds[-1]['pred_vals'][0]
                res_actuals.append(hp['actual_vals'][0])
            res_valid = (~np.isnan(res_actuals)) & (np.array(res_actuals) != 0)
            if res_valid.any():
                resolved_mape = mape(np.array(res_actuals)[res_valid], np.array(res_preds)[res_valid])
            else:
                resolved_mape = float('inf')
            resolved_label = f"One-Step MAPE ({step - 1}mo)"
        else:
            resolved_mape = None
            resolved_label = "Resolved MAPE"

        m_cols = st.columns([1] * 3 + [1])
        for i, dt in enumerate(step_data['pred_dates']):
            with m_cols[i]:
                pred_val = model_data['pred_vals'][i]
                target_val = prod_data['baseline_vals'][i]
                target_str = f"{target_val:,.0f}" if not np.isnan(target_val) else "—"
                diff = pred_val - target_val if not (np.isnan(target_val) or np.isnan(pred_val)) else 0
                diff_pct = (diff / target_val * 100) if target_val != 0 and not np.isnan(target_val) and not np.isnan(pred_val) else 0
                above = diff >= 0
                status = "▲" if above else "▼"
                vclass = "above" if above else "below"
                vtext = f"+{diff_pct:.1f}%" if above else f"{diff_pct:.1f}%"
                st.markdown(f"""
                <div class="fc-card">
                    <div class="fc-card-header">
                        <span class="fc-card-month">{dt.strftime('%B %Y')}</span>
                        <span class="fc-card-status">{status}</span>
                    </div>
                    <div class="fc-card-forecast-label">Forecast</div>
                    <div class="fc-card-forecast-value">{pred_val:,.0f}</div>
                    <hr class="fc-card-divider">
                    <div class="fc-card-target-row">
                        <div>
                            <div class="fc-card-target-label">Sales Target</div>
                            <div class="fc-card-target-value">{target_str}</div>
                        </div>
                        <span class="fc-card-variance {vclass}">{vtext}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        with m_cols[3]:
            if resolved_mape is not None and resolved_mape != float('inf'):
                st.markdown(f"""
                <div class="fc-mape-card">
                    <div class="fc-mape-label">{resolved_label}</div>
                    <div class="fc-mape-value">{resolved_mape:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="fc-mape-card">
                    <div class="fc-mape-label">{resolved_label}</div>
                    <div class="fc-mape-value">—</div>
                </div>
                """, unsafe_allow_html=True)

    # Main Chart
    train_start = pd.Timestamp('2022-01-01')
    if st.session_state.sim_step > 0 and st.session_state.sim_history:
        step_data = st.session_state.sim_history[-1]
        prod_data = step_data['products'][selected_product]
        model_data = prod_data['models'].get(selected_model)
        if model_data is None:
            model_data = prod_data['models'].get('ARIMAX', next(iter(prod_data['models'].values())))
        train_end = step_data['train_end']
        pred_dates = step_data['pred_dates']
        pred_vals = model_data['pred_vals']
        base_vals = prod_data['baseline_vals']
    else:
        train_end = pd.Timestamp('2025-02-01')
        pred_dates = []
        pred_vals = np.array([])
        base_vals = np.array([])

    df_chart_train = df_prod[(df_prod['date'] >= train_start) & (df_prod['date'] <= train_end)].copy()

    fig = go.Figure()

    # Actuals (training only)
    fig.add_trace(go.Scatter(
        x=df_chart_train["date"], y=df_chart_train["actual"],
        name="Actual", mode="lines+markers",
        line=dict(color="#7c9ef5", width=2), marker=dict(size=4)
    ))

    # Faded past model forecasts (all previous steps)
    if st.session_state.sim_history:
        past_steps = st.session_state.sim_history[:-1] if len(pred_dates) > 0 else st.session_state.sim_history
        n_past = len(past_steps)
        for i, h in enumerate(past_steps):
            prod_h = h['products'][selected_product]
            h_last_actual = df_prod[df_prod['date'] == h['train_end']]['actual']
            if h_last_actual.empty:
                continue
            h_dates = [h['train_end']] + list(h['pred_dates'])
            past_md = prod_h['models'].get(selected_model)
            if past_md is None:
                past_md = prod_h['models'].get('ARIMAX', next(iter(prod_h['models'].values())))
            h_vals = [h_last_actual.iloc[0]] + list(past_md['pred_vals'])
            opacity = 0.15 + (i / max(1, n_past - 1)) * 0.3 if n_past > 1 else 0.25
            fig.add_trace(go.Scatter(
                x=h_dates, y=h_vals,
                name="Past Forecasts" if i == 0 else None,
                mode="lines+markers",
                line=dict(color=f"rgba(245,209,124,{opacity})", width=1.5, dash="dot"),
                marker=dict(size=3, color=f"rgba(245,209,124,{opacity})"),
                showlegend=i == 0,
                hoverinfo="skip"
            ))

    if len(pred_dates) > 0:
        last_actual_date = df_chart_train["date"].iloc[-1]
        last_actual_val = df_chart_train["actual"].iloc[-1]

        # Sales Target line (connect from last actual)
        base_line_dates = [last_actual_date] + list(pred_dates)
        base_line_vals = [last_actual_val] + list(base_vals)
        fig.add_trace(go.Scatter(
            x=base_line_dates, y=base_line_vals,
            name="Sales Target", mode="lines+markers",
            line=dict(color="#f5cc7c", width=1.5, dash="dot"),
            marker=dict(size=[4] + [6] * len(pred_dates), symbol=["circle"] + ["diamond"] * len(pred_dates))
        ))

        # Current model forecast (connect from last actual)
        model_line_dates = [last_actual_date] + list(pred_dates)
        model_line_vals = [last_actual_val] + list(pred_vals)

        fig.add_trace(go.Scatter(
            x=model_line_dates, y=model_line_vals,
            name="Current Forecast", mode="lines+markers",
            line=dict(color="#f5d17c", width=2.5, dash="dot"),
            marker=dict(size=[4] + [7] * len(pred_dates), symbol=["circle"] + ["star"] * len(pred_dates))
        ))

        # Shading
        fig.add_vrect(
            x0=pred_dates[0], x1=pred_dates[-1],
            fillcolor="rgba(245,209,124,0.06)", line_width=0,
            annotation_text="Prediction Window", annotation_position="top right",
            annotation_font_color="#f5d17c"
        )

    fig.update_layout(
        title=f"Walk-Forward Simulation — {selected_product}",
        xaxis_title="Date", yaxis_title="Sales", template="plotly_dark",
        hovermode="x unified", legend=dict(orientation="h", y=1.08),
        height=500, margin=dict(t=60, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    # History table (collapsed)
    if st.session_state.sim_history:
        with st.expander("📋 Simulation History", expanded=False):
            rows = []
            for i, h in enumerate(st.session_state.sim_history):
                pred_range = f"{h['pred_dates'][0]:%b %Y} – {h['pred_dates'][-1]:%b %Y}"
                ph = h['products'][selected_product]
                model_ph = ph['models'].get(selected_model)
                if model_ph is None:
                    model_ph = ph['models'].get('ARIMAX', next(iter(ph['models'].values())))
                fcast_strs = [f"{v:,.0f}" for v in model_ph['pred_vals']]
                base_strs = [f"{v:,.0f}" if not np.isnan(v) else "?" for v in ph['baseline_vals']]
                rows.append({
                    'Step': i + 1,
                    'Train End': h['train_end'].strftime('%b %Y'),
                    'Predicted': pred_range,
                    f'Forecast ({selected_model})': ' | '.join(fcast_strs),
                    'Sales Target': ' | '.join(base_strs),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── CAPACITY TAB ────────────────────────────────────────────────────────────────
with tab_capacity:
    st.subheader("🏭 Production Capacity Planning")
    st.markdown("Total forecasted volume across **all products** vs. monthly capacity. Each product uses its **best-performing model** (lowest resolved APE so far).")

    if st.session_state.sim_step > 0 and st.session_state.sim_history:
        cur = st.session_state.sim_history[-1]
        cap_dates = cur['pred_dates']
        cap_data = []
        for pi, dt in enumerate(cap_dates):
            total = 0
            for prod_key, pd_data in cur['products'].items():
                bm = get_best_model(prod_key)
                md = pd_data['models'].get(bm, pd_data['models']['ARIMAX'])
                total += md['pred_vals'][pi]
            cap_data.append({'date': dt, 'total_forecast': total})
        capacity_df = pd.DataFrame(cap_data)
        capacity_df['capacity'] = 1320000 * (5 / 60)
        shift_cap = 22000
        capacity_df['excess'] = (capacity_df['total_forecast'] - capacity_df['capacity']).clip(lower=0)
        capacity_df['extra_shifts'] = np.ceil(capacity_df['excess'].fillna(0) / shift_cap).astype(int)

        if not capacity_df.empty:
            cols = st.columns(len(capacity_df))
            for idx, row in capacity_df.iterrows():
                with cols[idx]:
                    num_shifts = row['extra_shifts']
                    needed = f"{num_shifts} extra shift{'s' if num_shifts > 1 else ''} ⚠️" if num_shifts > 0 else "Within capacity ✅"
                    color = "#f5d17c" if num_shifts > 0 else "#7cf5a5"
                    st.markdown(f"""
                    <div class="perf-card">
                        <div class="perf-label">{row['date'].strftime('%B %Y')}</div>
                        <div class="perf-value" style="color:{color}">{needed}</div>
                        <div class="perf-sub">Total Forecast: {row['total_forecast']:,.0f}</div>
                        <div class="perf-sub">Capacity: {row['capacity']:,.0f}</div>
                    </div>""", unsafe_allow_html=True)

        with st.expander("📋 Capacity Details"):
            st.dataframe(capacity_df.style.format({
                "total_forecast": "{:,.0f}",
                "capacity": "{:,.0f}",
                "excess": "{:,.0f}",
                "extra_shifts": "{:,d}",
            }), use_container_width=True, hide_index=True)
    else:
        st.info("▶️ Run the simulation to see capacity planning for the predicted months.")

# ── PERFORMANCE TAB ─────────────────────────────────────────────────────────────
with tab_performance:
    if st.session_state.sim_step > 1 and st.session_state.sim_history:
        all_models_data = {m: {'preds': [], 'actuals': [], 'baselines': []} for m in MODELS}
        for i in range(st.session_state.sim_step):
            hp = st.session_state.sim_history[i]['products'][selected_product]
            for mi in range(3):
                if hp['valid_mask'][mi]:
                    for m in MODELS:
                        md = hp['models'].get(m)
                        if md is not None:
                            all_models_data[m]['preds'].append(md['pred_vals'][mi])
                            all_models_data[m]['actuals'].append(hp['actual_vals'][mi])
                            all_models_data[m]['baselines'].append(hp['baseline_vals'][mi])

        resolved_count = len(all_models_data['ARIMAX']['preds'])
        if resolved_count > 0:
            st.subheader(f"📈 Walk-Forward Recursive Accuracy ({resolved_count} months resolved)")

            n_cols = min(3, len(MODELS))
            model_cols = st.columns(n_cols)
            best_model_name = min(MODELS, key=lambda m: mape(
                np.array(all_models_data[m]['actuals']),
                np.array(all_models_data[m]['preds'])
            ) if len(all_models_data[m]['preds']) > 0 else float('inf'))

            for mi, m in enumerate(MODELS):
                with model_cols[mi % n_cols]:
                    act = np.array(all_models_data[m]['actuals'])
                    pred = np.array(all_models_data[m]['preds'])
                    base = np.array(all_models_data[m]['baselines'])
                    if len(pred) > 0:
                        mm = mape(act, pred)
                        bm = mape(act, base)
                        better = mm < bm
                        color = "#7cf5a5" if better else "#f5d17c"
                        star = "⭐ " if m == best_model_name else ""
                        st.markdown(f"""
                        <div class="perf-card" style="border-color: {'#f5d17c' if m == best_model_name else '#3d4466'};">
                            <div class="perf-label">{star}{m}</div>
                            <div class="perf-value" style="color:{color}">{mm:.2f}%</div>
                            <div class="perf-sub">Sales Target: {bm:.2f}% {'▼' if better else '▲'}</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="perf-card">
                            <div class="perf-label">{m}</div>
                            <div class="perf-value" style="color:#5a6080;">—</div>
                            <div class="perf-sub">No predictions</div>
                        </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Model Comparison")

            comp_rows = []
            for m in MODELS:
                act = np.array(all_models_data[m]['actuals'])
                pred = np.array(all_models_data[m]['preds'])
                base = np.array(all_models_data[m]['baselines'])
                if len(pred) > 0:
                    mm = mape(act, pred)
                    bm = mape(act, base)
                    ma = mae(act, pred)
                    ms = mse(act, pred)
                    mr = rmse(act, pred)
                    winner = "✅" if mm < bm else "⚠️"
                    comp_rows.append({
                        'Model': m,
                        'MAPE (%)': f"{mm:.2f}",
                        'MAE': f"{ma:,.0f}",
                        'MSE': f"{ms:,.0f}",
                        'RMSE': f"{mr:,.0f}",
                        'vs Sales Target': winner,
                    })
            if comp_rows:
                comp_df = pd.DataFrame(comp_rows)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Detail per model (collapsed)
            with st.expander("📋 Simulation Forecast Detail", expanded=False):
                detail_tabs = st.tabs(MODELS)
                for mi, m in enumerate(MODELS):
                    with detail_tabs[mi]:
                        d_rows = []
                        for si, step_entry in enumerate(st.session_state.sim_history):
                            hp = step_entry['products'][selected_product]
                            md = hp['models'].get(m)
                            if md is None:
                                continue
                            for pmi in range(3):
                                is_resolved = hp['valid_mask'][pmi]
                                ape_val = md['ape'][pmi] if is_resolved and not np.isnan(md['ape'][pmi]) else None
                                d_rows.append({
                                    'Step': si + 1,
                                    'Month': step_entry['pred_dates'][pmi],
                                    'Forecast': f"{md['pred_vals'][pmi]:,.0f}",
                                    'Actual': f"{hp['actual_vals'][pmi]:,.0f}" if is_resolved else "—",
                                    'Sales Target': f"{hp['baseline_vals'][pmi]:,.0f}" if not np.isnan(hp['baseline_vals'][pmi]) else "—",
                                    'APE (%)': f"{ape_val:.1f}" if ape_val is not None else "—",
                                    'Resolved': "✅" if is_resolved else "—",
                                })
                        if d_rows:
                            st.dataframe(pd.DataFrame(d_rows), use_container_width=True, hide_index=True)
                        else:
                            st.caption("No data for this model yet.")

        else:
            st.info("No months resolved yet. Complete a second step to see walk-forward accuracy.")

    elif st.session_state.sim_step == 1:
        # Only one step done — show current step's predictions
        hp = st.session_state.sim_history[-1]['products'][selected_product]
        st.subheader("📈 After Step 1 — Current Predictions")
        st.info("Complete a second step to see walk-forward accuracy comparisons.")
        r1 = []
        for m in MODELS:
            md = hp['models'].get(m)
            if md is not None:
                r1.append({
                    'Model': m,
                    'Month 1': f"{md['pred_vals'][0]:,.0f}",
                    'Month 2': f"{md['pred_vals'][1]:,.0f}",
                    'Month 3': f"{md['pred_vals'][2]:,.0f}",
                })
        if r1:
            st.dataframe(pd.DataFrame(r1), use_container_width=True, hide_index=True)
    else:
        st.info("▶️ Run the simulation to see model performance comparison.")

    # Config expander
    with st.expander("📐 Training Configuration"):
        steps_done = st.session_state.sim_step
        if steps_done > 0:
            cur_end = st.session_state.sim_history[-1]['train_end'].strftime('%b %Y')
            st.markdown(f"""
            - **Simulation steps completed:** {steps_done} / 11
            - **Current train window:** Jan 2022 → {cur_end}
            - **Products simulated:** {len(st.session_state.sim_history[-1]['products'])}
            - **Models:** {', '.join(MODELS)}
            - **Exogenous features (ARIMAX/Prophet):** {', '.join(FEATURE_EXOG)}
            - **ML features (Lasso/RF/XGB/LGBM):** {', '.join(FEATURE_COLS)}
            """)
        else:
            st.markdown(f"""
            - **Training data:** Jan 2022 → Feb 2025
            - **Models:** {', '.join(MODELS)}
            - **Exogenous features (ARIMAX/Prophet):** {', '.join(FEATURE_EXOG)}
            - **ML features (Lasso/RF/XGB/LGBM):** {', '.join(FEATURE_COLS)}
            - **Run the simulation to see walk-forward performance**
            """)