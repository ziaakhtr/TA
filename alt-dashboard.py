import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import json

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
DATA_PATH = "final_ml_ready.csv"
FEATURE_EXOG = ['is_lebaran_window', 'is_year_end', 'usd_change_rate']
FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "roll_mean_3", "roll_mean_6", "roll_std_3",
    "trend_slope_12", "quarter",
    "is_lebaran_window", "is_year_end", "year_index",
    "fc_error_1", "fc_acc_1", "usd_change_rate"
]

# ── Helpers ─────────────────────────────────────────────────────────────────────
def mape(actual, pred):
    mask = (actual != 0) & (~np.isnan(actual))
    if not mask.any():
        return float('inf')
    return np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100

# ── Loaders ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_csv(path):
    return pd.read_csv(path, parse_dates=["date"])

@st.cache_data
def load_predictions():
    meta_path = 'saved_models/model_meta.json'
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    results = {}
    for prod, info in meta.items():
        safe = info['safe_name']
        results[prod] = {
            'pred_vals': np.load(f'saved_models/{safe}_preds.npy'),
            'actual_vals': np.load(f'saved_models/{safe}_actuals.npy'),
            'baseline_vals': np.load(f'saved_models/{safe}_baseline.npy'),
            'best_model': info['best_model'],
            'type': info['type'],
            'mape': info['mape'],
            'mae': info['mae'],
            'rmse': info['rmse'],
        }
    return results

# ── Landing Page ────────────────────────────────────────────────────────────────
if st.session_state.get('app_page', 'home') == 'home':
    st.markdown("""<style>section[data-testid="stSidebar"]{display:none} header{display:none} button[kind="primary"]{background:linear-gradient(135deg,#f5d17c,#e8b84b)!important;color:#0e1117!important;font-weight:700!important;font-size:1.1rem!important;padding:8px 48px!important;border-radius:40px!important;border:none!important} button[kind="primary"]:hover{transform:scale(1.04);box-shadow:0 0 24px rgba(245,209,124,0.35)}</style>""", unsafe_allow_html=True)
    logo_b64 = __import__("base64").b64encode(open("logo_cmu.png","rb").read()).decode()
    st.markdown(f"""
    <div class="hero-container">
        <img src="data:image/png;base64,{logo_b64}" width="220" style="display:block;margin:0 auto 16px">
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

st.markdown("""<style>section[data-testid="stSidebar"]{display:flex!important} header{display:block!important}</style>""", unsafe_allow_html=True)

# ── Load data ───────────────────────────────────────────────────────────────────
st.markdown("<p style='text-align:center'><img src='data:image/png;base64," + __import__("base64").b64encode(open("logo_cmu.png","rb").read()).decode() + "' width='280' style='display:inline-block'></p>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#f0f4ff;font-size:2.2rem;font-weight:700;margin-top:8px;margin-bottom:0'>Sistem Pendukung Keputusan Produksi</p>", unsafe_allow_html=True)

if not os.path.exists(DATA_PATH):
    st.error(f"Data file not found: `{DATA_PATH}`.")
    st.stop()

df_raw = load_csv(DATA_PATH)
products_sorted = sorted(df_raw["product"].unique())

predictions = load_predictions()
if predictions is None:
    st.error("No saved models found. Run `playground.ipynb` first to train and export models to `saved_models/`.")
    st.stop()

# ── Session state ───────────────────────────────────────────────────────────────
if 'app_page' not in st.session_state:
    st.session_state.app_page = 'home'

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🏠 Beranda", use_container_width=True, type="secondary"):
        st.session_state.app_page = 'home'
        st.rerun()
    selected_product = st.selectbox("Select product", products_sorted)
    st.divider()

    prod_pred = predictions[selected_product]
    st.markdown(f"**Best Model:** {prod_pred['best_model']}")
    st.markdown(f"**Type:** {prod_pred['type']}")
    st.markdown(f"**MAPE (H8):** {prod_pred['mape']:.2f}%" if prod_pred['mape'] != float('inf') else "**MAPE:** N/A")
    st.divider()

    with st.expander("📂 Data Source", expanded=False):
        st.caption(f"`{DATA_PATH}`")
    with st.expander("📐 Exogenous Features (ARIMAX)", expanded=False):
        for f in FEATURE_EXOG:
            st.caption(f"• {f}")
    with st.expander("📊 ML Features (Lasso/RF/XGB/LGBM)", expanded=False):
        for f in FEATURE_COLS:
            st.caption(f"• {f}")

# ── Data subset for chart ───────────────────────────────────────────────────────
df_prod = df_raw[df_raw["product"] == selected_product].copy().sort_values("date").reset_index(drop=True)
df_prod = df_prod.dropna(subset=['actual'] + FEATURE_EXOG)
df_prod = df_prod[df_prod['date'] <= '2026-04-30']

# ── Prediction window constants ─────────────────────────────────────────────────
PRED_DATES = pd.date_range('2025-10-01', periods=3, freq='MS')  # Oct, Nov, Dec 2025
TRAIN_END = pd.Timestamp('2025-09-01')
TRAIN_START = pd.Timestamp('2022-01-01')

# ── Tabs ────────────────────────────────────────────────────────────────────────
tab_forecast, tab_capacity, tab_performance = st.tabs(["📊 Forecast", "🏭 Production Shift Planning", "📈 Model Performance"])

# ── FORECAST TAB ────────────────────────────────────────────────────────────────
with tab_forecast:
    prod_pred = predictions[selected_product]
    pv = prod_pred['pred_vals']
    bv = prod_pred['baseline_vals']

    st.markdown(f"**Horizon H8 — {PRED_DATES[0].strftime('%b %Y')} to {PRED_DATES[-1].strftime('%b %Y')}**")
    st.markdown(f"**Model:** {prod_pred['best_model']}  |  **MAPE:** {prod_pred['mape']:.2f}%")

    cols = st.columns(3)
    for i, dt in enumerate(PRED_DATES):
        with cols[i]:
            pred_val = pv[i]
            target_val = bv[i]
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

    # Chart
    df_chart_train = df_prod[(df_prod['date'] >= TRAIN_START) & (df_prod['date'] <= TRAIN_END)].copy()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_chart_train["date"], y=df_chart_train["actual"],
        name="Actual", mode="lines+markers",
        line=dict(color="#7c9ef5", width=2), marker=dict(size=4)
    ))

    last_actual_date = df_chart_train["date"].iloc[-1]
    last_actual_val = df_chart_train["actual"].iloc[-1]

    base_line_dates = [last_actual_date] + list(PRED_DATES)
    base_line_vals = [last_actual_val] + list(bv)
    fig.add_trace(go.Scatter(
        x=base_line_dates, y=base_line_vals,
        name="Sales Target", mode="lines+markers",
        line=dict(color="#f5cc7c", width=1.5, dash="dot"),
        marker=dict(size=[4] + [6] * 3, symbol=["circle"] + ["diamond"] * 3)
    ))

    model_line_dates = [last_actual_date] + list(PRED_DATES)
    model_line_vals = [last_actual_val] + list(pv)
    fig.add_trace(go.Scatter(
        x=model_line_dates, y=model_line_vals,
        name="Forecast", mode="lines+markers",
        line=dict(color="#f5d17c", width=2.5, dash="dot"),
        marker=dict(size=[4] + [7] * 3, symbol=["circle"] + ["star"] * 3)
    ))

    fig.add_vrect(
        x0=PRED_DATES[0], x1=PRED_DATES[-1],
        fillcolor="rgba(245,209,124,0.06)", line_width=0,
        annotation_text="Prediction Window", annotation_position="top right",
        annotation_font_color="#f5d17c"
    )

    fig.update_layout(
        title=f"H8 Forecast — {selected_product}",
        xaxis_title="Date", yaxis_title="Sales", template="plotly_dark",
        hovermode="x unified", legend=dict(orientation="h", y=1.08),
        height=500, margin=dict(t=60, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

# ── CAPACITY TAB ────────────────────────────────────────────────────────────────
with tab_capacity:
    st.subheader("🏭 Production Capacity Planning")
    st.markdown("Total forecasted volume across **all products** vs. monthly capacity for H8 (Oct–Dec 2025). Each product uses its **best-performing model** from the notebook.")

    cap_data = []
    for pi, dt in enumerate(PRED_DATES):
        total = sum(predictions[p]['pred_vals'][pi] for p in products_sorted)
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
        st.info("No prediction data available.")

# ── PERFORMANCE TAB ─────────────────────────────────────────────────────────────
with tab_performance:
    st.subheader("📈 Best Model Performance per Product (H8: Oct–Dec 2025)")
    st.markdown("Each product uses the single best model identified by walk-forward cross-validation in the notebook.")

    cols = st.columns(2)
    for pi, prod in enumerate(products_sorted):
        with cols[pi % 2]:
            pd_pred = predictions[prod]
            mm = pd_pred['mape']
            st.markdown(f"""
            <div class="perf-card">
                <div class="perf-label">{prod}</div>
                <div class="perf-value" style="color:#7cf5a5;">{mm:.2f}%</div>
                <div class="perf-sub">Model: {pd_pred['best_model']}</div>
                <div class="perf-sub">MAE: {pd_pred['mae']:,.0f} | RMSE: {pd_pred['rmse']:,.0f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    comp_rows = []
    for prod in products_sorted:
        pd_pred = predictions[prod]
        comp_rows.append({
            'Product': prod,
            'Best Model': pd_pred['best_model'],
            'Type': pd_pred['type'],
            'MAPE (%)': f"{pd_pred['mape']:.2f}",
            'MAE': f"{pd_pred['mae']:,.0f}",
            'RMSE': f"{pd_pred['rmse']:,.0f}",
        })
    if comp_rows:
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    with st.expander("📐 Training Configuration"):
        st.markdown(f"""
        - **Training data:** Jan 2022 → Sep 2025
        - **Prediction horizon:** H8 — Oct, Nov, Dec 2025
        - **Products:** {', '.join(products_sorted)}
        - **Best models selected via 11-horizon walk-forward CV in notebook**
        - **Exogenous features (ARIMAX):** {', '.join(FEATURE_EXOG)}
        - **ML features (Lasso/RF/XGB/LGBM):** {', '.join(FEATURE_COLS)}
        """)
