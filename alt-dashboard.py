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
        margin-bottom: 0;
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
        margin-top: 0;
    }
    .fc-card-forecast-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f0f4ff;
        line-height: 1.1;
        margin: 0;
        padding-bottom: 3px;
        margin-bottom: 3px;
        border-bottom: 0.5px solid #23263d;
    }
    .fc-card-actual-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 3px;
        margin-bottom: 3px;
        border-bottom: 0.5px solid #23263d;
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
        padding: 70px 22px;
        border: 1px solid #2a2d45;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .fc-mape-label {
        font-size: 0.9rem;
        color: #f0f4ff;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
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
    st.markdown("<hr style='border: 1px solid #2a2d45; margin: 20px 0'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#f0f4ff; text-align:center; font-weight:600;'>Upload Inventory Data (CSV)</h4>", unsafe_allow_html=True)
    st.caption("Columns: <b>product</b>, <b>current_inventory</b>, <b>min_inventory</b>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose inventory CSV", type=["csv"], label_visibility="collapsed", key="inventory_csv")
    if uploaded_file is not None:
        try:
            inv_df = pd.read_csv(uploaded_file)
            required = ['product', 'current_inventory', 'min_inventory']
            if all(c in inv_df.columns for c in required):
                st.session_state.inventory_df = inv_df
                st.dataframe(inv_df, use_container_width=True, hide_index=True)
            else:
                st.error(f"CSV must have columns: {', '.join(required)}")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    elif st.session_state.get('inventory_df') is not None:
        st.dataframe(st.session_state.inventory_df, use_container_width=True, hide_index=True)

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
if 'inventory_df' not in st.session_state:
    st.session_state.inventory_df = None

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🏠 Beranda", use_container_width=True, type="secondary"):
        st.session_state.app_page = 'home'
        st.rerun()
    selected_product = st.selectbox("Select product", products_sorted)
    st.markdown("<hr style='border: none; border-top: 1px solid #AAAAAA; margin: 2px 0 16px;'>", unsafe_allow_html=True)
    prod_pred = predictions[selected_product]
    st.markdown(f"**Best Model:** {prod_pred['best_model']}")
    st.markdown("<hr style='border: none; border-top: 1px solid #AAAAAA; margin: 2px 0 16px;'>", unsafe_allow_html=True)

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
    av = prod_pred['actual_vals']

    cols = st.columns([1, 1, 1, 1])
    for i, dt in enumerate(PRED_DATES):
        with cols[i]:
            pred_val = pv[i]
            actual_val = av[i]
            target_val = bv[i]
            actual_str = f"{actual_val:,.0f}" if not np.isnan(actual_val) else "—"
            target_str = f"{target_val:,.0f}" if not np.isnan(target_val) else "—"
            diff = pred_val - target_val if not (np.isnan(target_val) or np.isnan(pred_val)) else 0
            diff_pct = (diff / target_val * 100) if target_val != 0 and not np.isnan(target_val) and not np.isnan(pred_val) else 0
            above = diff >= 0
            status = "▲" if above else "▼"
            vclass = "above" if above else "below"
            vtext = f"+{diff_pct:.1f}%" if above else f"{diff_pct:.1f}%"
            fc_diff = pred_val - actual_val if not (np.isnan(actual_val) or np.isnan(pred_val)) else 0
            fc_diff_pct = (fc_diff / actual_val * 100) if actual_val != 0 and not np.isnan(actual_val) and not np.isnan(pred_val) else 0
            fc_above = fc_diff >= 0
            fc_vclass = "above" if fc_above else "below"
            fc_vtext = f"+{fc_diff_pct:.1f}%" if fc_above else f"{fc_diff_pct:.1f}%"
            st.markdown(f"""
            <div class="fc-card">
                <div class="fc-card-header">
                    <span class="fc-card-month">{dt.strftime('%B %Y')}</span>
                    <span class="fc-card-status">{status}</span>
                </div>
                <div class="fc-card-forecast-label">Forecast</div>
                <div class="fc-card-forecast-value">{pred_val:,.0f}</div>
                <div class="fc-card-actual-row">
                    <div>
                        <div class="fc-card-target-label">Actual</div>
                        <div class="fc-card-target-value" style="color:#7c9ef5">{actual_str}</div>
                    </div>
                    <span class="fc-card-variance {fc_vclass}">{fc_vtext}</span>
                </div>
                <div class="fc-card-target-row">
                    <div>
                        <div class="fc-card-target-label">Sales Target</div>
                        <div class="fc-card-target-value">{target_str}</div>
                    </div>
                    <span class="fc-card-variance {vclass}">{vtext}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f"""
        <div class="fc-mape-card">
            <div class="fc-mape-label">Walk-Forward MAPE</div>
            <div class="fc-mape-value">{prod_pred['mape']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border: none; border-top: 1px solid #2a2d45; margin: 24px 0 16px;'>", unsafe_allow_html=True)
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

    actual_line_dates = [last_actual_date] + list(PRED_DATES)
    actual_line_vals = [last_actual_val] + list(av)
    fig.add_trace(go.Scatter(
        x=actual_line_dates, y=actual_line_vals,
        name="Actual (H8)", mode="lines+markers",
        line=dict(color="#7c9ef5", width=1.5, dash="dash"),
        marker=dict(size=[4] + [8] * 3, symbol=["circle"] + ["diamond"] * 3)
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
        title=dict(text=f"Rolling Forecast — {selected_product}", x=0.4, font=dict(size=20)),
        xaxis_title=dict(text="Date", font=dict(size=13)),
        yaxis_title=dict(text="Sales", font=dict(size=13)),
        template="plotly_dark",
        hovermode="x unified", legend=dict(orientation="h", y=1.08),
        height=500, margin=dict(t=80, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr style='border: none; border-top: 1px solid #2a2d45; margin: 24px 0 16px;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #f0f4ff; font-size: 1.25rem; font-weight: 700; margin-bottom: 16px;'>Tabel Peramalan</h3>", unsafe_allow_html=True)

    table_rows = []
    for j, dt in enumerate(PRED_DATES):
        fcast = pv[j]
        act = av[j]
        tgt = bv[j]
        ape = abs((act - fcast) / act) * 100 if act != 0 and not np.isnan(act) and not np.isnan(fcast) else float('nan')
        table_rows.append({
            'Month': dt.strftime('%b %Y'),
            'Forecast': f"{fcast:,.0f}",
            'Actual': f"{act:,.0f}",
            'Sales Target': f"{tgt:,.0f}",
            'Error (%)': f"{ape:.1f}" if not np.isnan(ape) else "—",
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

# ── CAPACITY TAB ────────────────────────────────────────────────────────────────
with tab_capacity:
    st.markdown("<h3 style='text-align: center; color: #f0f4ff; font-size: 1.25rem; font-weight: 700; margin-bottom: 16px;'>Rekomendasi Perencanaan Shift Produksi</h3>", unsafe_allow_html=True)

    inv_df = st.session_state.get('inventory_df')
    inv_lookup = dict(zip(inv_df['product'], inv_df['current_inventory'])) if inv_df is not None else {}
    min_lookup = dict(zip(inv_df['product'], inv_df['min_inventory'])) if inv_df is not None else {}

    cur_inv = {p: inv_lookup.get(p, 0) for p in products_sorted}
    min_inv = {p: min_lookup.get(p, 0) for p in products_sorted}

    missing = [p for p in products_sorted if p not in inv_lookup and inv_df is not None]
    if missing:
        st.warning(f"Products missing from inventory CSV (defaulting to 0): {', '.join(missing)}")

    MAX_EXTRA_SHIFTS = 24
    shift_cap = round(22000)
    base_capacity = round(1320000 * 0.377)

    cap_data = []
    inv_detail_rows = []

    for pi, dt in enumerate(PRED_DATES):
        total_forecast = 0
        total_backlog = 0
        prod_details = []

        for p in products_sorted:
            fcast = predictions[p]['pred_vals'][pi]
            inv = cur_inv[p]
            min_inv_val = min_inv[p]

            refill = max(0, min_inv_val - inv)
            effective_i = fcast + refill
            total_forecast += fcast
            total_backlog += refill
            prod_details.append({'product': p, 'forecast': fcast, 'start_inv': inv, 'min_inv': min_inv_val, 'refill': refill, 'effective_demand': effective_i})

        total_effective = total_forecast + total_backlog
        excess = max(0, total_effective - base_capacity)
        extra_shifts_raw = int(np.ceil(excess / shift_cap)) if excess > 0 else 0
        extra_shifts = min(MAX_EXTRA_SHIFTS, extra_shifts_raw)
        capped_excess = extra_shifts * shift_cap
        total_production = base_capacity + capped_excess
        hit_cap = extra_shifts_raw > MAX_EXTRA_SHIFTS

        safety_consumed = 0
        for d in prod_details:
            share = d['effective_demand'] / total_effective if total_effective > 0 else 0
            allocated = total_production * share
            d['allocated'] = allocated
            d['end_inv'] = d['start_inv'] - d['forecast'] + allocated
            cur_inv[d['product']] = d['end_inv']
            consumed = max(0, d['min_inv'] - d['end_inv'])
            d['safety_used'] = consumed
            safety_consumed += consumed
            if d['end_inv'] >= d['min_inv']:
                d['status'] = '✅ OK'
            elif d['end_inv'] >= 0:
                d['status'] = '⚠️ Below Min'
            else:
                d['status'] = '🔴 Depleted'

        cap_data.append({'date': dt, 'total_forecast': total_forecast, 'total_backlog': total_backlog, 'effective_demand': total_effective, 'capacity': base_capacity, 'excess': excess, 'extra_shifts_raw': extra_shifts_raw, 'extra_shifts': extra_shifts, 'total_production': total_production, 'hit_cap': hit_cap, 'safety_consumed': safety_consumed})

        for d in prod_details:
            inv_detail_rows.append({'Month': dt.strftime('%b %Y'), 'Product': d['product'], 'Start Inv': f"{d['start_inv']:,.0f}", 'Min Inv': f"{d['min_inv']:,.0f}", 'Forecast': f"{d['forecast']:,.0f}", 'Backlog': f"{d['refill']:,.0f}", 'Allocated Prod': f"{d['allocated']:,.0f}", 'End Inv': f"{d['end_inv']:,.0f}",  'Status': d['status']})

    capacity_df = pd.DataFrame(cap_data)

    if not capacity_df.empty:
        cols = st.columns(len(capacity_df))
        for idx, row in capacity_df.iterrows():
            with cols[idx]:
                num_shifts = row['extra_shifts']
                if num_shifts > 0:
                    text = f"{num_shifts} extra shift{'s' if num_shifts > 1 else ''} ⚠️"
                    if row['hit_cap']:
                        text += "<br><span style='font-size:0.7rem;color:#f5d17c'>(max 24)</span>"
                    color = "#f5d17c"
                else:
                    text = "Within capacity ✅"
                    color = "#7cf5a5"
                safety_text = f"<div class='perf-sub'>Safety Stock Used: {row['safety_consumed']:,.0f}</div>" if row['safety_consumed'] > 0 else ""
                st.markdown(f"""
                <div class="perf-card">
                    <div class="perf-label">{row['date'].strftime('%B %Y')}</div>
                    <div class="perf-value" style="color:{color}">{text}</div>
                    <div class="perf-sub">Forecast: {row['total_forecast']:,.0f}</div>
                    <div class="perf-sub">Effective Demand: {row['effective_demand']:,.0f}</div>
                    <div class="perf-sub">Capacity: {row['capacity']:,.0f}</div>
                    <div class="perf-sub">Total Production: {row['total_production']:,.0f}</div>
                    {safety_text}
                </div>""", unsafe_allow_html=True)

        st.markdown("<hr style='border: none; border-top: 1px solid #2a2d45; margin: 24px 0 16px;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: #f0f4ff; font-size: 1.25rem; font-weight: 700; margin-bottom: 16px;'>Tabel Kapasitas</h3>", unsafe_allow_html=True)
        st.dataframe(capacity_df.style.format({
            "total_forecast": "{:,.0f}",
            "total_backlog": "{:,.0f}",
            "effective_demand": "{:,.0f}",
            "capacity": "{:,.0f}",
            "excess": "{:,.0f}",
            "extra_shifts": "{:,d}",
            "total_production": "{:,.0f}",
            "safety_consumed": "{:,.0f}",
        }), use_container_width=True, hide_index=True)

        st.markdown("<hr style='border: none; border-top: 1px solid #2a2d45; margin: 24px 0 16px;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: #f0f4ff; font-size: 1.25rem; font-weight: 700; margin-bottom: 16px;'>Tabel Inventory per Produk</h3>", unsafe_allow_html=True)
        inv_detail_df = pd.DataFrame(inv_detail_rows)
        st.dataframe(inv_detail_df, use_container_width=True, hide_index=True)
    else:
        st.info("No prediction data available.")

# ── PERFORMANCE TAB ─────────────────────────────────────────────────────────────
with tab_performance:
    st.markdown("<h3 style='text-align: center; color: #f0f4ff; font-size: 1.25rem; font-weight: 700; margin-bottom: 16px;'>Performa Model Keseluruhan</h3>", unsafe_allow_html=True)

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

    st.markdown("<hr style='border: none; border-top: 1px solid #2a2d45; margin: 24px 0 16px;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #f0f4ff; font-size: 1.25rem; font-weight: 700; margin-bottom: 16px;'>Tabel Perbandingan Model</h3>", unsafe_allow_html=True)
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
