import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
import os

st.set_page_config(page_title="Forecast Dashboard", page_icon="📈", layout="wide")

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
DATA_PATH   = r"D:\Zia\ml_ready_with_hypothetical_po_beta.csv"

# Map each product to its specific model and scaler paths.
# Note: Update keys ("Product_A", etc.) to match your exact product names in the CSV.
PRODUCT_MODELS = {
    "MU-302-50Kg": {
        "model": r"D:\Zia\model302.pkl",
        "scaler": None
    },
    "Product_B": {
        "model": r"D:\Zia\lasso_model_B.pkl",
        "scaler": r"D:\Zia\scaler_lasso_B.pkl"
    },
    "Product_C": {
        "model": r"D:\Zia\lasso_model_C.pkl",
        "scaler": None  # Omit or set to None if no scaler is needed
    },
    "Product_D": {
        "model": r"D:\Zia\lasso_model_D.pkl",
        "scaler": None
    }
}

FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "roll_mean_3", "roll_mean_6", "roll_std_3",
    "trend_slope_12", "fc_error_1", "fc_acc_1"
]

FEATURE_EXOG = [
    "quarter", "is_lebaran_window", "is_year_end", "year_index", "fc_acc_1", "fc_error_1"
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def mape(actual, pred):
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100

def mae(actual, pred):
    return np.mean(np.abs(actual - pred))

def rmse(actual, pred):
    return np.sqrt(np.mean((actual - pred) ** 2))

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_csv(path):
    return pd.read_csv(path, parse_dates=["date"])

@st.cache_resource
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_expected_features(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if hasattr(obj, "get_booster"):
        try:
            return obj.get_booster().feature_names
        except Exception:
            pass
    if hasattr(obj, "feature_name"):
        try:
            return obj.feature_name()
        except Exception:
            pass
    if hasattr(obj, "model") and hasattr(obj.model, "exog_names"):
        if obj.model.exog_names is not None:
            return [c for c in obj.model.exog_names if c not in ["const", "intercept"]]
    return None

def detect_and_predict(X, model_path, scaler_path, model_family="Machine Learning"):
    from sklearn.pipeline import Pipeline
    obj = load_pkl(model_path)

    def prepare_X(X_input, component):
        expected = get_expected_features(component)
        if expected is not None:
            missing = [c for c in expected if c not in X_input.columns]
            if missing:
                raise ValueError(f"Missing required features: {missing}")
            return X_input[expected]
        return X_input

    if model_family == "Time Series (ARIMAX/VARX)":
        X_exog = prepare_X(X, obj)
        # Convert to numpy array to prevent pandas index alignment slicing errors in statsmodels/pmdarima
        exog_data = X_exog.values if not X_exog.empty else None
        
        if hasattr(obj, "forecast"):
            import inspect
            sig = inspect.signature(obj.forecast)
            
            pred = None
            if exog_data is not None:
                if "exog" in sig.parameters:
                    pred = obj.forecast(steps=len(X), exog=exog_data)
                elif "X" in sig.parameters:
                    pred = obj.forecast(n_periods=len(X), X=exog_data)
                else:
                    # Might be hidden in **kwargs (e.g. statsmodels VARMAX)
                    try:
                        pred = obj.forecast(steps=len(X), exog=exog_data)
                    except TypeError:
                        pred = obj.forecast(steps=len(X))
            else:
                if "n_periods" in sig.parameters:
                    pred = obj.forecast(n_periods=len(X))
                else:
                    pred = obj.forecast(steps=len(X))
                    
            if isinstance(pred, tuple):
                pred = pred[0]
            
            if isinstance(pred, pd.DataFrame):
                pred = pred.iloc[:, 0]
            
            pred_arr = np.asarray(pred)
            if pred_arr.ndim > 1:
                pred_arr = pred_arr[:, 0]  # Only take the first endogenous variable ('actual') for VARX
            return pred_arr, f"{type(obj).__name__} forecast"

    if isinstance(obj, Pipeline):
        return obj.predict(prepare_X(X, obj)), "sklearn Pipeline"
    if isinstance(obj, dict):
        model  = obj.get("model") or obj.get("estimator") or obj.get("regressor")
        scaler = obj.get("scaler") or obj.get("preprocessor") or obj.get("transformer")
        if model is not None:
            Xt = scaler.transform(prepare_X(X, scaler)) if scaler is not None else prepare_X(X, model)
            return model.predict(Xt), "dict bundle"
    if isinstance(obj, (list, tuple)) and len(obj) == 2:
        a, b = obj
        if hasattr(a, "transform") and hasattr(b, "predict"):
            return b.predict(a.transform(prepare_X(X, a))), "tuple (scaler, model)"
        if hasattr(b, "transform") and hasattr(a, "predict"):
            return a.predict(b.transform(prepare_X(X, b))), "tuple (model, scaler)"
    if scaler_path and os.path.exists(scaler_path):
        scaler = load_pkl(scaler_path)
        return obj.predict(scaler.transform(prepare_X(X, scaler))), f"bare model + {scaler_path}"
    return obj.predict(prepare_X(X, obj)), "bare model"

# ── Load data ─────────────────────────────────────────────────────────────────
st.title("📈 Sales Forecast Dashboard")

data_ok = os.path.exists(DATA_PATH)
if not data_ok:
    st.error(f"Data file not found: `{DATA_PATH}`.")
    st.stop()

df_raw = load_csv(DATA_PATH)
products = sorted(df_raw["product"].unique())
selected_product = st.selectbox("Select product", products)

# Look up model configuration for the selected product
prod_config = PRODUCT_MODELS.get(selected_product, {})
current_model_path  = prod_config.get("model")
current_scaler_path = prod_config.get("scaler")

model_ok  = bool(current_model_path and os.path.exists(current_model_path))
scaler_ok = bool(current_scaler_path and os.path.exists(current_scaler_path))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")
    model_family = st.radio("Model Family", ["Machine Learning", "Time Series (ARIMAX/VARX)"])
    st.divider()
    st.caption(f"{'✅' if model_ok  else '❌'} Model:  `{current_model_path or 'Not configured'}`")
    st.caption(f"{'✅' if scaler_ok else '⚪'} Scaler: `{current_scaler_path or 'Not configured'}` _(optional)_")
    st.caption(f"{'✅' if data_ok   else '❌'} Data:   `{DATA_PATH}`")
    st.divider()
    st.caption("Machine Learning Features:")
    for f in FEATURE_COLS:
        st.caption(f"• {f}")
    st.caption("Exogenous Variables (ARIMAX/VARX):")
    for f in FEATURE_EXOG:
        st.caption(f"• {f}")

df = df_raw[df_raw["product"] == selected_product].copy().sort_values("date").reset_index(drop=True)

# Split train/test
df_test  = df[df["split"] == "test"].copy()  if "split" in df.columns else df.copy()
df_train = df[df["split"] == "train"].copy() if "split" in df.columns else pd.DataFrame()

# ── Predict on TEST only ──────────────────────────────────────────────────────
model_forecast_test = None
model_info = ""

if model_family == "Time Series (ARIMAX/VARX)":
    available_features = [c for c in FEATURE_EXOG if c in df.columns]
else:
    available_features = [c for c in FEATURE_COLS + FEATURE_EXOG if c in df.columns]

if not model_ok:
    st.warning(f"Model not found or configured for {selected_product}: `{current_model_path}`.")
else:
    if not available_features and model_family != "Time Series (ARIMAX/VARX)":
        st.error("No valid feature columns found in the data.")
    elif df_test.empty:
        st.warning("No test rows found (split == 'test').")
    else:   
        X_test = df_test[available_features].fillna(0) if available_features else pd.DataFrame(index=df_test.index)
        try:
            model_forecast_test, model_info = detect_and_predict(X_test, current_model_path, current_scaler_path, model_family)
            df.loc[df["split"] == "test", "model_forecast"] = model_forecast_test
            df_test["model_forecast"] = model_forecast_test
            st.success(f"✅ {model_info} — {len(df_test)} test predictions generated.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ── Future Forecast (Next 3 Months) ────────────────────────────────
df_future = pd.DataFrame()
if model_ok and not df.empty and (available_features or model_family == "Time Series (ARIMAX/VARX)"):
    base_df = df.copy()
    
    if model_family == "Time Series (ARIMAX/VARX)":
        future_rows = []
        last_date = base_df["date"].max()
        for i in range(1, 4):
            next_date = last_date + pd.DateOffset(months=i)
            new_row = {"date": next_date, "split": "future"}
            new_row["quarter"] = next_date.quarter
            new_row["is_year_end"] = 1 if next_date.month == 12 else 0
            new_row["is_lebaran_window"] = 0
            new_row["year_index"] = next_date.year - base_df["date"].dt.year.min()
            if "fc_acc_1" in base_df.columns:
                new_row["fc_acc_1"] = base_df["fc_acc_1"].iloc[-1]
            else:
                new_row["fc_acc_1"] = 0
            if "fc_error_1" in base_df.columns:
                new_row["fc_error_1"] = base_df["fc_error_1"].iloc[-1]
            else:
                new_row["fc_error_1"] = 0
            future_rows.append(new_row)
            
        df_fut = pd.DataFrame(future_rows)
        X_future = df_fut[[c for c in FEATURE_EXOG if c in df_fut.columns]].fillna(0)
        
        try:
            pred_vals, info = detect_and_predict(X_future, current_model_path, current_scaler_path, model_family)
            for i, row in enumerate(future_rows):
                row["actual"] = np.nan
                row["model_forecast"] = pred_vals[i]
                row["forecast"] = pred_vals[i]
            df_future = pd.DataFrame(future_rows)
            df = pd.concat([df, df_future], ignore_index=True)
        except Exception as e:
            st.error(f"Time Series forecast failed: {e}")
            
    else:
        future_rows = []
        for _ in range(3):
            last_date = base_df["date"].max()
            next_date = last_date + pd.DateOffset(months=1)
            
            new_row = {"date": next_date, "split": "future"}
            actuals = base_df["actual"].values
            
            def get_lag(n): return actuals[-n] if len(actuals) >= n else 0
            
            new_row["lag_1"] = get_lag(1)
            new_row["lag_2"] = get_lag(2)
            new_row["lag_3"] = get_lag(3)
            new_row["lag_6"] = get_lag(6)
            new_row["lag_12"] = get_lag(12)
            
            new_row["roll_mean_3"] = np.mean(actuals[-3:]) if len(actuals) >= 3 else 0
            new_row["roll_mean_6"] = np.mean(actuals[-6:]) if len(actuals) >= 6 else 0
            new_row["roll_std_3"]  = np.std(actuals[-3:], ddof=1) if len(actuals) >= 3 else 0
            
            if len(actuals) >= 12:
                new_row["trend_slope_12"] = np.polyfit(np.arange(12), actuals[-12:], 1)[0]
            else:
                new_row["trend_slope_12"] = 0
                
            new_row["quarter"] = next_date.quarter
            new_row["is_year_end"] = 1 if next_date.month == 12 else 0
            new_row["is_lebaran_window"] = 0
            new_row["year_index"] = next_date.year - base_df["date"].dt.year.min()
            
            if "fc_error_1" in base_df.columns: new_row["fc_error_1"] = base_df["fc_error_1"].iloc[-1]
            else: new_row["fc_error_1"] = 0
                
            if "fc_acc_1" in base_df.columns: new_row["fc_acc_1"] = base_df["fc_acc_1"].iloc[-1]
            else: new_row["fc_acc_1"] = 0
                
            # fallback for any other features
            for f in available_features:
                if f not in new_row:
                    new_row[f] = base_df[f].iloc[-1] if f in base_df.columns else 0
                    
            X_new = pd.DataFrame([new_row])[available_features].fillna(0)
            try:
                pred_val, info = detect_and_predict(X_new, current_model_path, current_scaler_path, model_family)
                # Pass the forecast forward as the "actual" to calculate accurate recursive lags for the next step
                new_row["actual"] = pred_val[0]
                new_row["model_forecast"] = pred_val[0]
                new_row["forecast"] = pred_val[0]
                future_rows.append(new_row)
                base_df = pd.concat([base_df, pd.DataFrame([new_row])], ignore_index=True)
            except Exception as e:
                st.error(f"Recursive prediction failed: {e}")
                break
                
        if future_rows:
            df_future = pd.DataFrame(future_rows)
            # We nullify "actual" before pushing it to display so the blue Line doesn't extend into the future
            df_future_for_display = df_future.copy()
            df_future_for_display["actual"] = np.nan
            df = pd.concat([df, df_future_for_display], ignore_index=True)

# ── Performance cards (test set only) ────────────────────────────────────────
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

if model_forecast_test is not None and not df_test.empty:
    act  = df_test["actual"].values
    pred = df_test["model_forecast"].values
    base = df_test["forecast"].values

    m_mape  = mape(act, pred);   b_mape  = mape(act, base)
    m_mae   = mae(act, pred);    b_mae   = mae(act, base)
    m_rmse  = rmse(act, pred);   b_rmse  = rmse(act, base)

    c1, c2, c3 = st.columns(3)
    perf_card(c1, "MAPE",  m_mape,  b_mape,  unit="%")
    perf_card(c2, "MAE",   m_mae,   b_mae)
    perf_card(c3, "RMSE",  m_rmse,  b_rmse)
    st.markdown("<br>", unsafe_allow_html=True)
elif not df_test.empty:
    # show baseline perf only
    act  = df_test["actual"].values
    base = df_test["forecast"].values
    st.info(f"Baseline test MAPE: **{mape(act,base):.2f}%** | MAE: **{mae(act,base):,.0f}** | RMSE: **{rmse(act,base):,.0f}**")

# ── Main chart — ALL data, model line only on test ────────────────────────────
fig = go.Figure()

# Actual — full series
fig.add_trace(go.Scatter(x=df["date"], y=df["actual"],
    name="Actual", mode="lines+markers",
    line=dict(color="#7c9ef5", width=2), marker=dict(size=4)))

# Baseline forecast — full series (dotted)
fig.add_trace(go.Scatter(x=df["date"], y=df["forecast"],
    name="Baseline Forecast", mode="lines",
    line=dict(color="#f5a97c", width=1.5, dash="dot")))

# Model forecast — test set only
if "model_forecast" in df.columns:
    df_test_plot = df[df["split"] == "test"]
    if not df_test_plot.empty:
        fig.add_trace(go.Scatter(
            x=df_test_plot["date"],
            y=df_test_plot["model_forecast"],
            name=f"Model Forecast ({model_info})", mode="lines+markers",
            line=dict(color="#7cf5a5", width=2.5), marker=dict(size=5)))
            
    df_future_plot = df[df["split"] == "future"]
    if not df_future_plot.empty:
        last_known = df[df["split"] != "future"].iloc[[-1]].copy()
        last_known["model_forecast"] = last_known["model_forecast"].fillna(last_known["actual"])
        plot_future = pd.concat([last_known, df_future_plot])
        fig.add_trace(go.Scatter(
            x=plot_future["date"],
            y=plot_future["model_forecast"],
            name="Future Forecast (3M)", mode="lines+markers",
            line=dict(color="#f5d17c", width=2.5, dash="dot"), marker=dict(size=7, symbol="star")))

# Shade test region
if not df_test.empty:
    fig.add_vrect(x0=df_test["date"].min(), x1=df_test["date"].max(),
        fillcolor="rgba(124,158,245,0.06)", line_width=0,
        annotation_text="Test set", annotation_position="top left",
        annotation_font_color="#8a8fa8")

df_future_plot = df[df["split"] == "future"]
if not df_future_plot.empty:
    fig.add_vrect(x0=df_future_plot["date"].min(), x1=df_future_plot["date"].max(),
        fillcolor="rgba(245,209,124,0.06)", line_width=0,
        annotation_text="Future 3M", annotation_position="top right",
        annotation_font_color="#f5d17c")

fig.update_layout(
    title=f"Forecast vs Actual — {selected_product}",
    xaxis_title="Date", yaxis_title="Sales", template="plotly_dark",
    hovermode="x unified", legend=dict(orientation="h", y=1.08),
    height=440, margin=dict(t=60, b=40))
st.plotly_chart(fig, use_container_width=True)

# ── Error charts (test set) ───────────────────────────────────────────────────
if not df_test.empty:
    ca, cb = st.columns(2)

    with ca:
        err_base  = df_test["actual"] - df_test["forecast"]
        fig_err = go.Figure()
        fig_err.add_trace(go.Bar(x=df_test["date"], y=err_base,
            marker_color=np.where(err_base >= 0, "#7cf5a5", "#f5687c"),
            name="Baseline Error"))
        if "model_forecast" in df_test.columns:
            err_model = df_test["actual"] - df_test["model_forecast"]
            fig_err.add_trace(go.Scatter(x=df_test["date"], y=err_model,
                name="Model Error", mode="lines+markers",
                line=dict(color="#d97cf5", width=2)))
        fig_err.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
        fig_err.update_layout(title="Forecast Error on Test Set (Actual − Pred)",
            template="plotly_dark", height=320, margin=dict(t=50, b=30))
        st.plotly_chart(fig_err, use_container_width=True)

    with cb:
        fig_mape = go.Figure()
        # rolling MAPE per row on test
        if "model_forecast" in df_test.columns:
            mape_model    = np.abs((df_test["actual"] - df_test["model_forecast"]) / df_test["actual"].replace(0, np.nan)) * 100
            mape_baseline = np.abs((df_test["actual"] - df_test["forecast"])       / df_test["actual"].replace(0, np.nan)) * 100
            fig_mape.add_trace(go.Scatter(x=df_test["date"], y=mape_baseline,
                name="Baseline MAPE", mode="lines",
                line=dict(color="#f5a97c", width=1.5, dash="dot")))
            fig_mape.add_trace(go.Scatter(x=df_test["date"], y=mape_model,
                name="Model MAPE", mode="lines+markers",
                line=dict(color="#7cf5a5", width=2)))
            fig_mape.update_layout(title="MAPE per Period — Test Set",
                yaxis_title="MAPE (%)", template="plotly_dark",
                height=320, margin=dict(t=50, b=30))
            st.plotly_chart(fig_mape, use_container_width=True)

# ── Seasonal & lag ────────────────────────────────────────────────────────────
st.subheader("Seasonal & Lag Patterns")
c3, c4 = st.columns(2)
with c3:
    df["month"] = df["date"].dt.month
    monthly = df.groupby("month")["actual"].mean().reset_index()
    fig_sea = px.bar(monthly, x="month", y="actual", title="Avg Actual by Month",
        template="plotly_dark", color="actual", color_continuous_scale="blues")
    fig_sea.update_layout(height=300, margin=dict(t=50, b=30))
    st.plotly_chart(fig_sea, use_container_width=True)

with c4:
    lag_cols = [c for c in ["lag_1","lag_2","lag_3","lag_6","lag_12"] if c in df.columns]
    if lag_cols:
        corrs = {c: df["actual"].corr(df[c]) for c in lag_cols}
        fig_lag = go.Figure(go.Bar(x=list(corrs.keys()), y=list(corrs.values()),
            marker_color="#7c9ef5"))
        fig_lag.update_layout(title="Lag Correlation with Actual",
            template="plotly_dark", height=300, margin=dict(t=50, b=30))
        st.plotly_chart(fig_lag, use_container_width=True)

# ── Raw table ─────────────────────────────────────────────────────────────────
with st.expander("📋 Raw Data Table"):
    show_cols = ["date","split","actual","forecast"] + \
        (["model_forecast"] if "model_forecast" in df.columns else []) + \
        [c for c in ["fc_acc_1","fc_error_1"] if c in df.columns]
    st.dataframe(df[show_cols].style.format({
        "actual":"{:,.0f}", "forecast":"{:,.0f}",
        "model_forecast":"{:,.0f}", "fc_acc_1":"{:.3f}", "fc_error_1":"{:.3f}"
    }), use_container_width=True)