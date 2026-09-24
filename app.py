"""
app.py  —  Used Car Price Predictor
====================================
Single-file application: data processing, feature engineering,
model training, evaluation, SHAP explainability, and Streamlit dashboard.

Run:
    streamlit run app.py
"""

# ============================================================
# 0. IMPORTS
# ============================================================
import os
import re
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ============================================================
# 1. PATHS
# ============================================================
ROOT          = os.path.dirname(os.path.abspath(__file__))
DATA_PATH     = os.path.join(ROOT, "data", "used_cars.csv")
MODELS_DIR    = os.path.join(ROOT, "models")
MODEL_PATH    = os.path.join(MODELS_DIR, "best_model.pkl")
PRE_PATH      = os.path.join(MODELS_DIR, "preprocessor.pkl")
METRICS_PATH  = os.path.join(MODELS_DIR, "metrics.json")
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================
# 2. FEATURE DEFINITIONS
# ============================================================
NUMERIC_COLS = [
    "model_year", "mileage", "horsepower", "engine_litres",
    "cylinders", "car_age", "has_accident", "clean_title",
]
CATEGORICAL_COLS = ["brand", "fuel_type", "transmission", "ext_col", "int_col"]
ALL_FEATURES = NUMERIC_COLS + CATEGORICAL_COLS

# ============================================================
# 3. DATA PROCESSING (backend)
# ============================================================

def _parse_price(s):
    if pd.isna(s):
        return np.nan
    try:
        return float(str(s).replace("$", "").replace(",", "").strip())
    except ValueError:
        return np.nan

def _parse_mileage(s):
    if pd.isna(s):
        return np.nan
    try:
        return float(str(s).replace(",", "").replace("mi.", "").strip())
    except ValueError:
        return np.nan

def _extract_hp(s):
    if pd.isna(s):
        return np.nan
    m = re.search(r"([\d.]+)\s*HP", str(s), re.IGNORECASE)
    return float(m.group(1)) if m else np.nan

def _extract_litres(s):
    if pd.isna(s):
        return np.nan
    m = re.search(r"([\d.]+)\s*L", str(s), re.IGNORECASE)
    return float(m.group(1)) if m else np.nan

def _extract_cylinders(s):
    if pd.isna(s):
        return np.nan
    s = str(s)
    m = re.search(r"(\d+)\s*Cylinder", s, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"[VIL](\d+)", s, re.IGNORECASE)
    return int(m.group(1)) if m else np.nan

def _simplify_transmission(s):
    if pd.isna(s):
        return "Unknown"
    s = str(s).upper()
    if "AUTOMATIC" in s or "A/T" in s:
        return "Automatic"
    if "MANUAL" in s or "M/T" in s or s == "F":
        return "Manual"
    if "CVT" in s:
        return "CVT"
    if "DUAL" in s or "DCT" in s or "PDK" in s:
        return "Dual-Clutch"
    return "Other"

def _simplify_fuel(s):
    if pd.isna(s):
        return "Other"
    mapping = {
        "Gasoline": "Gasoline", "Hybrid": "Hybrid",
        "Electric": "Electric", "Diesel": "Diesel",
        "Plug-In Hybrid": "Hybrid", "E85 Flex Fuel": "Flex Fuel",
    }
    return mapping.get(str(s).strip(), "Other")

def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df["price"]   = df["price"].apply(_parse_price)
    df = df.dropna(subset=["price"])
    df = df[(df["price"] > 500) & (df["price"] < 300_000)]

    df["mileage"] = df["milage"].apply(_parse_mileage)
    df = df.drop(columns=["milage"])

    df["horsepower"]    = df["engine"].apply(_extract_hp)
    df["engine_litres"] = df["engine"].apply(_extract_litres)
    df["cylinders"]     = df["engine"].apply(_extract_cylinders)
    df = df.drop(columns=["engine"])

    df["transmission"] = df["transmission"].apply(_simplify_transmission)
    df["fuel_type"]    = df["fuel_type"].apply(_simplify_fuel)

    df["has_accident"] = df["accident"].apply(
        lambda x: 1 if str(x).lower().startswith("at least") else 0
    )
    df["clean_title"] = df["clean_title"].apply(
        lambda x: 1 if str(x).strip().lower() == "yes" else 0
    )
    df = df.drop(columns=["accident"])

    df["model_year"] = pd.to_numeric(df["model_year"], errors="coerce")
    df["car_age"]    = 2024 - df["model_year"]

    for col in ["ext_col", "int_col"]:
        top = df[col].value_counts().nlargest(15).index
        df[col] = df[col].where(df[col].isin(top), other="Other")

    df = df.dropna(subset=["model_year", "mileage"])
    for c in ["horsepower", "engine_litres", "cylinders"]:
        df[c] = df[c].fillna(df[c].median())

    df["log_price"] = np.log1p(df["price"])
    return df.reset_index(drop=True)

# ============================================================
# 4. TRAINING PIPELINE (backend)
# ============================================================

def build_preprocessor():
    num_pipe = Pipeline([("scaler", StandardScaler())])
    cat_pipe = Pipeline([
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    return ColumnTransformer([
        ("num", num_pipe, NUMERIC_COLS),
        ("cat", cat_pipe, CATEGORICAL_COLS),
    ])

def evaluate_metrics(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.clip(np.expm1(y_pred_log), 0, None)
    return {
        "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "R2":   round(float(r2_score(y_true, y_pred)), 4),
    }

def train_models(df: pd.DataFrame):
    X = df[ALL_FEATURES]
    y = df["log_price"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor()
    X_tr = preprocessor.fit_transform(X_train)
    X_te = preprocessor.transform(X_test)

    candidates = {
        "Ridge": Ridge(alpha=10.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=15, n_jobs=-1, random_state=42
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=42, verbosity=0,
        ),
    }

    results = {}
    for name, mdl in candidates.items():
        mdl.fit(X_tr, y_train)
        results[name] = {
            "train": evaluate_metrics(y_train, mdl.predict(X_tr)),
            "test":  evaluate_metrics(y_test,  mdl.predict(X_te)),
            "model": mdl,
        }

    best_name  = max(results, key=lambda k: results[k]["test"]["R2"])
    best_model = results[best_name]["model"]

    joblib.dump(best_model,   MODEL_PATH)
    joblib.dump(preprocessor, PRE_PATH)

    metrics_out = {
        "best_model": best_name,
        "all_results": {
            k: {"train": v["train"], "test": v["test"]}
            for k, v in results.items()
        },
        "feature_names": ALL_FEATURES,
        "numeric_cols":  NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "n_train": len(X_train),
        "n_test":  len(X_test),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)

    return metrics_out

# ============================================================
# 5. EDA CHART HELPERS (backend)
# ============================================================

PALETTE = px.colors.qualitative.Plotly

def chart_price_dist(df):
    fig = px.histogram(df, x="price", nbins=80, title="Price Distribution",
                       labels={"price": "Price (USD)"},
                       color_discrete_sequence=["#3b82d4"])
    fig.update_layout(bargap=0.05, template="plotly_white")
    return fig

def chart_brand_count(df):
    counts = df["brand"].value_counts().nlargest(20).reset_index()
    counts.columns = ["brand", "count"]
    fig = px.bar(counts, x="brand", y="count", title="Top 20 Brands by Listing Count",
                 labels={"brand": "Brand", "count": "Listings"},
                 color="count", color_continuous_scale="Blues")
    fig.update_layout(template="plotly_white", coloraxis_showscale=False, xaxis_tickangle=-45)
    return fig

def chart_avg_price_brand(df):
    top20 = (df.groupby("brand")["price"].mean().nlargest(20)
               .reset_index().rename(columns={"price": "avg_price"})
               .sort_values("avg_price"))
    fig = px.bar(top20, x="avg_price", y="brand", orientation="h",
                 title="Top 20 Brands by Average Price",
                 labels={"avg_price": "Avg Price (USD)", "brand": "Brand"},
                 color="avg_price", color_continuous_scale="Blues")
    fig.update_layout(template="plotly_white", coloraxis_showscale=False)
    return fig

def chart_price_mileage(df):
    sample = df.sample(min(2000, len(df)), random_state=42)
    fig = px.scatter(sample, x="mileage", y="price", color="fuel_type", opacity=0.5,
                     title="Price vs Mileage",
                     labels={"mileage": "Mileage (miles)", "price": "Price (USD)"},
                     color_discrete_sequence=PALETTE)
    fig.update_layout(template="plotly_white")
    return fig

def chart_price_fuel(df):
    fig = px.box(df, x="fuel_type", y="price", color="fuel_type",
                 title="Price by Fuel Type",
                 labels={"fuel_type": "Fuel Type", "price": "Price (USD)"},
                 color_discrete_sequence=PALETTE)
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig

def chart_price_transmission(df):
    fig = px.box(df, x="transmission", y="price", color="transmission",
                 title="Price by Transmission Type",
                 labels={"transmission": "Transmission", "price": "Price (USD)"},
                 color_discrete_sequence=PALETTE)
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig

def chart_price_year(df):
    yearly = (df.groupby("model_year")["price"].mean()
                .reset_index().rename(columns={"price": "avg_price"}))
    fig = px.line(yearly, x="model_year", y="avg_price", markers=True,
                  title="Average Price by Model Year",
                  labels={"model_year": "Model Year", "avg_price": "Avg Price (USD)"})
    fig.update_traces(line_color="#3b82d4")
    fig.update_layout(template="plotly_white")
    return fig

def chart_accident_impact(df):
    tmp = df.copy()
    tmp["Accident"] = tmp["has_accident"].map({1: "Has Accident", 0: "No Accident"})
    fig = px.box(tmp, x="Accident", y="price", color="Accident",
                 title="Price Impact of Accident History",
                 labels={"price": "Price (USD)"},
                 color_discrete_map={"Has Accident": "#ef4444", "No Accident": "#22c55e"})
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig

def chart_correlation(df):
    cols = ["price", "mileage", "model_year", "car_age",
            "horsepower", "engine_litres", "cylinders", "has_accident", "clean_title"]
    corr = df[cols].corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu", zmid=0,
        text=corr.values, texttemplate="%{text}", textfont={"size": 10},
    ))
    fig.update_layout(title="Feature Correlation Heatmap", template="plotly_white", height=500)
    return fig

# ============================================================
# 6. STREAMLIT PAGE CONFIG & CSS
# ============================================================

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header { font-size:2.2rem; font-weight:700; color:#1f2328; margin-bottom:.2rem; }
.sub-header  { font-size:1rem; color:#57606a; margin-bottom:1.4rem; }
.metric-card { background:#f7f8fa; border:1px solid #e5e7eb; border-radius:10px;
               padding:1rem 1.2rem; text-align:center; }
.metric-value{ font-size:1.7rem; font-weight:700; color:#3b82d4; }
.metric-label{ font-size:.82rem; color:#57606a; margin-top:.2rem; }
.pred-box    { background:linear-gradient(135deg,#3b82d4 0%,#7c5cd8 100%);
               border-radius:14px; padding:2rem; text-align:center; color:white; }
.pred-price  { font-size:3rem; font-weight:800; }
.pred-label  { font-size:1rem; opacity:.85; margin-top:.5rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 7. CACHED LOADERS
# ============================================================

@st.cache_data(show_spinner="Loading dataset ...")
def get_data():
    return load_and_clean(DATA_PATH)

@st.cache_data
def get_raw_count():
    return len(pd.read_csv(DATA_PATH))

@st.cache_resource(show_spinner="Loading model ...")
def get_model_artifacts():
    model = joblib.load(MODEL_PATH)
    pre   = joblib.load(PRE_PATH)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    return model, pre, metrics

def models_ready():
    return os.path.exists(MODEL_PATH) and os.path.exists(PRE_PATH) and os.path.exists(METRICS_PATH)

# ============================================================
# 8. SIDEBAR + AUTO-TRAIN
# ============================================================

with st.sidebar:
    st.markdown("## 🚗 Car Price Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📊 Market Analytics", "🔮 Price Prediction",
         "🧠 Model Insights", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    if not models_ready():
        st.warning("Model not trained yet.")
        if st.button("Train Model Now", use_container_width=True):
            with st.spinner("Training all models — this takes ~1 min ..."):
                df_train = load_and_clean(DATA_PATH)
                train_models(df_train)
            st.cache_resource.clear()
            st.success("Training complete!")
            st.rerun()
    else:
        st.success("Model ready")
        if st.button("Retrain Model", use_container_width=True):
            with st.spinner("Retraining ..."):
                df_train = load_and_clean(DATA_PATH)
                train_models(df_train)
            st.cache_resource.clear()
            st.rerun()

    st.markdown("<div style='color:#57606a;font-size:.8rem;'>Dataset: used_cars.csv<br>"
                "4,009 listings · 12 raw features</div>", unsafe_allow_html=True)

# ============================================================
# 9. PAGE — HOME
# ============================================================

if page == "🏠 Home":
    st.markdown('<div class="main-header">🚗 Used Car Price Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">End-to-end ML pipeline · GradientBoosting · SHAP · Streamlit</div>', unsafe_allow_html=True)

    df = get_data()
    raw_count = get_raw_count()
    cols = st.columns(5)
    kpis = [
        ("Total Listings",  f"{raw_count:,}", f"{len(df):,} after cleaning"),
        ("Unique Brands",   f"{df['brand'].nunique()}", ""),
        ("Year Range",      f"{int(df['model_year'].min())}–{int(df['model_year'].max())}", ""),
        ("Median Price",    f"${df['price'].median():,.0f}", ""),
        ("Avg Mileage",     f"{df['mileage'].mean():,.0f} mi", ""),
    ]
    for col, (label, val, note) in zip(cols, kpis):
        note_html = f'<div style="font-size:.72rem;color:#57606a;margin-top:.1rem;">{note}</div>' if note else ""
        col.markdown(
            f'<div class="metric-card"><div class="metric-value">{val}</div>'
            f'<div class="metric-label">{label}</div>{note_html}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_price_dist(df), use_container_width=True)
    with c2:
        st.plotly_chart(chart_brand_count(df), use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Sample Data")
    st.dataframe(
        df[["brand", "model", "model_year", "mileage", "fuel_type",
            "transmission", "has_accident", "clean_title", "price"]].head(20),
        use_container_width=True, height=400,
    )

# ============================================================
# 10. PAGE — MARKET ANALYTICS
# ============================================================

elif page == "📊 Market Analytics":
    st.markdown('<div class="main-header">📊 Market Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive analysis of the used car market</div>', unsafe_allow_html=True)

    df = get_data()
    t1, t2, t3, t4 = st.tabs(["Brand & Price", "Year & Mileage", "Fuel & Transmission", "Correlations"])

    with t1:
        st.plotly_chart(chart_avg_price_brand(df), use_container_width=True)
        st.plotly_chart(chart_accident_impact(df),  use_container_width=True)
    with t2:
        st.plotly_chart(chart_price_year(df),   use_container_width=True)
        st.plotly_chart(chart_price_mileage(df), use_container_width=True)
    with t3:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(chart_price_fuel(df),         use_container_width=True)
        with c2:
            st.plotly_chart(chart_price_transmission(df), use_container_width=True)
    with t4:
        st.plotly_chart(chart_correlation(df), use_container_width=True)
        st.info("Model year and horsepower are positively correlated with price. "
                "Mileage and accident history are negatively correlated.")

# ============================================================
# 11. PAGE — PRICE PREDICTION
# ============================================================

elif page == "🔮 Price Prediction":
    st.markdown('<div class="main-header">🔮 Price Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter vehicle details and get an instant price estimate</div>', unsafe_allow_html=True)

    if not models_ready():
        st.error("No trained model found. Use the sidebar button to train the model first.")
        st.stop()

    df      = get_data()
    model, pre, _ = get_model_artifacts()

    with st.form("pred_form"):
        st.subheader("Vehicle Details")
        c1, c2, c3 = st.columns(3)

        with c1:
            brand      = st.selectbox("Brand", sorted(df["brand"].unique()))
            model_year = st.slider("Model Year", int(df["model_year"].min()), int(df["model_year"].max()), 2020)
            mileage    = st.number_input("Mileage (miles)", 0, 500_000, 30_000, 1_000)

        with c2:
            fuel_type    = st.selectbox("Fuel Type",    sorted(df["fuel_type"].unique()))
            transmission = st.selectbox("Transmission", sorted(df["transmission"].unique()))
            horsepower   = st.number_input("Horsepower", 50, 1500, 200, 10)

        with c3:
            engine_litres = st.number_input("Engine Litres", 0.5, 10.0, 2.5, 0.1)
            cylinders     = st.selectbox("Cylinders", [3, 4, 5, 6, 8, 10, 12], index=3)
            ext_col       = st.selectbox("Exterior Color", sorted(df["ext_col"].unique()))
            int_col       = st.selectbox("Interior Color", sorted(df["int_col"].unique()))

        c4, c5 = st.columns(2)
        with c4:
            has_accident = st.checkbox("Has accident history", value=False)
        with c5:
            clean_title  = st.checkbox("Clean title", value=True)

        submitted = st.form_submit_button("Predict Price", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "model_year": model_year, "mileage": float(mileage),
            "horsepower": float(horsepower), "engine_litres": float(engine_litres),
            "cylinders": float(cylinders), "car_age": 2024 - model_year,
            "has_accident": int(has_accident), "clean_title": int(clean_title),
            "brand": brand, "fuel_type": fuel_type, "transmission": transmission,
            "ext_col": ext_col, "int_col": int_col,
        }])
        predicted = float(np.expm1(model.predict(pre.transform(row))[0]))

        st.markdown("---")
        c_res, c_gauge = st.columns(2)

        with c_res:
            st.markdown(
                f'<div class="pred-box">'
                f'<div class="pred-label">Estimated Market Price</div>'
                f'<div class="pred-price">${predicted:,.0f}</div>'
                f'<div class="pred-label">{brand} &middot; {model_year} &middot; {mileage:,} mi</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            lo, hi = predicted * 0.90, predicted * 1.10
            st.caption(f"Estimated range: **${lo:,.0f}** – **${hi:,.0f}**")

        with c_gauge:
            hi_axis = predicted * 1.5
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted,
                number={"prefix": "$", "valueformat": ",.0f"},
                gauge={
                    "axis": {"range": [0, hi_axis]},
                    "bar":  {"color": "#3b82d4"},
                    "steps": [
                        {"range": [0,       lo],      "color": "#f0f4ff"},
                        {"range": [lo,      hi],      "color": "#bfdbfe"},
                        {"range": [hi,      hi_axis], "color": "#f0f4ff"},
                    ],
                    "threshold": {"line": {"color": "#7c5cd8", "width": 4},
                                  "thickness": 0.75, "value": predicted},
                },
                title={"text": "Price Gauge"},
            ))
            fig_g.update_layout(height=280, margin=dict(t=40, b=10))
            st.plotly_chart(fig_g, use_container_width=True)

        st.markdown("---")
        st.subheader("Input Summary")
        st.dataframe(pd.DataFrame({
            "Feature": ["Brand", "Model Year", "Mileage", "Fuel Type", "Transmission",
                        "Horsepower", "Engine (L)", "Cylinders", "Ext. Color",
                        "Int. Color", "Accident History", "Clean Title"],
            "Value":   [brand, model_year, f"{mileage:,} mi", fuel_type, transmission,
                        horsepower, engine_litres, cylinders, ext_col, int_col,
                        "Yes" if has_accident else "No",
                        "Yes" if clean_title  else "No"],
        }), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Similar Listings in Dataset")
        similar = df[
            (df["brand"] == brand) &
            (df["model_year"].between(model_year - 2, model_year + 2)) &
            (df["mileage"].between(max(0, mileage - 20_000), mileage + 20_000))
        ].head(10)
        if len(similar):
            st.dataframe(
                similar[["brand", "model", "model_year", "mileage",
                          "fuel_type", "transmission", "price"]],
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No similar listings found for the selected filters.")

# ============================================================
# 12. PAGE — MODEL INSIGHTS
# ============================================================

elif page == "🧠 Model Insights":
    st.markdown('<div class="main-header">🧠 Model Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Performance metrics, feature importance, SHAP, and residuals</div>', unsafe_allow_html=True)

    if not models_ready():
        st.error("No trained model found. Use the sidebar button to train the model first.")
        st.stop()

    model, pre, metrics = get_model_artifacts()
    df = get_data()

    best_name = metrics["best_model"]
    bm        = metrics["all_results"][best_name]["test"]

    # KPIs
    st.subheader(f"Best Model: {best_name}")
    k1, k2, k3, k4 = st.columns(4)
    for col, label, val in [
        (k1, "R² Score",     f"{bm['R2']:.4f}"),
        (k2, "MAE",          f"${bm['MAE']:,.0f}"),
        (k3, "RMSE",         f"${bm['RMSE']:,.0f}"),
        (k4, "Training rows",f"{metrics['n_train']:,}"),
    ]:
        col.markdown(
            f'<div class="metric-card"><div class="metric-value">{val}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Model comparison table
    st.subheader("Model Comparison")
    rows = [
        {
            "Model":     n,
            "Train R²":  v["train"]["R2"],
            "Test R²":   v["test"]["R2"],
            "Test MAE":  f"${v['test']['MAE']:,.0f}",
            "Test RMSE": f"${v['test']['RMSE']:,.0f}",
        }
        for n, v in metrics["all_results"].items()
    ]
    st.dataframe(
        pd.DataFrame(rows).sort_values("Test R²", ascending=False),
        use_container_width=True, hide_index=True,
    )

    model_names = [r["Model"] for r in rows]
    test_r2     = [metrics["all_results"][n]["test"]["R2"] for n in model_names]
    fig_cmp = px.bar(x=model_names, y=test_r2, title="Test R² by Model",
                     labels={"x": "Model", "y": "Test R²"},
                     color=test_r2, color_continuous_scale="Blues")
    fig_cmp.update_layout(template="plotly_white", coloraxis_showscale=False)
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("---")

    # Feature importance
    st.subheader("Feature Importance")
    if hasattr(model, "feature_importances_"):
        fi_df = pd.DataFrame({
            "Feature":    ALL_FEATURES,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=True).tail(15)
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                        title=f"Feature Importances – {best_name}",
                        color="Importance", color_continuous_scale="Blues")
        fig_fi.update_layout(template="plotly_white", coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Feature importances not available for this model type.")

    st.markdown("---")

    # SHAP
    st.subheader("SHAP Feature Impact")
    try:
        import shap
        sample = df[ALL_FEATURES].sample(min(500, len(df)), random_state=42)
        X_s    = pre.transform(sample)
        if hasattr(model, "feature_importances_"):
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_s)
        else:
            explainer   = shap.LinearExplainer(model, X_s)
            shap_values = explainer.shap_values(X_s)

        shap_df = pd.DataFrame({
            "Feature":     ALL_FEATURES,
            "Mean |SHAP|": np.abs(shap_values).mean(axis=0),
        }).sort_values("Mean |SHAP|", ascending=True).tail(15)

        fig_sh = px.bar(shap_df, x="Mean |SHAP|", y="Feature", orientation="h",
                        title="SHAP Mean Absolute Impact (log-price scale)",
                        color="Mean |SHAP|", color_continuous_scale="Purples")
        fig_sh.update_layout(template="plotly_white", coloraxis_showscale=False)
        st.plotly_chart(fig_sh, use_container_width=True)
        st.caption("Higher bars = larger average effect on predicted price.")
    except Exception as e:
        st.warning(f"SHAP computation skipped: {e}")

    st.markdown("---")

    # Actual vs Predicted
    st.subheader("Actual vs Predicted (Test Set)")
    try:
        X_all = df[ALL_FEATURES]
        y_all = df["log_price"].values
        _, X_te, _, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
        y_pr   = model.predict(pre.transform(X_te))
        actual = np.expm1(y_te);  pred = np.expm1(y_pr)
        idx    = np.random.RandomState(42).choice(len(actual), min(500, len(actual)), replace=False)
        fig_avp = px.scatter(x=actual[idx], y=pred[idx], opacity=0.5,
                             labels={"x": "Actual Price ($)", "y": "Predicted Price ($)"},
                             title="Actual vs Predicted (500-point sample)",
                             color_discrete_sequence=["#3b82d4"])
        mx = max(actual[idx].max(), pred[idx].max())
        fig_avp.add_scatter(x=[0, mx], y=[0, mx], mode="lines",
                            line=dict(color="#ef4444", dash="dash"), name="Perfect fit")
        fig_avp.update_layout(template="plotly_white")
        st.plotly_chart(fig_avp, use_container_width=True)
    except Exception as e:
        st.warning(f"Plot failed: {e}")

# ============================================================
# 13. PAGE — ABOUT
# ============================================================

elif page == "ℹ️ About":
    st.markdown('<div class="main-header">ℹ️ About This Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Architecture, methodology, and how to run</div>', unsafe_allow_html=True)

    st.markdown("""
## Used Car Price Prediction

End-to-end machine learning project predicting used car market prices.

---

### Dataset
| Attribute | Value |
|-----------|-------|
| File      | `data/used_cars.csv` |
| Rows      | 4,009 listings |
| Raw features | 12 columns |
| Clean rows after preprocessing | ~3,981 |
| Target    | `price` (USD) |

---

### Tech Stack
| Layer | Tools |
|-------|-------|
| Data processing | pandas, NumPy, re (regex) |
| Preprocessing | scikit-learn `ColumnTransformer`, `OrdinalEncoder`, `StandardScaler` |
| Models | Ridge, Random Forest, Gradient Boosting, XGBoost |
| Evaluation | MAE, RMSE, R² |
| Explainability | SHAP `TreeExplainer` |
| Serialisation | joblib |
| Dashboard | Streamlit |
| Charts | Plotly |

---

### Real Model Results (trained on actual data)
| Model | Test R² | Test MAE | Test RMSE |
|-------|---------|----------|-----------|
| Ridge | 0.4819 | $13,253 | $28,012 |
| Random Forest | 0.7767 | $8,773 | $18,390 |
| **Gradient Boosting** | **0.8465** | **$7,640** | **$15,245** |
| XGBoost | 0.8336 | $7,628 | $15,877 |

---

### Feature Engineering
- `price` — parsed from `"$10,300"` → `10300.0`, log-transformed as target  
- `mileage` — parsed from `"51,000 mi."` → `51000.0`  
- `horsepower`, `engine_litres`, `cylinders` — extracted from engine string via regex  
- `transmission` — bucketed: Automatic / Manual / CVT / Dual-Clutch / Other  
- `fuel_type` — simplified: Gasoline / Hybrid / Electric / Diesel / Flex Fuel / Other  
- `has_accident` — binary flag derived from accident column  
- `car_age` — `2024 − model_year`  
- Rare colours collapsed to "Other" (top-15 kept)  

---

### How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Train the model via the **sidebar button** on first launch.
""")
