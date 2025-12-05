#!/usr/bin/env python3
"""
streamlit_dashboard.py — Coater Loss Analysis
Features: Theoretical Pie Breakdown, Dropdown Diagnostics, Enhanced Time Series.
"""
from __future__ import annotations
import os
import sys
import json
from io import BytesIO
from datetime import datetime, date, timedelta
from pathlib import Path
import traceback
import sqlite3

import streamlit as st

# --- 1. CONFIGURATION MUST BE FIRST ---
st.set_page_config(
    page_title="Coater Machine - Loss Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏭"
)

import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- 2. OPTIONAL LIBRARIES ---
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import cloudpickle
    HAS_CLOUDPICKLE = True
except ImportError:
    HAS_CLOUDPICKLE = False

# --- 3. PATHS & DEFAULTS ---
from model_connection import CombinedModel, MACHINE_MAX_WIDTH
BASE_DIR = Path(os.environ.get("COATER_BASE_DIR", Path.cwd()))
ARTIFACTS_DIR = Path(os.environ.get("COATER_ARTIFACTS_DIR", BASE_DIR / "artifacts"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = Path(os.environ.get("COATER_DB_PATH", BASE_DIR / "predictions.db"))
DEFAULT_BUNDLE = os.environ.get("COATER_DEFAULT_BUNDLE", str(ARTIFACTS_DIR / "combined_loss_model.joblib"))
LAST_PREDICTIONS_PATH = ARTIFACTS_DIR / "last_predictions.csv"
LOGO_PATH = BASE_DIR / "logo.png"

DEFAULTS = {
    "COLUMN_MAP": {
        "prod_track": "PROD_TRACK", "turnup_time": "TURNUP_TIME", "base_gsm": "BASE_GSM",
        "base_width": "BASE_WIDTH", "base_wgtscaled": "BASE_WGTSCALED", "base_track": "BASE_TRACK",
        "prod_gsm": "PROD_GSM", "prod_wgt": "PROD_WGT", "production_date": "PRODUCTION_DATE",
    },
    "DEFAULT_BUNDLE": DEFAULT_BUNDLE,
    "MACHINE_MAX_WIDTH": MACHINE_MAX_WIDTH
}

DEFAULT_CONFIG_PATH = Path(os.environ.get("COATER_ADMIN_CONFIG", BASE_DIR / "admin_config.json"))

# --- 4. ADMIN & UTILS ---
def load_admin_config():
    cfg = {"COLUMN_MAP": DEFAULTS["COLUMN_MAP"].copy(), "DEFAULT_BUNDLE": DEFAULTS["DEFAULT_BUNDLE"], "MACHINE_MAX_WIDTH": DEFAULTS["MACHINE_MAX_WIDTH"]}
    for k in ["MSSQL_HOST","MSSQL_PORT","MSSQL_USER","MSSQL_PASSWORD","MSSQL_DATABASE","PROD_TABLE"]:
        cfg[k] = os.environ.get(f"COATER_{k}", "")
    if DEFAULT_CONFIG_PATH.exists():
        try:
            dj = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(dj, dict):
                if "COLUMN_MAP" in dj: cfg["COLUMN_MAP"].update(dj["COLUMN_MAP"])
                if "DEFAULT_BUNDLE" in dj: cfg["DEFAULT_BUNDLE"] = dj["DEFAULT_BUNDLE"]
                for k in ["MSSQL_HOST","MSSQL_PORT","MSSQL_USER","MSSQL_PASSWORD","MSSQL_DATABASE","PROD_TABLE"]:
                    if k in dj: cfg[k] = dj[k]
        except Exception: pass
    return cfg

def save_admin_config(cfg: dict):
    try:
        DEFAULT_CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return True, None
    except Exception as e: return False, str(e)

ADMIN_CFG = load_admin_config()

def find_col_in_df(df_cols, possible_names):
    if df_cols is None: return None
    col_lower_map = {str(c).lower(): c for c in df_cols}
    for p in possible_names:
        if p in df_cols: return p
        pl = str(p).lower()
        if pl in col_lower_map: return col_lower_map[pl]
    for p in possible_names:
        token = str(p).lower()
        for c in df_cols:
            if token in str(c).lower(): return c
    return None

def df_to_excel_bytes(df: pd.DataFrame):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    buf.seek(0)
    return buf

# --- 5. THEME & STYLING ---
def apply_theme(theme_name):
    if theme_name == "Dark":
        st.markdown("""
        <style>
        [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background-color: #000000; color: #e6eef6; }
        [data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid #333; }
        h1, h2, h3, h4, h5, h6, label { color: #ffffff !important; }
        p, div, span { color: #e6eef6; }
        .stTextInput > div > div > input { color: #ffffff; background-color: #222; border: 1px solid #444; }
        .stSelectbox > div > div > div { color: #ffffff; background-color: #222; }
        .stDateInput > div > div > input { color: #ffffff; background-color: #222; }
        [data-testid="stDataFrame"] { background-color: #111111; border: 1px solid #333; }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { color: #e6eef6; }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] { border-bottom-color: #00B4D8; }
        .stButton>button { background-color: #00B4D8; color: #000000; border: none; font-weight: bold; border-radius: 6px; }
        .stButton>button:hover { background-color: #0096B4; color: #ffffff; }
        /* Logo Container Adjustments */
        [data-testid="stImage"] { margin-top: 10px; }
        </style>
        """, unsafe_allow_html=True)
        plt.style.use("dark_background")
    else:
        st.markdown("""
        <style>
        [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background-color: #ffffff; color: #000000; }
        [data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #ddd; }
        h1, h2, h3, h4, h5, h6, label { color: #000000 !important; }
        p, div, span { color: #212529; }
        .stTextInput > div > div > input { color: #000000; background-color: #ffffff; border: 1px solid #ced4da; }
        .stSelectbox > div > div > div { color: #000000; background-color: #ffffff; }
        .stDateInput > div > div > input { color: #000000; background-color: #ffffff; }
        [data-testid="stDataFrame"] { background-color: #ffffff; border: 1px solid #eee; }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { color: #333; }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] { border-bottom-color: #00B4D8; }
        .stButton>button { background-color: #00B4D8; color: #ffffff; border: none; font-weight: bold; border-radius: 6px; }
        .stButton>button:hover { background-color: #008CA8; }
        </style>
        """, unsafe_allow_html=True)
        plt.style.use("default")

# --- CUSTOM METRIC DISPLAY ---
def display_custom_metric(label, value, baseline, theme_mode, col):
    delta = value - baseline
    if delta < 0:
        color = "#00E676" if theme_mode == "Dark" else "#008000"
        arrow = "↓"
    elif delta > 0:
        color = "#FF5252" if theme_mode == "Dark" else "#D32F2F"
        arrow = "↑"
    else:
        color = "#B0BEC5" if theme_mode == "Dark" else "#607D8B"
        arrow = "—"
    
    delta_str = "No history" if baseline == 0 else f"{abs(delta):.2f} vs avg"

    with col:
        st.markdown(f"""
        <div style="background-color: rgba(255,255,255,0.02); padding: 10px; border-radius: 5px; border-left: 3px solid {color};">
            <p style="font-size: 14px; margin-bottom: 2px; opacity: 0.8; font-weight: 500;">{label}</p>
            <p style="font-size: 32px; font-weight: 700; margin: 0; line-height: 1.2; color: {color};">
                {value:,.2f} <span style="font-size: 16px; color: inherit;">kg {arrow}</span>
            </p>
            <p style="font-size: 12px; margin: 0; opacity: 0.6; padding-top: 4px;">
                {delta_str}
            </p>
        </div>
        """, unsafe_allow_html=True)

# --- 6. MODEL CLASSES & LOADERS ---
from dataclasses import dataclass, field
from typing import Dict, List



@st.cache_resource(show_spinner=False)
def load_bundle(path):
    if not path or not os.path.exists(path): return None
    try: return joblib.load(path)
    except: pass
    try: 
        with open(path, "rb") as fh: return pickle.load(fh)
    except: pass
    if HAS_CLOUDPICKLE:
        try: 
            with open(path, "rb") as fh: return cloudpickle.load(fh)
        except: pass
    return None

def load_bundle_diagnostics(path):
    diag = {"path": str(path) if path else None, "exists": False, "size_bytes": None, "attempts": []}
    if not path:
        diag["attempts"].append("No path provided")
        return diag
    p = Path(path)
    diag["exists"] = p.exists()
    if diag["exists"]:
        try: diag["size_bytes"] = p.stat().st_size
        except: pass
    return diag

# --- 7. DATABASE & LOGIC ---
def append_predictions_to_sqlite(df: pd.DataFrame, db_path: Path = DB_PATH, table: str = "predictions"):
    df = df.copy()
    if "Time" not in df.columns: df["Time"] = datetime.now().isoformat()
    if "TURNUP_TIME" in df.columns: df["TURNUP_TIME"] = df["TURNUP_TIME"].astype(str)
    
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
             df[c] = df[c].apply(lambda x: x.isoformat() if pd.notna(x) else None)
        elif pd.api.types.is_numeric_dtype(df[c]):
             df[c] = df[c].apply(lambda x: float(x) if pd.notna(x) else None)
        else:
             df[c] = df[c].astype(str)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cols = list(df.columns)
    cols_def = ", ".join([f'"{c}" TEXT' for c in cols])
    cur.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({cols_def})')
    cur.execute(f'PRAGMA table_info("{table}")')
    existing_cols = [r[1].lower() for r in cur.fetchall()]
    for c in cols:
        if c.lower() not in existing_cols:
            try: cur.execute(f'ALTER TABLE "{table}" ADD COLUMN "{c}" TEXT')
            except: pass
    conn.commit()
    placeholders = ", ".join(["?"] * len(cols))
    quoted_cols = ", ".join([f'"{c}"' for c in cols])
    rows = []
    for row in df[cols].to_numpy(): rows.append([None if pd.isna(x) else x for x in row])
    try:
        cur.executemany(f'INSERT INTO "{table}" ({quoted_cols}) VALUES ({placeholders})', rows)
        conn.commit()
    except Exception: pass
    finally: conn.close()

def read_app_logs(limit=5000):
    try:
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query(f"SELECT * FROM predictions ORDER BY Time DESC LIMIT {int(limit)}", conn)
        conn.close()
        for c in ["LOSS","UNAVOIDABLE_LOSS","AVOIDABLE_LOSS","Total_Loss","Unavoidable_Loss","Avoidable_Loss"]:
            col_match = find_col_in_df(df.columns, [c])
            if col_match: df[col_match] = pd.to_numeric(df[col_match], errors="coerce")
        return df
    except Exception: return pd.DataFrame()

def perform_calculations(df, cfg):
    df_calc = df.copy()
    cmap = cfg.get("COLUMN_MAP", {})
    col_map = {}
    for key, val in cmap.items():
        found = find_col_in_df(df.columns, [val])
        if found: col_map[found] = key.upper()
    if col_map: df_calc = df_calc.rename(columns=col_map)
    
    if "BASE_GSM" not in df_calc.columns:
        gsm_col = find_col_in_df(df_calc.columns, ["GSM", "Base_GSM", "base_gsm", "Target_GSM"])
        if gsm_col: df_calc["BASE_GSM"] = df_calc[gsm_col]
    if "BASE_WIDTH" not in df_calc.columns:
        width_col = find_col_in_df(df_calc.columns, ["BASE_WIDTH", "Deckle", "Deckle_cm", "Width"])
        if width_col: df_calc["BASE_WIDTH"] = df_calc[width_col]
        
    num_cols = ["BASE_GSM", "BASE_WGTSCALED", "PROD_GSM", "PROD_WGT"]
    for c in num_cols:
        if c in df_calc.columns: df_calc[c] = pd.to_numeric(df_calc[c], errors='coerce').fillna(0.0)
    
    missing = [c for c in num_cols if c not in df_calc.columns]
    if missing: st.toast(f"⚠️ Missing columns: {missing}. Loss calc may fail.", icon="⚠️")

    if "PROD_GSM" in df_calc and "BASE_GSM" in df_calc:
        df_calc["COATWEIGHT"] = df_calc["PROD_GSM"] - df_calc["BASE_GSM"]
    if "BASE_WGTSCALED" in df_calc and "BASE_GSM" in df_calc:
        df_calc["AREA"] = df_calc.apply(lambda r: r["BASE_WGTSCALED"]/r["BASE_GSM"] if r["BASE_GSM"]>0 else 0, axis=1)
    if "AREA" in df_calc and "COATWEIGHT" in df_calc:
        df_calc["CHEMICAL_WEIGHT"] = df_calc["AREA"] * df_calc["COATWEIGHT"]
    if "BASE_WGTSCALED" in df_calc and "CHEMICAL_WEIGHT" in df_calc:
        df_calc["THEORETICAL_WEIGHT"] = df_calc["BASE_WGTSCALED"] + df_calc["CHEMICAL_WEIGHT"]
    if "THEORETICAL_WEIGHT" in df_calc and "PROD_WGT" in df_calc:
        df_calc["LOSS"] = df_calc["THEORETICAL_WEIGHT"] - df_calc["PROD_WGT"]
    
    if "LOSS" not in df_calc.columns or df_calc["LOSS"].isna().all():
        existing_loss = find_col_in_df(df_calc.columns, ["Total Loss", "Loss", "LOSS_KG"])
        if existing_loss: df_calc["LOSS"] = pd.to_numeric(df_calc[existing_loss], errors='coerce')
    return df_calc

def load_production_rows_mssql(start_date, end_date, cfg):
    import pymssql
    start_str = start_date.isoformat(); end_str = end_date.isoformat()
    conn = pymssql.connect(
        server=cfg.get("MSSQL_HOST"), port=int(cfg.get("MSSQL_PORT", 1433)),
        user=cfg.get("MSSQL_USER"), password=cfg.get("MSSQL_PASSWORD"),
        database=cfg.get("MSSQL_DATABASE"), timeout=15
    )
    try:
        t_col = cfg.get('COLUMN_MAP',{}).get('turnup_time','TURNUP_TIME')
        q = f"SELECT * FROM {cfg.get('PROD_TABLE')} WHERE [{t_col}] BETWEEN %s AND %s"
        return pd.read_sql_query(q, conn, params=(start_str, end_str))
    finally: conn.close()

# ==============================================================================
#  MAIN UI STRUCTURE
# ==============================================================================

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.title("🎛️ Controls")
theme = st.sidebar.radio("App Theme", ["Dark", "Light"], horizontal=True)
apply_theme(theme)

st.sidebar.markdown("---")
st.sidebar.subheader("Data Source")
data_source = st.sidebar.radio("Select Source", ["OptiVision Database", "CSV File"], index=0)

cfg = ADMIN_CFG.copy()
prod_df = pd.DataFrame()
run_predict = False

if data_source == "OptiVision Database":
    today = date.today()
    d_range = st.sidebar.date_input("Production Date Range", value=(today - timedelta(days=1), today))
    if st.sidebar.button("🚀 Load Data (OptiVision)", type="primary"):
        run_predict = True
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV/XLSX", type=['csv', 'xlsx'])
    if uploaded_file and st.sidebar.button("🚀 Load Data (CSV)", type="primary"):
        run_predict = True

# --- QUICK SIMULATOR (SIDEBAR) ---
st.sidebar.markdown("---")
with st.sidebar.expander("🧪 Quick Simulator", expanded=False):
    st.write("Adjust sliders to simulate loss.")
    sim_gsm = st.slider("GSM", 50, 300, 150)
    sim_width = st.slider("Width (cm)", 50, 200, 100)
    
    bundle = load_bundle(cfg.get("DEFAULT_BUNDLE", DEFAULTS["DEFAULT_BUNDLE"]))
    if bundle:
        sim_df = pd.DataFrame([{"BASE_GSM": sim_gsm, "BASE_WIDTH": sim_width, "Diameter": 0}])
        try:
            sim_pred = bundle.predict_df(sim_df)
            stages = ["PACKING_WEIGHT", "TOP_LAYER_LOSS", "CORE_END_LOSS", "SPLICE_LOSS"]
            for k in sim_pred.columns:
                if "packing" in k.lower(): sim_pred = sim_pred.rename(columns={k: "PACKING_WEIGHT"})
                if "top" in k.lower(): sim_pred = sim_pred.rename(columns={k: "TOP_LAYER_LOSS"})
                if "core" in k.lower(): sim_pred = sim_pred.rename(columns={k: "CORE_END_LOSS"})
                if "splice" in k.lower(): sim_pred = sim_pred.rename(columns={k: "SPLICE_LOSS"})
            
            for s in stages: 
                if s not in sim_pred: sim_pred[s] = 0.0
            total_sim = sim_pred[stages].sum(axis=1).iloc[0]
            st.metric("Predicted Unavoidable", f"{total_sim:.2f} kg")
        except: st.error("Model Error")
    else:
        st.warning("Model not loaded")

# --- MAIN TITLE & HEADER ---
col_title, col_logo = st.columns([5, 1])
with col_title:
    st.title("Coater Machine - Loss Analysis")
with col_logo:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width='stretch')

tabs = st.tabs(["📊 Data Analysis", "📈 Diagnostics & Metrics", "📝 Logs", "⚙️ Model Ops", "🔒 Admin"])

# --- TAB 1: DATA ANALYSIS --
with tabs[0]:
    st.header("Data Analysis")
    
    m1, m2, m3 = st.columns(3)
    
    if run_predict:
        with st.spinner("Processing..."):
            bundle = load_bundle(cfg.get("DEFAULT_BUNDLE", DEFAULTS["DEFAULT_BUNDLE"]))
            if not bundle:
                st.error("Model bundle not found. Check Model Ops.")
            else:
                try:
                    # 1. Fetch Historical Baseline (Avg of last 1000 logs)
                    hist_df = read_app_logs(1000)
                    hist_total = 0.0; hist_unav = 0.0; hist_avoi = 0.0
                    
                    if not hist_df.empty:
                        hc_loss = find_col_in_df(hist_df.columns, ["LOSS", "Total_Loss"])
                        hc_unav = find_col_in_df(hist_df.columns, ["UNAVOIDABLE_LOSS", "Unavoidable_Loss"])
                        hc_avoi = find_col_in_df(hist_df.columns, ["AVOIDABLE_LOSS", "Avoidable_Loss"])
                        
                        if hc_loss: hist_total = hist_df[hc_loss].mean()
                        if hc_unav: hist_unav = hist_df[hc_unav].mean()
                        if hc_avoi: hist_avoi = hist_df[hc_avoi].mean()

                    # 2. LOAD & CALCULATE
                    if data_source == "OptiVision Database":
                        prod_df = load_production_rows_mssql(d_range[0], d_range[1], cfg)
                    elif data_source == "CSV File":
                        if uploaded_file.name.endswith('.csv'): prod_df = pd.read_csv(uploaded_file)
                        else: prod_df = pd.read_excel(uploaded_file)
                    
                    if not prod_df.empty:
                        df_calc = perform_calculations(prod_df, cfg)
                        try:
                            preds = bundle.predict_df(df_calc)
                            pred_map = {"packing": "PACKING_WEIGHT", "top_layer": "TOP_LAYER_LOSS", "core_end": "CORE_END_LOSS", "splice": "SPLICE_LOSS"}
                            for k in preds.columns:
                                std_k = k.lower()
                                for pk, pv in pred_map.items():
                                    if pk in std_k: df_calc[pv] = preds[k]
                            
                            stages = ["PACKING_WEIGHT", "TOP_LAYER_LOSS", "CORE_END_LOSS", "SPLICE_LOSS"]
                            for s in stages: 
                                if s not in df_calc: df_calc[s] = 0.0
                                
                            df_calc["UNAVOIDABLE_LOSS"] = df_calc[stages].sum(axis=1)
                            if "LOSS" in df_calc:
                                # Force Positive Loss
                                df_calc["LOSS"] = df_calc["LOSS"].abs()
                                # Force Non-Negative Avoidable
                                raw_avoidable = df_calc["LOSS"] - df_calc["UNAVOIDABLE_LOSS"]
                                df_calc["AVOIDABLE_LOSS"] = raw_avoidable.clip(lower=0.0)
                            else:
                                df_calc["AVOIDABLE_LOSS"] = np.nan
                                
                            append_predictions_to_sqlite(df_calc)
                            df_calc.to_csv(LAST_PREDICTIONS_PATH, index=False)
                            
                            # 3. DISPLAY CUSTOM METRICS
                            avg_total_curr = df_calc['LOSS'].mean() if "LOSS" in df_calc else 0
                            avg_unav_curr = df_calc['UNAVOIDABLE_LOSS'].mean()
                            avg_avoi_curr = df_calc['AVOIDABLE_LOSS'].mean() if "AVOIDABLE_LOSS" in df_calc else 0
                            
                            display_custom_metric("Total Loss (Avg)", avg_total_curr, hist_total, theme, m1)
                            display_custom_metric("Unavoidable Loss (Avg)", avg_unav_curr, hist_unav, theme, m2)
                            display_custom_metric("Avoidable Loss (Avg)", avg_avoi_curr, hist_avoi, theme, m3)
                            
                            st.subheader("Detailed Data")
                            st.dataframe(df_calc.head(100), height=300)
                            
                            ts = datetime.now().strftime("%Y%m%d")
                            st.download_button("💾 Download Results CSV", df_calc.to_csv(index=False).encode('utf-8'), f"predictions_{ts}.csv")
                            
                        except Exception as e:
                            st.error(f"Prediction Error: {e}")
                            st.write(traceback.format_exc())
                    else:
                        st.warning("No data returned from source.")
                except Exception as e:
                    st.error(f"Data Load Error: {e}")
    else:
        # Default view (using metrics only, color neutral)
        logs_df = read_app_logs(1000)
        avg_total = 0.0
        avg_unav = 0.0
        avg_avoi = 0.0
        if not logs_df.empty:
            total_col = find_col_in_df(logs_df.columns, ["LOSS", "Total_Loss"])
            unav_col = find_col_in_df(logs_df.columns, ["UNAVOIDABLE_LOSS", "Unavoidable_Loss"])
            avoi_col = find_col_in_df(logs_df.columns, ["AVOIDABLE_LOSS", "Avoidable_Loss"])
            if total_col: avg_total = logs_df[total_col].mean()
            if unav_col: avg_unav = logs_df[unav_col].mean()
            if avoi_col: avg_avoi = logs_df[avoi_col].mean()
        
        display_custom_metric("Avg Total Loss", avg_total, avg_total, theme, m1)
        display_custom_metric("Avg Unavoidable Loss", avg_unav, avg_unav, theme, m2)
        display_custom_metric("Avg Avoidable Loss", avg_avoi, avg_avoi, theme, m3)
        st.info("👈 Use the Sidebar to Load Data & Predict")

# --- TAB 2: DIAGNOSTICS ---
with tabs[1]:
    st.header("📊 Diagnostics & Metrics")
    
    col_up, col_act = st.columns([2,1])
    uploaded_diag = col_up.file_uploader("Upload Analysis File", type=['csv','xlsx'])
    
    with col_act:
        st.write("Or load from:")
        if st.button("📂 Local DB Logs"):
            st.session_state['diag_df'] = read_app_logs(5000)
            st.session_state['diag_src'] = "SQLite"
        if st.button("🔄 Last Run"):
            if LAST_PREDICTIONS_PATH.exists():
                st.session_state['diag_df'] = pd.read_csv(LAST_PREDICTIONS_PATH)
                st.session_state['diag_src'] = "Last Saved"

    if uploaded_diag:
        try:
            st.session_state['diag_df'] = pd.read_csv(uploaded_diag) if uploaded_diag.name.endswith('.csv') else pd.read_excel(uploaded_diag)
            st.session_state['diag_src'] = "Upload"
        except: pass

    # Ensure session state exists
    if 'diag_df' not in st.session_state:
        st.session_state['diag_df'] = pd.DataFrame()
    
    # Check if Compute triggered (persisted across reruns)
    if "metrics_computed" not in st.session_state:
        st.session_state["metrics_computed"] = False

    if not st.session_state['diag_df'].empty:
        df_diag = st.session_state['diag_df']
        st.success(f"Loaded {len(df_diag)} rows from {st.session_state.get('diag_src', '?')}")
        
        if st.button("🚀 Compute Metrics", type="primary"):
            st.session_state["metrics_computed"] = True
            
        # Only show content if computed flag is True
        if st.session_state["metrics_computed"]:
            
            # Normalize Columns
            req_cols = ["LOSS", "UNAVOIDABLE_LOSS", "AVOIDABLE_LOSS", "PACKING_WEIGHT", "TOP_LAYER_LOSS", "CORE_END_LOSS", "SPLICE_LOSS", "PROD_WGT"]
            for c in req_cols:
                col_name = find_col_in_df(df_diag.columns, [c, c.lower(), c.title()])
                if col_name: df_diag[c] = pd.to_numeric(df_diag[col_name], errors='coerce').fillna(0.0)
                else: df_diag[c] = 0.0
            
            # Force Positive Loss & Non-Negative Avoidable
            df_diag["LOSS"] = df_diag["LOSS"].abs()
            df_diag["AVOIDABLE_LOSS"] = df_diag["AVOIDABLE_LOSS"].clip(lower=0.0)

            # DROPDOWN FOR ANALYSIS SELECTION
            view_mode = st.selectbox("Select Analysis View", 
                ["📊 KPI Overview", "🍰 Loss Breakdown (Pie Chart)", "📦 Distribution Analysis", "📈 Time Series Analysis"]
            )
            
            st.markdown("---")

            # 1. KPI OVERVIEW
            if view_mode == "📊 KPI Overview":
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total Loss", f"{df_diag['LOSS'].sum():,.0f} kg")
                k2.metric("Unavoidable", f"{df_diag['UNAVOIDABLE_LOSS'].sum():,.0f} kg")
                k3.metric("Avoidable", f"{df_diag['AVOIDABLE_LOSS'].sum():,.0f} kg")
                pct = (df_diag['AVOIDABLE_LOSS'].sum() / df_diag['LOSS'].sum() * 100) if df_diag['LOSS'].sum() > 0 else 0
                k4.metric("Avoidable %", f"{pct:.1f}%")

            # 2. PIE CHART (THEORETICAL WEIGHT BREAKDOWN)
            elif view_mode == "🍰 Loss Breakdown (Pie Chart)":
                st.subheader("Theoretical Weight Breakdown")
                # Summing components
                actual_wgt = df_diag["PROD_WGT"].sum()
                avoidable_loss = df_diag["AVOIDABLE_LOSS"].sum()
                unavoidable_total = df_diag["UNAVOIDABLE_LOSS"].sum()
                
                packing = df_diag["PACKING_WEIGHT"].sum()
                top_layer = df_diag["TOP_LAYER_LOSS"].sum()
                core_end = df_diag["CORE_END_LOSS"].sum()
                splice = df_diag["SPLICE_LOSS"].sum()
                
                theoretical_sum = actual_wgt + avoidable_loss + packing + top_layer + core_end + splice
                
                # Chart Values (Detailed for Pie)
                pie_labels = ["Actual Product", "Avoidable Loss", "Packing", "Top Layer", "Core End", "Splice"]
                pie_values = [actual_wgt, avoidable_loss, packing, top_layer, core_end, splice]
                colors = ['#00E676', '#FF5252', '#29B6F6', '#FF7043', '#AB47BC', '#FFA726']
                
                c1, c2 = st.columns([1.5, 1])
                
                with c1:
                    if HAS_PLOTLY:
                        pl_theme = "plotly_dark" if theme == "Dark" else "plotly_white"
                        fig = px.pie(values=pie_values, names=pie_labels, hole=0.4, color_discrete_sequence=colors, template=pl_theme)
                        fig.update_layout(height=350, margin=dict(t=0,b=0,l=0,r=0), paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, width='stretch')
                
                with c2:
                    # Table Logic: Summary vs Detailed Toggle
                    show_details = st.toggle("Show Detailed Unavoidable Breakdown", value=False)
                    
                    if show_details:
                        tbl_data = {
                            "Category": ["Actual Product", "Avoidable Loss", "Packing", "Top Layer", "Core End", "Splice"],
                            "Total Weight (kg)": [actual_wgt, avoidable_loss, packing, top_layer, core_end, splice]
                        }
                    else:
                        tbl_data = {
                            "Category": ["Actual Product", "Avoidable Loss", "Unavoidable (Total)"],
                            "Total Weight (kg)": [actual_wgt, avoidable_loss, unavoidable_total]
                        }
                        
                    breakdown_df = pd.DataFrame(tbl_data)
                    breakdown_df["% Contribution"] = (breakdown_df["Total Weight (kg)"] / theoretical_sum * 100)
                    st.dataframe(breakdown_df.style.format({"Total Weight (kg)": "{:,.2f}", "% Contribution": "{:.2f}%"}), width='stretch')

            # 3. DISTRIBUTION ANALYSIS
            elif view_mode == "📦 Distribution Analysis":
                st.subheader("Distribution Analysis")
                
                # Detail Level Selection
                detail_level = st.radio("Detail Level", ["Summary", "Detailed Statistics"], horizontal=True)
                
                box_cols = st.columns(3)
                metrics = ["LOSS", "UNAVOIDABLE_LOSS", "AVOIDABLE_LOSS"]
                names = ["Total Loss", "Unavoidable", "Avoidable"]
                
                for i, m in enumerate(metrics):
                    d = df_diag[m]
                    q1, q3 = d.quantile(0.25), d.quantile(0.75)
                    outliers = len(d[(d < (q1 - 1.5 * (q3-q1))) | (d > (q3 + 1.5 * (q3-q1)))])
                    with [box_cols[0], box_cols[1], box_cols[2]][i]:
                        st.markdown(f"**{names[i]}**")
                        st.caption(f"Mean: {d.mean():.2f} kg | Outliers: {outliers}")
                        if HAS_PLOTLY:
                            pl_theme = "plotly_dark" if theme == "Dark" else "plotly_white"
                            fig_box = px.box(y=d, points="outliers", template=pl_theme)
                            fig_box.update_layout(yaxis_title="kg", margin=dict(l=20,r=20,t=10,b=10), height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                            st.plotly_chart(fig_box, width='stretch')
                
                st.markdown("### Statistical Summary Table")
                if detail_level == "Summary":
                    stats_df = df_diag[metrics].agg(['mean', 'median', 'sum']).T
                    stats_df.columns = ['Mean', 'Median', 'Total Sum']
                    st.dataframe(stats_df.style.format("{:,.2f}"))
                else:
                    stats_df = df_diag[metrics].describe().T
                    st.dataframe(stats_df.style.format("{:,.2f}"))

            # 4. TIME SERIES ANALYSIS (ENHANCED)
            elif view_mode == "📈 Time Series Analysis":
                st.subheader("Time Series Analysis")
                date_col = find_col_in_df(df_diag.columns, ["TURNUP_TIME", "Production_Date", "Time"])
                if date_col and HAS_PLOTLY:
                    try:
                        df_diag[date_col] = pd.to_datetime(df_diag[date_col])
                        df_ts = df_diag.sort_values(by=date_col)
                        
                        # Add markers, fill area
                        pl_theme = "plotly_dark" if theme == "Dark" else "plotly_white"
                        fig_ts = px.area(df_ts, x=date_col, y="LOSS", title=f"Loss Trend ({date_col})", template=pl_theme, markers=True)
                        
                        # Enhance layout
                        fig_ts.update_traces(line_color='#00B4D8', fillcolor='rgba(0, 180, 216, 0.1)')
                        fig_ts.update_layout(
                            hovermode="x unified",
                            height=500,
                            paper_bgcolor='rgba(0,0,0,0)', 
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        fig_ts.update_xaxes(
                            rangeslider_visible=True,
                            rangeselector=dict(
                                buttons=list([
                                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                                    dict(count=1, label="1d", step="day", stepmode="backward"),
                                    dict(count=7, label="1w", step="day", stepmode="backward"),
                                    dict(step="all")
                                ])
                            )
                        )
                        st.plotly_chart(fig_ts, width='stretch')
                    except Exception as e:
                        st.info(f"Could not parse date column '{date_col}'.")
                else:
                    st.info("No valid date column found.")

            # Export Button (Always Visible)
            st.divider()
            st.download_button("📥 Export Diagnostics CSV", df_diag.to_csv(index=False).encode('utf-8'), f"diagnostics_{datetime.now()}.csv")

# --- TAB 3: LOGS ---
with tabs[2]:
    st.header("📝 Logs")
    st.dataframe(read_app_logs(5000), width='stretch')
    if st.button("Clear All Logs"):
        try: os.remove(DB_PATH)
        except: pass
        st.rerun()

# --- TAB 4: MODEL OPS ---
with tabs[3]:
    st.header("⚙️ Model Ops — Load / Diagnostics / Upload / Retrain")
    
    # Section 1: Default & Test
    st.caption("Default bundle used by the app:")
    st.code(ADMIN_CFG.get("DEFAULT_BUNDLE"))
    test_path = st.text_input("Bundle path to test (leave blank to use DEFAULT)", value=ADMIN_CFG.get("DEFAULT_BUNDLE"))
    c1, c2 = st.columns([1,5])
    if c1.button("Load bundle (test)"):
        b = load_bundle(test_path)
        if b: st.success("Bundle Loaded Successfully")
        else: st.error("Failed to load")
    if c2.button("Run diagnostics"):
        diag = load_bundle_diagnostics(test_path)
        st.json(diag)

    st.markdown("---")

    # Section 2: Upload
    st.subheader("Upload new joblib/pkl bundle")
    up_model = st.file_uploader("Upload .joblib/.pkl", type=["joblib","pkl"])
    save_path = st.text_input("Save uploaded to (optional)", value=str(ARTIFACTS_DIR / f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"))
    
    if up_model and save_path:
        with open(save_path, "wb") as f:
            f.write(up_model.read())
        ADMIN_CFG["DEFAULT_BUNDLE"] = save_path
        save_admin_config(ADMIN_CFG)
        st.success(f"Saved to {save_path} and set as default.")

    st.markdown("---")

    # Section 3: Retrain
    st.subheader("Retrain quick Ridge models from CSV (admin)")
    train_file = st.file_uploader("Upload CSV/XLSX for retrain", type=["csv","xlsx"])
    retrain_out = st.text_input("Save retrained combined to", value=str(ARTIFACTS_DIR / f"retrained_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"))
    
    if train_file and st.button("Start retrain (quick)"):
        try:
            if train_file.name.endswith(".csv"):
                train_df = pd.read_csv(train_file)
            else:
                train_df = pd.read_excel(train_file)
            
            col_map = {}
            for c in train_df.columns:
                lc = c.lower()
                if "gsm" in lc and "base" in lc: col_map[c] = "GSM"
                elif "gsm" in lc: col_map[c] = "GSM"
                if "width" in lc or "deckle" in lc: col_map[c] = "Deckle"
                if "pack" in lc: col_map[c] = "Packing_Weight_kg"
                if "top" in lc: col_map[c] = "Top_Layer_Loss_kg"
                if "core" in lc: col_map[c] = "Core_End_Loss_kg"
                if "splice" in lc: col_map[c] = "Splice_Loss_kg"
            
            train_df = train_df.rename(columns=col_map)
            
            from sklearn.preprocessing import PolynomialFeatures, StandardScaler
            from sklearn.linear_model import RidgeCV
            
            X = train_df[["GSM", "Deckle"]].values
            poly = PolynomialFeatures(degree=2)
            Xp = poly.fit_transform(X)
            scaler = StandardScaler().fit(Xp)
            Xs = scaler.transform(Xp)
            
            models = {}
            for t in ["Packing_Weight_kg", "Top_Layer_Loss_kg", "Core_End_Loss_kg", "Splice_Loss_kg"]:
                y = train_df[t].values
                m = RidgeCV().fit(Xs, y)
                models[t.replace("_kg","").upper()] = m
                
            bundle = {
                "poly": poly,
                "scaler": scaler,
                "models": models,
                "machine_max_width": 172.0
            }
            joblib.dump(bundle, retrain_out)
            st.success(f"Retrained and saved to {retrain_out}")
            
        except Exception as e:
            st.error(f"Retrain failed: {e}")

with tabs[4]:
    st.header("🔒 Admin")
    if not st.session_state.get("admin_auth"):
        u = st.text_input("Username"); p = st.text_input("Password", type="password")
        if st.button("Login"):
            if u == "admin" and p == "admin": 
                st.session_state["admin_auth"] = True
                st.rerun()
    else:
        c1, c2 = st.columns(2)
        h = c1.text_input("MSSQL Host", ADMIN_CFG.get("MSSQL_HOST"))
        u = c1.text_input("User", ADMIN_CFG.get("MSSQL_USER"))
        p = c2.text_input("Pass", ADMIN_CFG.get("MSSQL_PASSWORD"), type="password")
        d = c2.text_input("DB", ADMIN_CFG.get("MSSQL_DATABASE"))
        t = st.text_input("Table", ADMIN_CFG.get("PROD_TABLE"))
        
        st.subheader("Column Mapping")
        with st.form("mapping"):
            cm = ADMIN_CFG.get("COLUMN_MAP", DEFAULTS["COLUMN_MAP"]).copy()
            # Filtered valid keys
            valid_keys = ["prod_track", "turnup_time", "base_gsm", "base_width", "base_wgtscaled", "base_track", "prod_gsm", "prod_wgt"]
            filtered_cm = {k: v for k, v in cm.items() if k in valid_keys}
            
            c1, c2 = st.columns(2)
            new_map = {}
            for i, (k, v) in enumerate(filtered_cm.items()):
                with (c1 if i % 2 == 0 else c2):
                    new_map[k] = st.text_input(f"{k.upper()}", v)
            
            if st.form_submit_button("Save Config"):
                upd = ADMIN_CFG.copy()
                upd.update({"MSSQL_HOST":h, "MSSQL_USER":u, "MSSQL_PASSWORD":p, "MSSQL_DATABASE":d, "PROD_TABLE":t, "COLUMN_MAP": new_map})
                save_admin_config(upd)
                st.success("Saved")