#!/usr/bin/env python3
# Compatible with Python 3.7
# Generalized paths via environment variables; optimized for production/internal server.

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")  # must be set before sklearn/joblib imports

import argparse
import json
from pathlib import Path
import joblib
from model_connection import MACHINE_MAX_WIDTH
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
from joblib import parallel_backend

# Config: can be overridden with env vars
BASE_DIR = Path(os.environ.get("COATER_BASE_DIR", Path.cwd()))
DEFAULT_DATA_PATH = Path(os.environ.get("COATER_DATA_PATH", BASE_DIR / "Input_Data.csv"))
DEFAULT_OUT_DIR = Path(os.environ.get("COATER_OUT_DIR", BASE_DIR / "artifacts"))
RANDOM_STATE = int(os.environ.get("COATER_RANDOM_STATE", 42))
CV_SPLITS = int(os.environ.get("COATER_CV_SPLITS", 5))
POLY_DEGREE = int(os.environ.get("COATER_POLY_DEGREE", 2))
ALPHAS = [float(x) for x in os.environ.get("COATER_ALPHAS", "0.01,0.1,1.0,10.0").split(",")]
SAVE_PREFIX = os.environ.get("COATER_SAVE_PREFIX", "loss_model_")
PLOT_DPI = int(os.environ.get("COATER_PLOT_DPI", 150))
DEFAULT_N_JOBS = int(os.environ.get("COATER_N_JOBS", 1))  # safe default; increase after testing

# Expected exact names in the user's uploaded CSV
EXPECTED_FEATURES = {
    "GSM": "BASE_GSM",
    "Deckle": "BASE_WIDTH"
}
# Expected target columns (exact)
EXPECTED_TARGETS = ["PACKING_WEIGHT", "TOP_LAYER_LOSS", "CORE_END_LOSS", "SPLICE_LOSS"]

def detect_features_and_targets(df):
    cols = list(df.columns)

    # Prefer exact names supplied by user
    gsm_col = EXPECTED_FEATURES["GSM"] if EXPECTED_FEATURES["GSM"] in cols else None
    deckle_col = EXPECTED_FEATURES["Deckle"] if EXPECTED_FEATURES["Deckle"] in cols else None

    # Targets: require the four exact columns; if not all present, fall back to keyword detection
    targets = [t for t in EXPECTED_TARGETS if t in cols]

    if len(targets) != len(EXPECTED_TARGETS):
        # fallback: detect by keywords (case-insensitive)
        lower_cols = {c.lower(): c for c in cols}
        detected = []
        for c in cols:
            lc = c.lower()
            if any(k in lc for k in ["packing", "top", "core", "splice", "loss"]):
                detected.append(c)
        # prefer explicit matches if still any of the expected names found
        for t in EXPECTED_TARGETS:
            if t.lower() in lower_cols and lower_cols[t.lower()] not in detected:
                detected.append(lower_cols[t.lower()])
        # remove duplicates while preserving order
        seen = set()
        targets = []
        for c in detected:
            if c not in seen:
                seen.add(c)
                targets.append(c)
        # final trim to up to 4 targets
        targets = targets[:4]

    return {
        "GSM": gsm_col,
        "Deckle": deckle_col,
        "Targets": targets
    }

def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def train_and_save(data_path, out_dir, use_loocv=False, no_plots=False, n_jobs=DEFAULT_N_JOBS):
    data_path = Path(data_path)
    out_dir = Path(out_dir)
    if not data_path.exists():
        raise FileNotFoundError("Input CSV not found: {}".format(data_path))

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(str(data_path))
    df.columns = [c.strip() for c in df.columns]

    detected = detect_features_and_targets(df)

    # Validate presence of required inputs and targets
    missing = []
    if detected["GSM"] is None:
        missing.append(EXPECTED_FEATURES["GSM"])
    if detected["Deckle"] is None:
        missing.append(EXPECTED_FEATURES["Deckle"])
    if len(detected["Targets"]) == 0:
        missing.append("Targets (PACKING_WEIGHT, TOP_LAYER_LOSS, CORE_END_LOSS, SPLICE_LOSS)")
    if missing:
        raise RuntimeError("Required columns not detected: {}".format(", ".join(missing)))

    # Standardize column names internally: BASE_GSM -> GSM, BASE_WIDTH -> Deckle_cm
    col_map = {}
    if detected["GSM"] and detected["GSM"] != "GSM":
        col_map[detected["GSM"]] = "GSM"
    if detected["Deckle"] and detected["Deckle"] != "Deckle_cm":
        col_map[detected["Deckle"]] = "Deckle_cm"
    if col_map:
        df = df.rename(columns=col_map)

    target_cols = detected["Targets"]

    # Ensure numeric for required columns
    required_feature_cols = ["GSM", "Deckle_cm"]
    required_cols = [c for c in required_feature_cols if c in df.columns] + target_cols
    df = ensure_numeric(df, required_cols)
    df = df.dropna(subset=[c for c in required_cols if c in df.columns]).reset_index(drop=True)
    if df.shape[0] == 0:
        raise RuntimeError("No rows left after cleaning. Check CSV values and column mapping.")

    # Create Deckle_Eff (normalized width) using MACHINE_MAX_WIDTH if available
    if "Deckle_cm" in df.columns:
        MACHINE_MAX_WIDTH_LOCAL = float(df["Machine_Max_Width"].iloc[0]) if "Machine_Max_Width" in df.columns else float(os.environ.get("COATER_MACHINE_MAX_WIDTH", MACHINE_MAX_WIDTH))
        df["Deckle_Eff"] = df["Deckle_cm"] / MACHINE_MAX_WIDTH_LOCAL

    # Use exactly the same modeling pipeline as before, but only use the input features BASE_GSM and BASE_WIDTH
    # Prefer using Deckle_Eff (normalized) and GSM as features (same as original behavior)
    feature_list = []
    if "GSM" in df.columns:
        feature_list.append("GSM")
    if "Deckle_Eff" in df.columns:
        feature_list.append("Deckle_Eff")
    elif "Deckle_cm" in df.columns:
        feature_list.append("Deckle_cm")

    X = df[feature_list].copy()
    y = df[target_cols].copy()

    # convert, fill and reduce precision
    X = X.apply(pd.to_numeric, errors="coerce").fillna(X.median()).astype(np.float32)
    y = y.apply(pd.to_numeric, errors="coerce")

    n_rows = len(df)
    if use_loocv:
        from sklearn.model_selection import LeaveOneOut
        cv = LeaveOneOut()
    else:
        cv = KFold(n_splits=min(CV_SPLITS, max(2, n_rows)), shuffle=True, random_state=RANDOM_STATE)

    models = {}
    cv_preds = {}
    metrics = {}

    for tcol in y.columns:
        y_vec = pd.to_numeric(y[tcol], errors="coerce").values
        valid_mask = ~np.isnan(y_vec)
        if valid_mask.sum() == 0:
            cv_preds[tcol] = np.full(len(y_vec), np.nan, dtype=float)
            metrics[tcol] = {"r2": float("nan"), "mae": float("nan")}
            continue

        X_valid = X.loc[valid_mask].values.astype(np.float32)
        y_valid = y_vec[valid_mask].astype(np.float32)

        pipe_cv = Pipeline([
            ("poly", PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)),
            ("scaler", StandardScaler()),
            ("reg", RandomForestRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE))
        ])

        with parallel_backend("threading"):
            preds_cv = cross_val_predict(pipe_cv, X_valid, y_valid, cv=cv, n_jobs=n_jobs)

        pred_full = np.full(len(y_vec), np.nan, dtype=float)
        pred_full[valid_mask] = preds_cv
        cv_preds[tcol] = pred_full

        r2 = float(r2_score(y_valid, preds_cv)) if valid_mask.sum() > 1 else float("nan")
        mae = float(mean_absolute_error(y_valid, preds_cv)) if valid_mask.sum() > 0 else float("nan")
        metrics[tcol] = {"r2": r2, "mae": mae}

        reg_final = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE)
        pipe_final = Pipeline([
            # Poly features might be redundant for RF but harmless; scaling is also not strictly needed for RF but good practice in pipeline
            ("poly", PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)),
            ("scaler", StandardScaler()),
            ("reg", reg_final)
        ])
        pipe_final.fit(X_valid, y_valid)
        models[tcol] = pipe_final

    # Compute actual and predicted totals (sum of target loss components)
    df["Total_Loss_Actual"] = y.sum(axis=1)
    total_pred = np.zeros(len(df), dtype=float)
    for tcol in y.columns:
        pred = cv_preds.get(tcol, np.full(len(df), np.nan, dtype=float))
        total_pred += np.nan_to_num(pred, nan=0.0)
    df["Total_Loss_Predicted"] = total_pred

    valid_total_mask = ~np.isnan(df["Total_Loss_Predicted"])
    if valid_total_mask.sum() == 0:
        metrics["Total_Loss"] = {"r2": float("nan"), "mae": float("nan")}
    else:
        total_r2 = float(r2_score(df.loc[valid_total_mask, "Total_Loss_Actual"], df.loc[valid_total_mask, "Total_Loss_Predicted"]))
        total_mae = float(mean_absolute_error(df.loc[valid_total_mask, "Total_Loss_Actual"], df.loc[valid_total_mask, "Total_Loss_Predicted"]))
        metrics["Total_Loss"] = {"r2": total_r2, "mae": total_mae}

    # save artifacts
    model_dir = out_dir / "models"
    plot_dir = out_dir / "plots"
    audit_path = out_dir / "cv_predictions_audit.csv"
    summary_path = out_dir / "model_summary_checklist.csv"
    metadata_path = out_dir / "metadata.json"
    model_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    saved_models = {}
    for tcol, pipe in models.items():
        fname = model_dir / "{}{}.joblib".format(SAVE_PREFIX, tcol.replace(" ", "_"))
        joblib.dump(pipe, str(fname))
        saved_models[tcol] = str(fname)

    metadata = {
        "features": feature_list,
        "targets": list(y.columns),
        "metrics_cv": metrics,
        "saved_models": saved_models,
        "poly_degree": POLY_DEGREE,
        "alphas": ALPHAS
    }
    with open(str(metadata_path), "w") as fh:
        json.dump(metadata, fh, indent=2)

    # Summary table
    rows = []
    for tcol in list(y.columns) + ["Total_Loss"]:
        if tcol == "Total_Loss":
            mean_v = float(df["Total_Loss_Actual"].mean())
            std_v = float(df["Total_Loss_Actual"].std())
            r2_v = metrics["Total_Loss"]["r2"]
            mae_v = metrics["Total_Loss"]["mae"]
        else:
            mean_v = float(y[tcol].mean())
            std_v = float(y[tcol].std())
            r2_v = metrics[tcol]["r2"]
            mae_v = metrics[tcol]["mae"]
        acc = 100.0 * (1.0 - mae_v / mean_v) if mean_v != 0 else float("nan")
        rows.append([tcol, mean_v, std_v, r2_v, mae_v, acc])

    summary_df = pd.DataFrame(rows, columns=["Loss_Component", "Mean (kg)", "Std (kg)", "R2", "MAE (kg)", "Accuracy (%)"])
    summary_df.to_csv(str(summary_path), index=False)

    # Audit CSV with input features, actuals and CV predictions
    audit = df[feature_list].copy()
    for tcol in y.columns:
        audit["{}_actual".format(tcol)] = y[tcol]
        audit["{}_pred_cv".format(tcol)] = cv_preds.get(tcol, np.full(len(df), np.nan))
    audit["Total_Loss_Actual"] = df["Total_Loss_Actual"]
    audit["Total_Loss_Predicted"] = df["Total_Loss_Predicted"]
    audit.to_csv(str(audit_path), index=False)

    # Plots
    if not no_plots:
        for tcol in list(y.columns) + ["Total_Loss"]:
            if tcol == "Total_Loss":
                y_true = df["Total_Loss_Actual"].values
                y_pred = df["Total_Loss_Predicted"].values
            else:
                y_true = y[tcol].values
                y_pred = cv_preds.get(tcol, np.full(len(df), np.nan))
            if np.all(np.isnan(y_pred)):
                continue
            plt.figure(figsize=(5,5))
            mask_valid = ~np.isnan(y_pred)
            plt.scatter(y_true[mask_valid], y_pred[mask_valid], edgecolor='k', alpha=0.7, s=12)
            mn = float(min(y_true[mask_valid].min(), y_pred[mask_valid].min()))
            mx = float(max(y_true[mask_valid].max(), y_pred[mask_valid].max()))
            plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1)
            plt.xlabel("Actual (kg)")
            plt.ylabel("Predicted (kg)")
            try:
                r2_val = r2_score(y_true[mask_valid], y_pred[mask_valid])
                mae_val = mean_absolute_error(y_true[mask_valid], y_pred[mask_valid])
            except Exception:
                r2_val = float("nan"); mae_val = float("nan")
            plt.title("{} — Actual vs Predicted\nR²={:.3f}, MAE={:.3f} kg".format(tcol, r2_val, mae_val))
            plt.tight_layout()
            plt.savefig(str(plot_dir / "{}_Actual_vs_Predicted.png".format(tcol.replace(" ", "_"))), dpi=PLOT_DPI)
            plt.close()

    # Console summary
    print("\n=== TRAINING COMPLETE ===")
    print("Rows used: {}".format(len(df)))
    print("Models saved to: {}".format(model_dir))
    print("Summary CSV: {}".format(summary_path))
    print("CV audit CSV: {}".format(audit_path))
    print("Plots: {}".format(plot_dir))
    for tcol in list(y.columns) + ["Total_Loss"]:
        r = metrics.get(tcol, {}).get("r2", float("nan"))
        m = metrics.get(tcol, {}).get("mae", float("nan"))
        print(" - {}: R2={:.3f}, MAE={:.3f}".format(tcol, (r if r is not None else float("nan")), (m if m is not None else float("nan"))))

    return metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train unavoidable loss models (py3.7 production-ready).")
    parser.add_argument("--data", "-d", type=str, default=str(DEFAULT_DATA_PATH), help="Path to input CSV file.")
    parser.add_argument("--out_dir", "-o", type=str, default=str(DEFAULT_OUT_DIR), help="Directory to save artifacts.")
    parser.add_argument("--loocv", action="store_true", help="Use LOOCV instead of KFold.")
    parser.add_argument("--no_plots", action="store_true", help="Disable plots.")
    parser.add_argument("--n_jobs", type=int, default=DEFAULT_N_JOBS, help="n_jobs for cross_val_predict (default=1).")
    args = parser.parse_args()
    train_and_save(data_path=Path(args.data), out_dir=Path(args.out_dir), use_loocv=args.loocv, no_plots=args.no_plots, n_jobs=args.n_jobs)
