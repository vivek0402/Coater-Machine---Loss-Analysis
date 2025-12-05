#!/usr/bin/env python3
"""
Build a CombinedModel joblib that wraps per-target pipelines into one object.
Canonical inputs now use:
 - BASE_GSM (preferred; will accept GSM as alternative)
 - BASE_WIDTH (direct column) and computed BASE_WIDTH_EFF (normalized)
This script maps legacy names if present and computes BASE_WIDTH_EFF from BASE_WIDTH.
"""

import os
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

BASE_DIR = Path(os.environ.get("COATER_BASE_DIR", Path.cwd()))
MODEL_DIR = Path(os.environ.get("COATER_MODEL_DIR", BASE_DIR / "artifacts" / "models"))
METADATA_PATH = Path(os.environ.get("COATER_METADATA_PATH", BASE_DIR / "artifacts" / "metadata.json"))
AUDIT_CSV = Path(os.environ.get("COATER_AUDIT_CSV", BASE_DIR / "artifacts" / "cv_predictions_audit.csv"))
COMBINED_OUT_PATH = Path(os.environ.get("COATER_COMBINED_OUT", BASE_DIR / "combined_loss_model.joblib"))

# Canonical features the combined model will expect
FALLBACK_FEATURE_NAMES = ["BASE_GSM", "BASE_WIDTH_EFF"]

MACHINE_MAX_WIDTH = 172.0

# Alternative mapping to help users who provide GSM / BASE_GSM / BASE_WIDTH (legacy)
ALTERNATIVE_FEATURES = {
    "BASE_GSM": ["BASE_GSM", "base_gsm", "GSM", "gsm"],
    # The dataset no longer uses 'deckle'; use BASE_WIDTH as canonical source for width
    "BASE_WIDTH": ["BASE_WIDTH", "base_width", "width", "WIDTH", "baseWidth"],
}

@dataclass
class CombinedModel:
    feature_names: List[str]
    pipelines: Dict[str, object]
    residual_std: Dict[str, float] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    def _get_machine_max_width(self) -> float:
        # Try metadata, then environment, then default
        mw = None
        try:
            mw = float(self.metadata.get("machine_max_width")) if self.metadata.get("machine_max_width") is not None else None
        except Exception:
            mw = None
        if mw is None:
            try:
                mw = float(os.environ.get("COATER_MACHINE_MAX_WIDTH", 172.0))
            except Exception:
                mw = float(MACHINE_MAX_WIDTH)
        return mw

    def _compute_base_width_eff(self, df: pd.DataFrame) -> pd.Series:
        mw = self._get_machine_max_width()
        # Prefer canonical BASE_WIDTH column if present
        if "BASE_WIDTH" in df.columns:
            bw = pd.to_numeric(df["BASE_WIDTH"], errors="coerce")
            return bw / float(mw)
        # Fallback: try alternative names
        for alt in ALTERNATIVE_FEATURES["BASE_WIDTH"]:
            if alt in df.columns:
                bw = pd.to_numeric(df[alt], errors="coerce")
                return bw / float(mw)
        raise ValueError("Cannot compute BASE_WIDTH_EFF: no BASE_WIDTH-like column present in input dataframe.")

    def validate_input_df(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure input dataframe contains canonical feature_names.
        If alternatives are present (e.g., GSM, BASE_GSM, BASE_WIDTH), create/copy/compute canonical columns.
        """
        df = X_df.copy()

        # Create BASE_GSM if missing and alternatives present (GSM etc.)
        if "BASE_GSM" not in df.columns:
            for alt in ALTERNATIVE_FEATURES["BASE_GSM"]:
                if alt in df.columns:
                    df["BASE_GSM"] = pd.to_numeric(df[alt], errors="coerce")
                    break

        # Ensure BASE_WIDTH exists if provided with alternative name variants
        if "BASE_WIDTH" not in df.columns:
            for alt in ALTERNATIVE_FEATURES["BASE_WIDTH"]:
                if alt in df.columns:
                    df["BASE_WIDTH"] = pd.to_numeric(df[alt], errors="coerce")
                    break

        # Compute BASE_WIDTH_EFF if required by feature_names and missing
        if "BASE_WIDTH_EFF" in self.feature_names and "BASE_WIDTH_EFF" not in df.columns:
            df["BASE_WIDTH_EFF"] = self._compute_base_width_eff(df)

        # After attempts, verify all required features are present
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            raise ValueError("Missing required feature columns after mapping: {}. Provided columns: {}".format(missing, list(X_df.columns)))

        # Enforce numeric dtype for required columns
        for c in self.feature_names:
            df[c] = pd.to_numeric(df[c], errors="raise")

        return df

    def _prepare_X_for_pipeline(self, X_df: pd.DataFrame) -> pd.DataFrame:
        return X_df[self.feature_names].copy()

    def predict_df(self, X_df: pd.DataFrame) -> pd.DataFrame:
        X_sub = self.validate_input_df(X_df)
        X_prep = self._prepare_X_for_pipeline(X_sub)
        out = {}
        for tname, pipe in self.pipelines.items():
            try:
                preds = pipe.predict(X_prep.values)
            except Exception:
                preds = pipe.predict(X_prep)
            out[tname] = np.asarray(preds).reshape(-1)
        return pd.DataFrame(out, index=X_sub.index)

    def predict_single(self, **kwargs):
        # Accept canonical names or alternatives (BASE_GSM/BASE_WIDTH/GSM); build row and predict
        row = dict(kwargs)
        df_row = pd.DataFrame([row])
        preds = self.predict_df(df_row)
        return preds.iloc[0].to_dict()

    def predict_with_intervals_df(self, X_df: pd.DataFrame, alpha: float = 0.05):
        preds = self.predict_df(X_df)
        try:
            from scipy.stats import norm
            z = norm.ppf(1 - alpha/2.0)
        except Exception:
            # sensible defaults
            z = 1.96 if abs(alpha - 0.05) < 1e-6 else 1.645 if abs(alpha - 0.1) < 1e-6 else 2.576 if abs(alpha - 0.01) < 1e-6 else 1.96
        out = preds.copy()
        for t in preds.columns:
            std = float(self.residual_std.get(t, np.nan))
            out[f"{t}_lower"] = preds[t] - z * std
            out[f"{t}_upper"] = preds[t] + z * std
        return out

def build_combined_model(model_dir: Path = MODEL_DIR,
                         metadata_path: Optional[Path] = METADATA_PATH,
                         audit_csv: Optional[Path] = AUDIT_CSV,
                         fallback_features: List[str] = FALLBACK_FEATURE_NAMES) -> CombinedModel:

    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    # load model files with expected prefix
    files = sorted(model_dir.glob("loss_model_*.joblib"))
    if not files:
        files = sorted(model_dir.glob("*.joblib"))
        if not files:
            raise FileNotFoundError(f"No model files found in {model_dir}")

    pipelines = {}
    for f in files:
        name = f.stem
        tname = name.replace("loss_model_", "", 1) if name.startswith("loss_model_") else name
        pipelines[tname] = joblib.load(str(f))

    metadata = {}
    feature_names = None
    if metadata_path and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            feature_names = metadata.get("features", None)
            if feature_names and isinstance(feature_names, list):
                # Normalize legacy names to new canonical names
                normalized = []
                for fn in feature_names:
                    if fn in ("GSM", "gsm", "BASE_GSM"):
                        normalized.append("BASE_GSM")
                    elif fn in ("Deckle_Eff", "Deckle_eff", "deckle_eff", "Deckle_EFF"):
                        normalized.append("BASE_WIDTH_EFF")
                    elif fn in ("Deckle_cm", "deckle_cm", "Deckle_cm"):
                        normalized.append("BASE_WIDTH")
                    else:
                        normalized.append(fn)
                feature_names = normalized
            else:
                feature_names = None
        except Exception:
            feature_names = None

    if feature_names is None:
        feature_names = list(fallback_features)

    # Ensure feature_names are canonical: replace any legacy tokens
    feature_names = [
        "BASE_GSM" if f in ("GSM", "gsm") else ("BASE_WIDTH_EFF" if f in ("Deckle_Eff", "Deckle_eff", "deckle_eff", "Deckle_EFF") else f)
        for f in feature_names
    ]

    # Validate final feature_names
    if not isinstance(feature_names, list) or not all(isinstance(x, str) for x in feature_names):
        raise RuntimeError(f"Resolved invalid feature_names: {feature_names}")

    # compute residual stds from audit CSV if available
    residual_std = {}
    if audit_csv and audit_csv.exists():
        try:
            df_audit = pd.read_csv(audit_csv)
            for tname in pipelines.keys():
                actual_col = f"{tname}_actual"
                pred_col = f"{tname}_pred_cv"
                if actual_col in df_audit.columns and pred_col in df_audit.columns:
                    s = pd.to_numeric(df_audit[actual_col], errors="coerce") - pd.to_numeric(df_audit[pred_col], errors="coerce")
                    s = s.dropna()
                    residual_std[tname] = float(s.std(ddof=1)) if len(s) >= 2 else float(np.nan)
                else:
                    residual_std[tname] = float(np.nan)
        except Exception:
            residual_std = {t: float(np.nan) for t in pipelines.keys()}
    else:
        residual_std = {t: float(np.nan) for t in pipelines.keys()}

    combined = CombinedModel(feature_names=feature_names, pipelines=pipelines, residual_std=residual_std, metadata=metadata)
    return combined

def main():
    combined = build_combined_model()
    COMBINED_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(combined, str(COMBINED_OUT_PATH))
    print("Saved combined model to:", str(COMBINED_OUT_PATH))
    print("Targets:", list(combined.pipelines.keys()))
    print("Features expected:", combined.feature_names)

if __name__ == "__main__":
    main()
