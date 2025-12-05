# Technical Documentation

**Project**: Coater Machine Loss Analysis  
**Version**: 2.0  
**Stack**: Python, Streamlit, Scikit-Learn

---

## 1. Architecture Overview

The system is composed of three decoupled layers:

1.  **Training Layer (`ml_model_code.py`)**:
    *   Ingests raw production data (`Input_Data.csv`).
    *   Performs feature engineering (Standard Scaling, Polynomial Features).
    *   Trains independent **Random Forest Regressors** for each target variable.
    *   Validates performance using Cross-Validation (K-Fold).

2.  **Logic Layer (`model_connection.py`)**:
    *   Defines the `CombinedModel` class, which acts as the standardized interface (API) between the raw models and the UI.
    *   Handles feature normalization (e.g., `BASE_WIDTH_EFF` calculation).
    *   Manages single sources of truth for constants (e.g., `MACHINE_MAX_WIDTH`).

3.  **Presentation Layer (`dashboard.py`)**:
    *   A Streamlit-based web application.
    *   Loads the serialized `CombinedModel`.
    *   Provides interactive tools for operators to simulate scenarios ("What-If" analysis) and view historical trends.

---

## 2. Machine Learning Approach

### Data Pipeline
*   **Inputs**: `BASE_GSM` (Grammage), `BASE_WIDTH` (Deckle).
*   **Targets**:
    1.  `PACKING_WEIGHT`
    2.  `TOP_LAYER_LOSS`
    3.  `CORE_END_LOSS`
    4.  `SPLICE_LOSS`

### Model Selection
We utilize **Random Forest Regression** (Ensemble Learning).
*   *Why?* The relationship between machine width and certain loss types (like Core End Loss) is non-linear and noisy. Random Forest handles these step-functions and non-linearities better than linear Ridge Regression, yielding a **3x improvement** in accuracy for difficult targets.

### Performance Metrics (Current)
| Target | R² Score | MAE (kg) | Status |
| :--- | :--- | :--- | :--- |
| Packing Weight | 0.99 | 0.008 | ✅ Production Ready |
| Top Layer Loss | 0.96 | 0.046 | ✅ Production Ready |
| Core End Loss | 0.30 | 0.557 | ⚠️ Data Limited |
| Splice Loss | 0.90 | 0.205 | ✅ Production Ready |

*Note: Core End Loss accuracy is currently limited by data consistency (identical inputs yielding inconsistent outputs in the training set).*

---

## 3. Codebase Guide

### `model_connection.py`
This is the **core dependency**. It defines:
*   `CombinedModel`: The wrapper class used to bundle multiple pipeline objects.
*   `MACHINE_MAX_WIDTH`: The global constant for machine specifications (default: `172.0`).

### `regenerate_bundle.py`
A utility script that avoids "pickle hell". It cleanly imports `model_connection` and packages the individual `.joblib` models from the training step into a single artifact (`combined_loss_model.joblib`) that the dashboard can load safely.

---

## 4. Deployment Details

*   **Runtime**: Python 3.7+
*   **Dependencies**: See `requirements.txt` (implied).
*   **Environment Variables**:
    *   `COATER_ADMIN_CONFIG`: Path to custom JSON config.
    *   `COATER_DB_PATH`: Path to SQLite database for logs.

---

## 5. Future Improvements

1.  **Feature Expansion**: Collect *Operator Name*, *Shift ID*, and *Line Speed* to improve `Core End Loss` prediction.
2.  **Database Integration**: Connect directly to MSSQL/PostgreSQL for live production data instead of CSV uploads.
