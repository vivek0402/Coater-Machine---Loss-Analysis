# Coater Machine - Loss Analysis Dashboard 🏭

![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-green.svg)

An industrial analytics tool designed to optimize coating machine operations by predicting and visualizing material losses. This application combines Machine Learning (Random Forest) with an interactive Streamlit dashboard to help operators reduce avoidable waste.

## 🌟 Key Features

- **Interactive Dashboard**: Real-time analysis of production metrics using Streamlit.
- **Predictive ML**: Random Forest models to forecast:
  - `Packing Weight` (R² ≈ 0.99)
  - `Top Layer Loss` (R² ≈ 0.96)
  - `Splice Loss` (R² ≈ 0.90)
  - `Core End Loss` (R² ≈ 0.30)
- **Visualization**: Historical trends, Pareto charts for loss contributors, and distribution analysis (box plots).
- **Diagnostics**: Automated detection of outliers and process shifts (±3σ control limits).
- **Configurable**: Centralized configuration for machine parameters (e.g., Max Width).

## 📂 Project Structure

```
Coater_Dashboard/
├── dashboard.py           # Main Streamlit application
├── ml_model_code.py       # ML Training pipeline (Random Forest)
├── regenerate_bundle.py   # Tool to package models for the dashboard
├── model_connection.py    # Shared logic & schema definitions
├── Input_Data.csv         # Training dataset (Example)
├── admin_config.json      # Runtime configuration settings
└── artifacts/             # Generated models and plots
    ├── models/            # Individual trained models (.joblib)
    └── combined_loss_model.joblib # Production model bundle
```

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- Recommended: Virtual environment

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/coater-dashboard.git
   cd Coater_Dashboard
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   # OR individually:
   pip install streamlit pandas numpy scikit-learn matplotlib plotly joblib
   ```

### Running the App

1. **Launch the Dashboard**:
   ```bash
   streamlit run dashboard.py
   ```
2. Open your browser to `http://localhost:8501`.

### Retraining Models

To update the machine learning models with new data:

1. Place your new data in `Input_Data.csv`.
2. Run the training script:
   ```bash
   python ml_model_code.py
   ```
   *This trains the models and saves them to `artifacts/models/`.*
3. Generate the production bundle:
   ```bash
   python regenerate_bundle.py
   ```
   *This packages the models into `combined_loss_model.joblib` for the dashboard.*

## ⚙️ Configuration

- **`admin_config.json`**: Modify this file to change default settings, such as `DEFAULT_BUNDLE` path or database connections.
- **Environment Variables**:
  - `COATER_Machine_MAX_WIDTH`: Override the machine width (Default: 172.0).

## 🤝 Contribution

1. Fork the Project
2. Create your Feature Branch
3. Commit your Changes
4. Push to the Branch
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
