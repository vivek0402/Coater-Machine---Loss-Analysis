# User Walkthrough Guide 🚶‍♂️

Welcome to the Coater Machine Loss Analysis Dashboard! This guide will take you step-by-step from zero to running your first analysis.

---

## 🏗️ Part 1: First-Time Setup

**Goal**: Get the application running on your computer.

1.  **Install Python**:
    *   Ensure you have Python 3.7 or newer installed.
    *   Type `python --version` in your terminal to check.

2.  **Get the Code**:
    *   Download or unzip the `Coater_Dashboard` folder to your computer.
    *   Open a terminal/command prompt and navigate to the folder:
        ```bash
        cd path/to/Coater_Dashboard
        ```

3.  **Install Libraries**:
    *   We need a few tools to run the math and charts. Run this command:
        ```bash
        pip install -r requirements.txt
        ```
    *   *Note*: If you don't have `requirements.txt` yet, simply run:
        ```bash
        pip install streamlit pandas numpy scikit-learn matplotlib plotly joblib
        ```

---

## 🚀 Part 2: Launching the Dashboard

**Goal**: Open the visual interface.

1.  **Run the App**:
    *   In your terminal inside the `Coater_Dashboard` folder, type:
        ```bash
        streamlit run dashboard.py
        ```

2.  **View in Browser**:
    *   A new tab should automatically open in your web browser (Chrome, Edge, etc.).
    *   If not, look at the terminal output and click the link that looks like `http://localhost:8501`.

---

## 🖥️ Part 3: Using the Dashboard

**Goal**: Understand what you are seeing.

### 1. The Sidebar (Left Panel)
*   **Navigation**: Switch between "Dashboard", "Diagnostics", and "Model Info".
*   **Filters**: If you upload new data, filter by Date Range or Product Type here.

### 2. Main Dashboard (Center)
*   **KPI Cards**: At the top, you see the predicted losses (in kg) for the current machine settings.
    *   *Packing Weight*
    *   *Top Layer Loss*
    *   *Splice Loss*
    *   *Core End Loss*
*   **Total Loss**: The big number showing the sum of all losses.
*   **Charts**:
    *   *Loss Composition*: A pie chart showing which type of loss is the biggest problem predicted to be.
    *   *Trend Prediction*: A line graph showing loss over time (if time-series data is present).

---

## 🧪 Part 4: "What-If" Analysis

**Goal**: Use the ML model to optimize settings.

*   *Scenario*: You want to know, "If we increase the Machine Width, what happens to the loss?"
*   *Action*:
    1.  The dashboard (depending on version) may have sliders or inputs for `GSM` and `Width`.
    2.  Change the **Machine Width** value.
    3.  Watch the **Predicted Loss** numbers update instantly.
    *   *Insight*: Use this to find the "sweet spot" where loss is minimized.

---

## 🔄 Part 5: Retraining the Model (Advanced)

**Goal**: Teach the AI with new data.

1.  **Prepare Data**:
    *   Save your new production data as `Input_Data.csv` in the folder.
    *   Ensure columns `BASE_GSM` and `BASE_WIDTH` exist.

2.  **Run Training**:
    *   In the terminal, stop the dashboard (Ctrl+C) and run:
        ```bash
        python ml_model_code.py
        ```
    *   You will see output like `Top_Layer_Loss: R2=0.96`.

3.  **Bundle for Dashboard**:
    *   Run this command to package the new brains for the app:
        ```bash
        python regenerate_bundle.py
        ```

4.  **Restart Dashboard**:
    *   `streamlit run dashboard.py`
    *   Your dashboard is now using the smarter, updated models!

---

## ❓ Troubleshooting

*   **"Command not found"**: Make sure you installed Python and added it to your PATH.
*   **"ModuleNotFoundError"**: You missed a library. Run the `pip install` step again.
*   **"NameError: MACHINE_MAX_WIDTH"**: This was a bug in older versions. If you see this, ensure you are using the latest `dashboard.py` code.
