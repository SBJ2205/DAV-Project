# 🛡️ Small Business Insolvency Early-Warning Visual Analytics System

> **Data Analysis & Visualization (DAV) Minor Project**  
> An explainable machine-learning system that predicts bankruptcy risk for small and medium enterprises (SMEs) and visualizes the exact financial drivers behind each prediction using **Explainable AI (SHAP)** and **What-If Sensitivity Trajectories**.

---

## 🌟 Key Highlights

- **Predictive Modeling**: Evaluates **Logistic Regression (Baseline)**, **Random Forest (Balanced)**, and **XGBoost (Cost-Sensitive)** on severely imbalanced financial data.
- **Scientific Imbalanced Evaluation**: Employs **PR-AUC (Precision-Recall AUC)** and Recall as primary metrics rather than misleading raw accuracy.
- **Explainable AI (XAI)**: Utilizes game-theoretic **SHAP (SHapley Additive exPlanations)** values for both global model importance and record-level positive/negative factor attribution.
- **What-If Risk Trajectory Engine**: Simulates real-time managerial interventions (e.g. reducing borrowing dependency or improving liquidity by $\pm 30\%$) and recalculates model-predicted risk curves.
- **Interactive Multi-Page Streamlit Dashboard**: Intuitive visual analytics for financial analysts, loan officers, and students.
- **Zero Data Leakage**: Preprocessing transformations fitted strictly on training partitions; 100% test coverage with 17 automated unit tests.

---

## 📊 Dataset Profile

- **Dataset**: Taiwan Economic Journal (TEJ) Company Bankruptcy Prediction Dataset
- **Total Records**: 6,819 companies
- **Features**: 95 raw financial indicators $\rightarrow$ 94 retained predictors (1 constant column `Net Income Flag` dynamically removed).
- **Target Variable**: `Bankrupt?` (Binary: 0 = Healthy, 1 = Bankrupt)
- **Class Distribution**: **220 Bankrupt (3.23%)** vs **6,599 Healthy (96.77%)** — *Severe Imbalance*
- **Data Nature**: Cross-sectional financial observations.

---

## 🏆 Model Performance Benchmark

Models were evaluated on the held-out test cohort ($N = 1,364$, $20\%$):

| Model | ROC-AUC | PR-AUC (Key) | Recall | Precision | F1-Score | Specificity |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.9171 | 0.3190 | **0.8182** | 0.1856 | 0.3025 | 0.8803 |
| **Random Forest (Selected)** | **0.9486** | **0.4999** | 0.6591 | 0.3718 | **0.4754** | **0.9629** |
| **XGBoost (Weighted)** | **0.9531** | 0.4779 | 0.6136 | **0.3913** | **0.4779** | **0.9682** |

> **Selected Model**: **Random Forest** achieved the highest PR-AUC (**0.4999**) and highest F1-score (**0.4754**) while maintaining an exceptional specificity of **96.29%** (low false alarms).

---

## 🧠 Top 5 Financial Distress Predictors (Global SHAP)

1. **Continuous interest rate (after tax)** — High interest burdens relative to operating profits.
2. **Borrowing dependency** — Heavy reliance on external debt financing.
3. **Persistent EPS in the Last Four Seasons** — Multi-quarter earnings erosion.
4. **Total debt / Total net worth** — Excessive financial leverage.
5. **Equity to Liability** — Inadequate equity buffers.

---

## 📂 Project Architecture

```
DAV-Project/
├── data/
│   ├── raw/data.csv                          # Untouched raw dataset
│   └── processed/
│       ├── train.csv                         # 80% Stratified training set (5,455 records)
│       └── test.csv                          # 20% Stratified test set (1,364 records)
├── dashboard/
│   ├── app.py                                # Main Streamlit application entry point
│   ├── pages/
│   │   ├── overview.py                       # Overview, risk histogram & benchmark table
│   │   ├── risk_assessment.py                # Company selector, risk badge & financial tabs
│   │   ├── explainability.py                 # SHAP waterfall & global importance
│   │   └── what_if.py                        # What-if sensitivity simulation & Plotly curve
│   └── components/
│       ├── risk_card.py                      # Reusable risk tier progress card
│       ├── charts.py                         # Reusable Plotly visualizations
│       └── tables.py                         # Benchmark and financial comparison tables
├── models/
│   ├── logistic_model.pkl                    # Serialized Logistic Regression model
│   ├── logistic_preprocessor.pkl             # Fitted Scaler + Imputer
│   ├── random_forest_model.pkl               # Serialized Random Forest model
│   ├── xgboost_model.pkl                     # Serialized XGBoost model
│   └── tree_preprocessor.pkl                 # Fitted Imputer
├── src/
│   ├── data/                                 # Data ingestion and preprocessing
│   ├── evaluation/                           # Evaluation metrics and curve plotting
│   ├── explainability/                       # SHAP wrapper and factor decomposition
│   ├── trajectory/                           # What-if sensitivity engine
│   └── utils/                                # Logging utilities
├── outputs/
│   ├── data_quality/                         # Data quality reports & metadata
│   ├── figures/                              # Generated figures (EDA, ROC, PR, SHAP)
│   ├── metrics/                              # Comparison tables & CV results
│   ├── validation/validation_report.md       # Quality assurance audit
│   └── research/research_summary.md          # Research findings summary
├── scripts/                                  # Pipeline execution scripts
├── tests/                                    # 17 Automated pytest unit & integration tests
├── .gitignore
├── pytest.ini
├── requirements.txt
└── DAV_Project_Explanation_Guide.docx        # Presentation guide for viva
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/SBJ2205/DAV-Project.git
cd DAV-Project
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python -m venv .venv
source .venv/Scripts/activate       # On Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch the Interactive Dashboard
```bash
streamlit run dashboard/app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

### 4. Run Automated Tests
```bash
pytest -v
```

---

## 👥 Contributors & Acknowledgements
- Developed for **Data Analysis and Visualization (DAV)** Lab Project.
