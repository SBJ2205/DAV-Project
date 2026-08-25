"""
Main Streamlit Application Entry Point.
Small Business Insolvency Early-Warning Visual Analytics System.
"""
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.pages.explainability import render_explainability_page
from dashboard.pages.overview import render_overview_page
from dashboard.pages.risk_assessment import render_risk_assessment_page
from dashboard.pages.what_if import render_what_if_page
from src.explainability.shap_explainer import compute_shap_values

# 1. Page Configuration
st.set_page_config(
    page_title="Insolvency Early-Warning Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Custom CSS Theme
st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"] {
            display: none !important;
        }
        .main-header {
            font-size: 1.8rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0px;
        }
        .sub-header {
            font-size: 0.95rem;
            color: #64748b;
            margin-bottom: 20px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.6rem;
            font-weight: 700;
            color: #1e293b;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
            background-color: #f8fafc;
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_datasets():
    """Load preprocessed training and test datasets."""
    train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


@st.cache_resource
def load_models_and_shap():
    """Load model, preprocessors, and compute cached SHAP matrix."""
    models_dir = PROJECT_ROOT / "models"
    rf_model = joblib.load(models_dir / "random_forest_model.pkl")
    tree_prep = joblib.load(models_dir / "tree_preprocessor.pkl")
    logistic_model = joblib.load(models_dir / "logistic_model.pkl")
    logistic_prep = joblib.load(models_dir / "logistic_preprocessor.pkl")

    # Load test data to precompute SHAP matrix
    _, test_df = load_datasets()
    X_test = test_df.drop(columns=["Bankrupt?"])
    X_test_mat = tree_prep.transform(X_test)

    # Use first 50 training instances as background
    train_df, _ = load_datasets()
    X_train = train_df.drop(columns=["Bankrupt?"])
    X_train_mat = tree_prep.transform(X_train)
    X_bg = X_train_mat[:50]

    explainer = shap.TreeExplainer(rf_model, data=X_bg, model_output="probability")
    shap_vals, base_val = compute_shap_values(explainer, X_test_mat)

    return {
        "model": rf_model,
        "tree_prep": tree_prep,
        "logistic_model": logistic_model,
        "logistic_prep": logistic_prep,
        "explainer": explainer,
        "shap_vals": shap_vals,
        "base_val": base_val,
    }


def main():
    # Load cached artifacts
    try:
        train_df, test_df = load_datasets()
        artifacts = load_models_and_shap()
    except Exception as e:
        st.error(f"Error loading system assets: {e}. Please ensure data preprocessing and model training have run.")
        st.stop()

    rf_model = artifacts["model"]
    tree_prep = artifacts["tree_prep"]
    shap_vals = artifacts["shap_vals"]

    X_test = test_df.drop(columns=["Bankrupt?"])
    X_test_mat = tree_prep.transform(X_test)
    test_probs = rf_model.predict_proba(X_test_mat)[:, 1]
    feature_names = X_test.columns.tolist()

    # Calculate cohort medians for benchmarking
    cohort_medians = X_test.median().to_dict()

    # Bundle for pages
    data_bundle = {
        "train_df": train_df,
        "test_df": test_df,
        "feature_names": feature_names,
        "test_probs": test_probs,
        "model": rf_model,
        "tree_prep": tree_prep,
        "test_shap_vals": shap_vals,
        "cohort_medians": cohort_medians,
    }

    # Sidebar Navigation
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
        st.title("🛡️ Insolvency Early-Warning")
        st.caption("Visual Analytics & Explainable AI")

        page_choice = st.radio(
            "System Navigation:",
            [
                "📊 1. Overview & Benchmark",
                "🔍 2. Risk Assessment",
                "🧠 3. Explainability (SHAP)",
                "🔮 4. What-If Trajectory",
            ],
            index=0,
        )

        st.markdown("---")
        st.markdown("### ⚙️ Active System State")
        st.markdown("- **Model**: `Random Forest (Balanced)`")
        st.markdown("- **Cohort Size**: `1,364 records`")
        st.markdown("- **PR-AUC**: `0.4999`")
        st.markdown("- **ROC-AUC**: `0.9486`")
        st.markdown("---")
        st.caption("© DAV Research Project | Visual Analytics Lab")

    # Route to Selected Page
    if "1. Overview" in page_choice:
        render_overview_page(data_bundle)
    elif "2. Risk Assessment" in page_choice:
        render_risk_assessment_page(data_bundle)
    elif "3. Explainability" in page_choice:
        render_explainability_page(data_bundle)
    elif "4. What-If" in page_choice:
        render_what_if_page(data_bundle)


if __name__ == "__main__":
    main()
