"""
Dashboard Page: System Overview & Model Intelligence.
"""
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.charts import render_risk_distribution_chart
from dashboard.components.tables import render_model_comparison_table

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIG_COMP_DIR = PROJECT_ROOT / "outputs" / "figures" / "model_comparison"


def render_overview_page(data_bundle: dict):
    st.markdown("## 📊 Small Business Insolvency Early-Warning Visual Analytics System")
    st.markdown(
        """
        An explainable machine-learning visual analytics framework that estimates corporate insolvency risk 
        and explains the underlying financial factors driving individual predictions.
        """
    )

    st.markdown("---")

    # 1. Headline KPI Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric(label="Total Observations", value="6,819")
    with kpi2:
        st.metric(label="Bankrupt Companies", value="220 (3.23%)")
    with kpi3:
        st.metric(label="Healthy Companies", value="6,599 (96.77%)")
    with kpi4:
        st.metric(label="Selected Model", value="Random Forest")
    with kpi5:
        st.metric(label="PR-AUC (Primary Metric)", value="0.4999", delta="+0.467 vs Random")

    st.markdown("---")

    # 2. Risk Distribution across Cohort
    test_probs = data_bundle["test_probs"]
    st.subheader("🎯 Insolvency Risk Distribution Across Test Set")
    st.markdown("Breakdown of model-predicted bankruptcy probabilities for the held-out test cohort ($N=1,364$).")
    
    col_chart, col_stats = st.columns([2, 1])
    with col_chart:
        fig_dist = render_risk_distribution_chart(test_probs)
        st.plotly_chart(fig_dist, use_container_width=True)
    with col_stats:
        high_cnt = (test_probs >= 0.60).sum()
        med_cnt = ((test_probs >= 0.30) & (test_probs < 0.60)).sum()
        low_cnt = (test_probs < 0.30).sum()

        st.markdown("### Risk Tier Summary")
        st.markdown(f"🔴 **High Risk (>60%)**: `{high_cnt}` companies ({high_cnt/len(test_probs)*100:.1f}%)")
        st.markdown(f"🟡 **Moderate Risk (30–60%)**: `{med_cnt}` companies ({med_cnt/len(test_probs)*100:.1f}%)")
        st.markdown(f"🟢 **Low Risk (<30%)**: `{low_cnt}` companies ({low_cnt/len(test_probs)*100:.1f}%)")
        st.info("💡 **Configurable Tiers**: High-risk alerts allow credit and loan officers to prioritize early intervention.")

    st.markdown("---")

    # 3. Model Benchmark Comparison
    st.subheader("🏆 Model Benchmark & Performance Comparison")
    st.markdown(
        r"""
        Models were trained strictly on training data using Stratified 5-Fold Cross-Validation 
        and evaluated on the unseen held-out test set ($20\%$). Because the dataset is severely imbalanced 
        ($3.23\%$ bankrupt), **PR-AUC and Recall** are the primary evaluation criteria.
        """
    )

    comp_path = METRICS_DIR / "model_comparison.csv"
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
        render_model_comparison_table(comp_df)

    # 4. Methodological Transparency & Limitations
    with st.expander("ℹ️ Research Methodology & Dataset Transparency", expanded=False):
        st.markdown(
            """
            - **Cross-Sectional Dataset**: The underlying dataset represents single-period financial snapshots for companies. It does not contain longitudinal multi-year timelines for the same firm.
            - **Non-Causal Interpretability**: Model importances and SHAP values indicate associative contributions to model predictions, not proven causal financial mechanisms.
            - **Zero Data Leakage**: Imputation, scaling, and hyperparameter decisions were strictly fitted on training folds.
            - **What-If Sensitivity**: Trajectory charts represent controlled single-variable sensitivity simulations rather than guaranteed future financial trajectories.
            """
        )
