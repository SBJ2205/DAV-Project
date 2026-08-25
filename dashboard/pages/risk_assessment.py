"""
Dashboard Page: Company Insolvency Risk Assessment.
"""
from typing import Dict
import pandas as pd
import streamlit as st

from dashboard.components.risk_card import render_risk_card
from dashboard.components.tables import render_financial_group_table


def render_risk_assessment_page(data_bundle: dict):
    st.markdown("## 🔍 Company Insolvency Risk Assessment")
    st.markdown("Inspect individual company records, predicted bankruptcy risk scores, and granular financial profiles.")

    test_df = data_bundle["test_df"]
    test_probs = data_bundle["test_probs"]
    feature_names = data_bundle["feature_names"]
    medians = data_bundle["cohort_medians"]

    # Record Selection Controls
    col_filter, col_select = st.columns([1, 2])
    with col_filter:
        risk_filter = st.selectbox(
            "Filter Test Cohort by Risk Tier:",
            ["All Records", "High Risk (>60%)", "Moderate Risk (30-60%)", "Low Risk (<30%)"],
        )

    # Filter indices based on selection
    if risk_filter == "High Risk (>60%)":
        valid_indices = [i for i, p in enumerate(test_probs) if p >= 0.60]
    elif risk_filter == "Moderate Risk (30-60%)":
        valid_indices = [i for i, p in enumerate(test_probs) if 0.30 <= p < 0.60]
    elif risk_filter == "Low Risk (<30%)":
        valid_indices = [i for i, p in enumerate(test_probs) if p < 0.30]
    else:
        valid_indices = list(range(len(test_probs)))

    with col_select:
        record_idx = st.selectbox(
            "Select Company Record:",
            options=valid_indices,
            format_func=lambda idx: f"Dataset Record #{idx} — Predicted Risk: {test_probs[idx]*100:.1f}% (Actual: {'Bankrupt' if test_df.iloc[idx]['Bankrupt?'] == 1 else 'Healthy'})",
            index=0 if len(valid_indices) > 0 else 0,
        )

    st.markdown("---")

    # Render Risk Badge Card
    selected_prob = float(test_probs[record_idx])
    render_risk_card(selected_prob, record_idx)

    # Actual outcome verification note
    actual_label = int(test_df.iloc[record_idx]["Bankrupt?"])
    st.caption(f"📌 **Ground Truth Label in Dataset**: `{'Bankrupt (1)' if actual_label == 1 else 'Healthy / Non-Bankrupt (0)'}`")

    st.markdown("---")

    # Categorized Financial Profile
    st.subheader("📑 Categorized Financial Profile & Benchmark Comparison")
    st.markdown("Compare the selected company's financial indicators against the test cohort median.")

    rec_features = test_df.iloc[record_idx].drop("Bankrupt?").to_dict()

    tab1, tab2, tab3, tab4 = st.tabs([
        "💧 Liquidity & Working Capital",
        "⚖️ Leverage & Solvency",
        "📈 Profitability & Growth",
        "⚙️ Operational & Cash Flow",
    ])

    with tab1:
        liq_cols = [c for c in feature_names if any(w in c.lower() for w in ["quick", "current", "cash", "working capital", "liquidity"])][:8]
        sub_dict = {c: rec_features[c] for c in liq_cols}
        render_financial_group_table(sub_dict, medians)

    with tab2:
        lev_cols = [c for c in feature_names if any(w in c.lower() for w in ["debt", "liability", "borrowing", "equity to", "leverage"])][:8]
        sub_dict = {c: rec_features[c] for c in lev_cols}
        render_financial_group_table(sub_dict, medians)

    with tab3:
        prof_cols = [c for c in feature_names if any(w in c.lower() for w in ["roa", "roe", "profit", "income", "eps", "margin", "growth"])][:8]
        sub_dict = {c: rec_features[c] for c in prof_cols}
        render_financial_group_table(sub_dict, medians)

    with tab4:
        ops_cols = [c for c in feature_names if any(w in c.lower() for w in ["turnover", "inventory", "asset", "revenue", "expense", "operating"])][:8]
        sub_dict = {c: rec_features[c] for c in ops_cols}
        render_financial_group_table(sub_dict, medians)
