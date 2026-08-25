"""
Reusable UI Components: Formatted Tables and Metrics.
"""
from typing import Dict, List
import pandas as pd
import streamlit as st


def render_model_comparison_table(metrics_df: pd.DataFrame):
    """Render a clean model comparison table with highlights."""
    styled_df = metrics_df.copy()
    for col in ["roc_auc", "pr_auc", "recall", "precision", "f1", "specificity"]:
        if col in styled_df.columns:
            styled_df[col] = styled_df[col].apply(lambda x: f"{x:.4f}")

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
    )


def render_financial_group_table(features_dict: Dict[str, float], benchmark_dict: Dict[str, float]):
    """Render financial ratio comparison with healthy benchmarks."""
    rows = []
    for feat, val in features_dict.items():
        bench = benchmark_dict.get(feat, val)
        diff = val - bench
        rows.append({
            "Financial Ratio": feat,
            "Company Value": f"{val:.4f}",
            "Cohort Median": f"{bench:.4f}",
            "Deviation": f"{diff:+.4f}",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
