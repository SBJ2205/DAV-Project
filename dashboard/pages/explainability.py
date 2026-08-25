"""
Dashboard Page: Explainable AI & SHAP Insights.
"""
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.charts import render_shap_waterfall_chart

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIG_SHAP_DIR = PROJECT_ROOT / "outputs" / "figures" / "shap"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"


def render_explainability_page(data_bundle: dict):
    st.markdown("## 🧠 Explainable AI & SHAP Feature Insights")
    st.markdown(
        """
        Explainable AI helps risk analysts understand **why** a company received a specific bankruptcy risk prediction. 
        SHAP (SHapley Additive exPlanations) values quantify each financial feature's positive or negative contribution 
        towards the model's output.
        """
    )

    test_df = data_bundle["test_df"]
    test_probs = data_bundle["test_probs"]
    feature_names = data_bundle["feature_names"]
    shap_vals = data_bundle["test_shap_vals"]  # Computed SHAP matrix for test set

    # Record Selection
    col_sel, col_info = st.columns([2, 1])
    with col_sel:
        record_idx = st.selectbox(
            "Select Company Record to Explain:",
            options=list(range(len(test_df))),
            format_func=lambda idx: f"Dataset Record #{idx} — Predicted Risk: {test_probs[idx]*100:.1f}%",
            index=311 if len(test_df) > 311 else 0,
        )

    prob = float(test_probs[record_idx])
    rec_shap = shap_vals[record_idx]

    # Individual Record Explanation
    st.subheader(f"🔍 Record #{record_idx} Individual Explanation Breakdown")

    # Waterfall / Bar chart of top drivers
    fig_shap = render_shap_waterfall_chart(
        feature_names=feature_names,
        shap_values=rec_shap,
        base_val=0.0628,
        predicted_val=prob,
        max_display=10,
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    # Positive vs Negative Contributor Cards
    col_pos, col_neg = st.columns(2)
    df_factors = pd.DataFrame({
        "feature": feature_names,
        "value": test_df.iloc[record_idx].drop("Bankrupt?").values,
        "shap": rec_shap,
    })

    pos_drivers = df_factors[df_factors["shap"] > 0].sort_values(by="shap", ascending=False).head(5)
    neg_drivers = df_factors[df_factors["shap"] < 0].sort_values(by="shap", ascending=True).head(5)

    with col_pos:
        st.markdown("#### 🔴 Top Risk-Increasing Drivers")
        st.caption("Financial factors that push the model's predicted bankruptcy probability higher.")
        for _, row in pos_drivers.iterrows():
            st.markdown(
                f"- **{row['feature']}**: value = `{row['value']:.4f}` $\\rightarrow$ **+{row['shap']:.4f}** SHAP impact"
            )

    with col_neg:
        st.markdown("#### 🔵 Top Risk-Reducing Buffers")
        st.caption("Financial factors that stabilize the company and pull the predicted risk lower.")
        if len(neg_drivers) == 0:
            st.info("No significant risk-reducing factors identified for this distressed record.")
        else:
            for _, row in neg_drivers.iterrows():
                st.markdown(
                    f"- **{row['feature']}**: value = `{row['value']:.4f}` $\\rightarrow$ **{row['shap']:.4f}** SHAP impact"
                )

    st.markdown("---")

    # Global SHAP Feature Importance
    st.subheader("🌐 Global Model-Level SHAP Feature Importance")
    st.markdown("Across the entire test cohort, which financial features have the largest aggregate impact on bankruptcy risk?")

    glob_csv = METRICS_DIR / "shap_global_importance.csv"
    if glob_csv.exists():
        glob_df = pd.read_csv(glob_csv)
        top15 = glob_df.head(15)

        fig_glob = px.bar(
            top15.sort_values(by="mean_abs_shap", ascending=True),
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            labels={"mean_abs_shap": "Mean |SHAP Value|", "feature": "Financial Indicator"},
            title="Top 15 Global Financial Risk Predictors",
            color_discrete_sequence=["#1e3a8a"],
        )
        fig_glob.update_layout(template="plotly_white", height=420, margin=dict(l=200, r=30, t=50, b=30))
        st.plotly_chart(fig_glob, use_container_width=True)

    # Static Publication Plots Expandable
    with st.expander("🖼️ View High-Resolution Publication SHAP Plots (Beeswarm & Summary)"):
        b_path = FIG_SHAP_DIR / "shap_beeswarm.png"
        bar_path = FIG_SHAP_DIR / "shap_summary_bar.png"
        if b_path.exists():
            st.image(str(b_path), caption="SHAP Beeswarm Plot (Feature Value Impact Distribution)", use_container_width=True)
        if bar_path.exists():
            st.image(str(bar_path), caption="SHAP Global Mean Absolute Importance Bar Chart", use_container_width=True)
