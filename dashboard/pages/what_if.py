"""
Dashboard Page: What-If Risk Trajectory & Sensitivity Engine.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from src.trajectory.what_if import generate_what_if_trajectory

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def render_what_if_page(data_bundle: dict):
    st.markdown("## 🔮 What-If Risk Trajectory & Sensitivity Engine")
    st.markdown(
        """
        Simulate hypothetical managerial interventions and analyze how changes in core financial ratios 
        shift the model's predicted bankruptcy probability.
        """
    )

    st.warning(
        "⚠️ **Sensitivity Disclaimer**: This engine performs controlled single-variable sensitivity analysis. "
        "It visualizes model responsiveness under hypothetical scenarios, holding all other financial variables constant."
    )

    test_df = data_bundle["test_df"]
    test_probs = data_bundle["test_probs"]
    feature_names = data_bundle["feature_names"]
    model = data_bundle["model"]
    tree_prep = data_bundle["tree_prep"]
    shap_vals = data_bundle["test_shap_vals"]

    # 1. Select Record
    col_rec, col_feat = st.columns([1, 2])
    with col_rec:
        record_idx = st.selectbox(
            "Select Company Record:",
            options=list(range(len(test_df))),
            format_func=lambda idx: f"Record #{idx} (Current Risk: {test_probs[idx]*100:.1f}%)",
            index=311 if len(test_df) > 311 else 0,
        )

    rec_features = test_df.iloc[record_idx].drop("Bankrupt?")
    current_risk = float(test_probs[record_idx])

    # Suggest top SHAP features for this record
    rec_shap = shap_vals[record_idx]
    top_indices = np.argsort(np.abs(rec_shap))[::-1][:6]
    top_features = [feature_names[i] for i in top_indices]

    with col_feat:
        selected_feature = st.selectbox(
            "Select Financial Ratio to Simulate (Top SHAP Drivers Recommended):",
            options=top_features + [f for f in feature_names if f not in top_features],
        )

    current_feat_val = float(rec_features[selected_feature])

    st.markdown("---")

    # 2. Simulation Controls
    st.subheader(f"🎛️ Simulation Parameters for: `{selected_feature}`")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(label="Current Feature Value", value=f"{current_feat_val:.4f}")
    with c2:
        st.metric(label="Current Predicted Risk", value=f"{current_risk*100:.1f}%")
    with c3:
        sim_range = st.slider(
            "Simulation Range (% Variation):",
            min_value=10,
            max_value=50,
            value=30,
            step=5,
            help="Defines the percentage spread around the current value.",
        )

    # Relative percentages
    percents = list(np.linspace(-sim_range, sim_range, 11))

    # Generate Trajectory
    result = generate_what_if_trajectory(
        company_features=rec_features,
        feature_name=selected_feature,
        model=model,
        preprocessor=tree_prep,
        relative_percents=percents,
    )

    traj_df = result["trajectory_df"]

    # 3. Trajectory Visualization
    st.plotly_chart(result["fig"], use_container_width=True)

    # 4. Scenario Evaluation & Summary Callout
    st.markdown("### 📋 Scenario Trajectory Data Table")
    col_tbl, col_msg = st.columns([1, 1])

    with col_tbl:
        display_tbl = traj_df.copy()
        display_tbl["scenario_value"] = display_tbl["scenario_value"].apply(lambda v: f"{v:.4f}")
        display_tbl["risk_percentage"] = display_tbl["risk_percentage"].apply(lambda r: f"{r:.2f}%")
        display_tbl["is_current"] = display_tbl["is_current"].apply(lambda x: "📍 Current" if x else "")
        display_tbl.columns = ["Scenario Value", "Predicted Risk (Prob)", "Risk (%)", "Status"]
        st.dataframe(display_tbl, use_container_width=True, hide_index=True)

    with col_msg:
        st.markdown("#### 💡 Trajectory Insights")
        st.info(result["summary_text"])

        # Best improvement scenario
        min_row = traj_df.loc[traj_df["predicted_risk"].idxmin()]
        delta_p = (min_row["predicted_risk"] - current_risk) * 100
        if delta_p < 0:
            st.success(
                f"🌟 **Optimal Simulated Target**: Modifying `{selected_feature}` to `{min_row['scenario_value']:.4f}` "
                f"reduces model-predicted insolvency risk by **{abs(delta_p):.1f} percentage points** "
                f"(from {current_risk*100:.1f}% down to {min_row['predicted_risk']*100:.1f}%)."
            )
        else:
            st.warning(
                f"Varying this single parameter did not reduce risk below current baseline ({current_risk*100:.1f}%). "
                f"Multi-ratio remediation may be necessary."
            )
