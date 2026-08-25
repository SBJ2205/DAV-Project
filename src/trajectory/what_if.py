"""
What-If Risk Trajectory / Sensitivity Engine (Segment 8).
Simulates model sensitivity to hypothetical changes in key financial ratios.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.utils.logger import setup_logger

logger = setup_logger("what_if_engine")


def generate_what_if_trajectory(
    company_features: pd.Series,
    feature_name: str,
    model: Any,
    preprocessor: Any,
    scenario_values: Optional[List[float]] = None,
    relative_percents: Optional[List[float]] = None,
    min_bound: Optional[float] = None,
    max_bound: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Generate model-predicted sensitivity trajectory for a single financial feature.

    Args:
        company_features: Full feature vector (pd.Series) for a single record.
        feature_name: Name of the financial feature to vary.
        model: Trained classifier (Random Forest, XGBoost, or Logistic Regression).
        preprocessor: Fitted preprocessor used for the model.
        scenario_values: Specific absolute scenario values.
        relative_percents: List of relative percentages to simulate (e.g., [-30, -20, -10, 0, 10, 20, 30]).
        min_bound: Minimum allowed value for simulation bounds.
        max_bound: Maximum allowed value for simulation bounds.

    Returns:
        Dictionary containing trajectory DataFrame, current risk, scenario metrics, and Plotly chart.
    """
    if feature_name not in company_features.index:
        raise ValueError(f"Feature '{feature_name}' not found in company feature vector.")

    current_val = float(company_features[feature_name])
    feature_cols = company_features.index.tolist()

    # Determine scenario points
    if scenario_values is None:
        if relative_percents is None:
            relative_percents = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]

        calculated_values = []
        for pct in relative_percents:
            val = current_val * (1.0 + pct / 100.0) if current_val != 0 else (pct / 100.0)
            calculated_values.append(val)
        scenario_values = sorted(list(set(calculated_values)))

    # Ensure current value is included
    if current_val not in scenario_values:
        scenario_values.append(current_val)
        scenario_values = sorted(scenario_values)

    # Apply realistic bounds if supplied
    if min_bound is not None:
        scenario_values = [max(v, min_bound) for v in scenario_values]
    if max_bound is not None:
        scenario_values = [min(v, max_bound) for v in scenario_values]
    scenario_values = sorted(list(set(scenario_values)))

    # Build matrix of synthetic records: keep 93 other features frozen
    records = []
    for val in scenario_values:
        row_copy = company_features.copy()
        row_copy[feature_name] = val
        records.append(row_copy)

    df_scenarios = pd.DataFrame(records, columns=feature_cols)

    # Transform through preprocessor
    X_mat = preprocessor.transform(df_scenarios)
    predicted_probs = model.predict_proba(X_mat)[:, 1]

    # Current risk
    current_idx = scenario_values.index(current_val)
    current_risk = float(predicted_probs[current_idx])

    trajectory_df = pd.DataFrame({
        "scenario_value": [round(v, 4) for v in scenario_values],
        "predicted_risk": [round(p, 4) for p in predicted_probs],
        "risk_percentage": [round(p * 100, 2) for p in predicted_probs],
        "is_current": [bool(v == current_val) for v in scenario_values],
    })

    # Summary text
    min_risk = float(predicted_probs.min())
    max_risk = float(predicted_probs.max())
    summary_text = (
        f"Under simulated sensitivity analysis, varying '{feature_name}' from {min(scenario_values):.3f} to {max(scenario_values):.3f} "
        f"shifts the model-predicted bankruptcy probability between {min_risk*100:.1f}% and {max_risk*100:.1f}% "
        f"(baseline current prediction: {current_risk*100:.1f}% at value {current_val:.3f})."
    )

    # Build interactive Plotly chart
    fig = go.Figure()

    # Trajectory line
    fig.add_trace(go.Scatter(
        x=trajectory_df["scenario_value"],
        y=trajectory_df["predicted_risk"] * 100,
        mode="lines+markers",
        name="Risk Trajectory",
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=8, color="#1f77b4"),
        hovertemplate="Feature Value: %{x:.4f}<br>Predicted Risk: %{y:.1f}%<extra></extra>",
    ))

    # Highlight current value point
    fig.add_trace(go.Scatter(
        x=[current_val],
        y=[current_risk * 100],
        mode="markers",
        name="Current Value",
        marker=dict(color="#d9534f", size=14, symbol="diamond"),
        hovertemplate="<b>Current Status</b><br>Value: %{x:.4f}<br>Risk: %{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        title=f"What-If Risk Sensitivity Trajectory: {feature_name}",
        xaxis_title=f"{feature_name} (Simulated Scenario Value)",
        yaxis_title="Model-Predicted Bankruptcy Probability (%)",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
        hovermode="closest",
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return {
        "feature_name": feature_name,
        "current_value": current_val,
        "current_risk": round(current_risk, 4),
        "trajectory_df": trajectory_df,
        "summary_text": summary_text,
        "fig": fig,
    }
