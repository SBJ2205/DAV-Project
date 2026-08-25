"""
Reusable UI Components: Plotly Visualizations.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_risk_distribution_chart(probabilities: np.ndarray, current_idx: Optional[int] = None) -> go.Figure:
    """Create interactive histogram and density of risk distribution."""
    df_probs = pd.DataFrame({"Risk": probabilities * 100})
    
    fig = px.histogram(
        df_probs,
        x="Risk",
        nbins=30,
        color_discrete_sequence=["#3b82f6"],
        labels={"Risk": "Predicted Bankruptcy Probability (%)"},
        title="Distribution of Predicted Bankruptcy Risk Across Test Cohort",
    )
    
    # Add vertical line for high risk cutoff (60%)
    fig.add_vline(x=60, line_width=2, line_dash="dash", line_color="#ef4444", annotation_text="High Risk (60%)", annotation_position="top right")
    fig.add_vline(x=30, line_width=2, line_dash="dash", line_color="#f59e0b", annotation_text="Moderate (30%)", annotation_position="top right")
    
    if current_idx is not None and 0 <= current_idx < len(probabilities):
        curr_p = probabilities[current_idx] * 100
        fig.add_vline(x=curr_p, line_width=3, line_color="#10b981", annotation_text=f"Record #{current_idx} ({curr_p:.1f}%)", annotation_position="top left")

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Predicted Bankruptcy Probability (%)",
        yaxis_title="Company Count",
        margin=dict(l=30, r=30, t=50, b=30),
        height=320,
    )
    return fig


def render_shap_waterfall_chart(
    feature_names: List[str],
    shap_values: List[float],
    base_val: float,
    predicted_val: float,
    max_display: int = 8,
) -> go.Figure:
    """Create interactive horizontal bar / waterfall explanation chart."""
    df = pd.DataFrame({
        "feature": feature_names,
        "shap": shap_values,
        "abs_shap": np.abs(shap_values),
    }).sort_values(by="abs_shap", ascending=False).head(max_display)

    # Sort so most positive at top
    df = df.sort_values(by="shap", ascending=True)

    colors = ["#ef4444" if s > 0 else "#3b82f6" for s in df["shap"]]

    fig = go.Figure(go.Bar(
        x=df["shap"],
        y=df["feature"],
        orientation="h",
        marker=dict(color=colors),
        text=[f"{s:+.4f}" for s in df["shap"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>SHAP Impact: %{x:+.4f}<extra></extra>",
    ))

    fig.update_layout(
        title="Top Financial Feature Contributions (SHAP Impact on Predicted Risk)",
        xaxis_title="SHAP Value (Red = Increases Risk | Blue = Decreases Risk)",
        yaxis_title="",
        template="plotly_white",
        margin=dict(l=150, r=40, t=50, b=40),
        height=380,
    )
    return fig
