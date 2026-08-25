"""
Reusable UI Components: Risk Card & Status Badges.
"""
import streamlit as st


def render_risk_card(predicted_prob: float, record_id: int):
    """
    Render a prominent, beautifully styled risk badge and probability card.
    """
    pct = predicted_prob * 100

    if predicted_prob >= 0.60:
        tier = "HIGH RISK"
        bg_color = "#fdf2f2"
        border_color = "#f98080"
        text_color = "#c81e1e"
        badge_bg = "#e02424"
        desc = "Strong multiple financial distress signals detected across leverage and profitability ratios."
    elif predicted_prob >= 0.30:
        tier = "MODERATE RISK"
        bg_color = "#fdf8f6"
        border_color = "#fbd5c0"
        text_color = "#9f580a"
        badge_bg = "#d97706"
        desc = "Moderate distress indicators observed; recommended for proactive financial monitoring."
    else:
        tier = "LOW RISK (HEALTHY)"
        bg_color = "#f3faf7"
        border_color = "#84e1bc"
        text_color = "#03543f"
        badge_bg = "#0e9f6e"
        desc = "Financial ratios align closely with stable, solvent benchmark profiles."

    st.markdown(
        f"""
        <div style="background-color: {bg_color}; border: 1.5px solid {border_color}; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 1.1rem; font-weight: 700; color: #1e293b;">Dataset Record #{record_id}</span>
                <span style="background-color: {badge_bg}; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 700; letter-spacing: 0.5px;">
                    {tier}
                </span>
            </div>
            <div style="display: flex; align-items: baseline; margin: 12px 0;">
                <span style="font-size: 2.5rem; font-weight: 800; color: {text_color}; line-height: 1;">
                    {pct:.1f}%
                </span>
                <span style="margin-left: 10px; font-size: 0.95rem; color: #475569; font-weight: 500;">
                    Model-Predicted Bankruptcy Probability
                </span>
            </div>
            <div style="width: 100%; background-color: #e2e8f0; height: 10px; border-radius: 5px; overflow: hidden; margin-bottom: 10px;">
                <div style="width: {min(pct, 100):.1f}%; background-color: {badge_bg}; height: 100%; border-radius: 5px; transition: width 0.5s ease;"></div>
            </div>
            <div style="font-size: 0.85rem; color: #64748b; margin-top: 4px;">
                {desc}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
