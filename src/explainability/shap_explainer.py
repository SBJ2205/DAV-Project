"""
SHAP Explainability Module (Segment 7).
Provides model-agnostic and tree-specific SHAP explanations with strict non-causal language.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
from src.utils.logger import setup_logger

logger = setup_logger("shap_explainer")
FIG_SHAP_DIR = PROJECT_ROOT / "outputs" / "figures" / "shap"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"


def create_shap_explainer(model: Any, X_background: np.ndarray) -> shap.Explainer:
    """
    Create appropriate SHAP explainer for given model.

    Args:
        model: Trained sklearn, xgboost, or ensemble model.
        X_background: Background feature matrix for expected values.

    Returns:
        Configured shap.Explainer instance.
    """
    model_type = type(model).__name__
    logger.info(f"Initializing SHAP explainer for model type: {model_type}")

    if "RandomForest" in model_type or "XGB" in model_type:
        # Use TreeExplainer for tree ensembles
        return shap.TreeExplainer(model, data=X_background, model_output="probability" if "RandomForest" in model_type else "raw")
    elif "LogisticRegression" in model_type:
        return shap.LinearExplainer(model, X_background)
    else:
        # Generic fallback
        return shap.Explainer(model.predict_proba, X_background)


def compute_shap_values(explainer: shap.Explainer, X: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute SHAP values and base value for a dataset.

    Args:
        explainer: Initialized shap.Explainer.
        X: Feature matrix.

    Returns:
        (shap_values_array, base_value)
    """
    shap_explanation = explainer(X)
    shap_vals = shap_explanation.values

    # If binary classification returns 3D array (n_samples, n_features, 2), select positive class (index 1)
    if shap_vals.ndim == 3 and shap_vals.shape[-1] == 2:
        shap_vals = shap_vals[:, :, 1]
        base_val = float(shap_explanation.base_values[0, 1]) if hasattr(shap_explanation.base_values, "__getitem__") else float(shap_explanation.base_values)
    else:
        base_val = float(shap_explanation.base_values[0]) if hasattr(shap_explanation.base_values, "__getitem__") else float(shap_explanation.base_values)

    return shap_vals, base_val


def compute_global_shap_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    save_csv: bool = True,
) -> pd.DataFrame:
    """
    Calculate mean absolute SHAP values for global feature importance.

    Returns:
        Sorted DataFrame with columns ['feature', 'mean_abs_shap', 'mean_shap'].
    """
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    mean_raw = np.mean(shap_values, axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
        "mean_shap": mean_raw,
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    if save_csv:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = METRICS_DIR / "shap_global_importance.csv"
        importance_df.to_csv(csv_path, index=False)
        logger.info(f"Saved Global SHAP Importance to: {csv_path}")

    return importance_df


def generate_global_shap_plots(
    explainer: shap.Explainer,
    shap_values: np.ndarray,
    X_df: pd.DataFrame,
    max_display: int = 20,
):
    """Generate and save SHAP Summary Bar Plot and Beeswarm Plot."""
    FIG_SHAP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. SHAP Bar Plot
    plt.figure(figsize=(10, 8), dpi=300)
    shap.summary_plot(shap_values, X_df, plot_type="bar", max_display=max_display, show=False)
    plt.title(f"Global SHAP Feature Importance (Top {max_display} Features)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    bar_path = FIG_SHAP_DIR / "shap_summary_bar.png"
    plt.savefig(bar_path)
    plt.close()
    logger.info(f"Saved SHAP bar plot to: {bar_path}")

    # 2. SHAP Beeswarm / Dot Plot
    plt.figure(figsize=(10, 8), dpi=300)
    shap.summary_plot(shap_values, X_df, max_display=max_display, show=False)
    plt.title(f"SHAP Beeswarm: Impact of Feature Value on Model Output", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    beeswarm_path = FIG_SHAP_DIR / "shap_beeswarm.png"
    plt.savefig(beeswarm_path)
    plt.close()
    logger.info(f"Saved SHAP beeswarm plot to: {beeswarm_path}")


def explain_single_record(
    record_shap: np.ndarray,
    feature_names: List[str],
    feature_values: np.ndarray,
    predicted_prob: float,
    record_id: Union[int, str] = 0,
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Format individual company record explanation into structured risk drivers.

    Returns:
        Dictionary containing top risk-increasing and risk-reducing factors.
    """
    df_factors = pd.DataFrame({
        "feature": feature_names,
        "feature_value": feature_values,
        "shap_value": record_shap,
        "abs_shap": np.abs(record_shap),
    })

    risk_increasing = df_factors[df_factors["shap_value"] > 0].sort_values(by="shap_value", ascending=False).head(top_n)
    risk_reducing = df_factors[df_factors["shap_value"] < 0].sort_values(by="shap_value", ascending=True).head(top_n)

    summary_text = (
        f"Dataset Record #{record_id} has a model-predicted bankruptcy probability of {predicted_prob*100:.1f}%. "
        f"The prediction is most strongly elevated by {', '.join(risk_increasing['feature'].head(3).tolist())}, "
        f"while {', '.join(risk_reducing['feature'].head(2).tolist()) if len(risk_reducing) > 0 else 'few indicators'} "
        f"mitigate the estimated risk."
    )

    return {
        "record_id": record_id,
        "predicted_probability": round(predicted_prob, 4),
        "risk_category": "High Risk" if predicted_prob >= 0.60 else ("Medium Risk" if predicted_prob >= 0.30 else "Low Risk"),
        "risk_increasing_factors": risk_increasing.to_dict(orient="records"),
        "risk_reducing_factors": risk_reducing.to_dict(orient="records"),
        "summary_explanation": summary_text,
    }
