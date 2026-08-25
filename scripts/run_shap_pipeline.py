"""
Script to execute Segment 7 SHAP Explainability on the selected model.
"""
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.shap_explainer import (
    compute_global_shap_importance,
    compute_shap_values,
    create_shap_explainer,
    explain_single_record,
    generate_global_shap_plots,
)
from src.utils.logger import setup_logger

logger = setup_logger("shap_pipeline")
MODELS_DIR = PROJECT_ROOT / "models"
FIG_SHAP_DIR = PROJECT_ROOT / "outputs" / "figures" / "shap"


def run_shap_analysis():
    FIG_SHAP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data and models
    logger.info("Loading test dataset and selected model for SHAP explainability...")
    train_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train.csv")
    test_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")

    target_col = "Bankrupt?"
    X_train = train_df.drop(columns=[target_col])
    X_test = test_df.drop(columns=[target_col])
    feature_names = X_test.columns.tolist()

    # Load Random Forest model (selected best overall)
    rf_model = joblib.load(MODELS_DIR / "random_forest_model.pkl")
    tree_prep = joblib.load(MODELS_DIR / "tree_preprocessor.pkl")

    X_train_mat = tree_prep.transform(X_train)
    X_test_mat = tree_prep.transform(X_test)

    # Sample background for SHAP to optimize computation speed
    np.random.seed(42)
    bg_indices = np.random.choice(len(X_train_mat), size=min(100, len(X_train_mat)), replace=False)
    X_bg = X_train_mat[bg_indices]

    # Sample test instances for global explanation
    test_sample_indices = np.random.choice(len(X_test_mat), size=min(200, len(X_test_mat)), replace=False)
    X_test_sample = X_test_mat[test_sample_indices]
    X_test_sample_df = pd.DataFrame(X_test_sample, columns=feature_names)

    # 2. Build Explainer & Compute SHAP Values
    logger.info("Creating TreeExplainer...")
    explainer = shap.TreeExplainer(rf_model, data=X_bg, model_output="probability")
    shap_vals, base_val = compute_shap_values(explainer, X_test_sample)

    logger.info(f"Computed SHAP values with shape: {shap_vals.shape}, Base value: {base_val:.4f}")

    # 3. Global SHAP Importance & Visualizations
    importance_df = compute_global_shap_importance(shap_vals, feature_names, save_csv=True)
    logger.info(f"Top 5 Global Features by mean |SHAP|:\n{importance_df.head(5)}")

    generate_global_shap_plots(explainer, shap_vals, X_test_sample_df, max_display=15)

    # 4. Individual Company Record Explanations (High-Risk vs Low-Risk)
    test_probs = rf_model.predict_proba(X_test_mat)[:, 1]
    high_risk_idx = int(np.argmax(test_probs))
    low_risk_idx = int(np.argmin(test_probs))

    # Compute exact SHAP for high-risk and low-risk records
    high_rec_mat = X_test_mat[[high_risk_idx]]
    low_rec_mat = X_test_mat[[low_risk_idx]]

    high_shap, _ = compute_shap_values(explainer, high_rec_mat)
    low_shap, _ = compute_shap_values(explainer, low_rec_mat)

    high_explanation = explain_single_record(
        high_shap[0], feature_names, X_test.iloc[high_risk_idx].values, test_probs[high_risk_idx], record_id=high_risk_idx
    )
    low_explanation = explain_single_record(
        low_shap[0], feature_names, X_test.iloc[low_risk_idx].values, test_probs[low_risk_idx], record_id=low_risk_idx
    )

    logger.info(f"High-Risk Record #{high_risk_idx} Explanation: {high_explanation['summary_explanation']}")
    logger.info(f"Low-Risk Record #{low_risk_idx} Explanation: {low_explanation['summary_explanation']}")

    # Save individual waterfall plots
    try:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        shap_exp_obj = shap.Explanation(
            values=high_shap[0],
            base_values=base_val,
            data=X_test.iloc[high_risk_idx].values,
            feature_names=feature_names,
        )
        shap.plots.waterfall(shap_exp_obj, max_display=10, show=False)
        plt.title(f"SHAP Waterfall: High-Risk Record #{high_risk_idx} (Risk: {test_probs[high_risk_idx]*100:.1f}%)", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIG_SHAP_DIR / "high_risk_waterfall.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        shap_exp_obj_low = shap.Explanation(
            values=low_shap[0],
            base_values=base_val,
            data=X_test.iloc[low_risk_idx].values,
            feature_names=feature_names,
        )
        shap.plots.waterfall(shap_exp_obj_low, max_display=10, show=False)
        plt.title(f"SHAP Waterfall: Low-Risk Record #{low_risk_idx} (Risk: {test_probs[low_risk_idx]*100:.1f}%)", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIG_SHAP_DIR / "low_risk_waterfall.png")
        plt.close()
        logger.info(f"Saved waterfall plots to {FIG_SHAP_DIR}")
    except Exception as e:
        logger.warning(f"Note on waterfall plot rendering: {e}")

    logger.info("SHAP explainability pipeline completed successfully!")


if __name__ == "__main__":
    run_shap_analysis()
