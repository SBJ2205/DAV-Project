"""
Unit tests for SHAP Explainability module (Segment 7).
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.explainability.shap_explainer import (
    compute_global_shap_importance,
    compute_shap_values,
    create_shap_explainer,
    explain_single_record,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"


@pytest.fixture
def dataset_and_model():
    df = pd.read_csv(TEST_DATA_PATH)
    X = df.drop(columns=["Bankrupt?"])
    model = joblib.load(MODELS_DIR / "random_forest_model.pkl")
    tree_prep = joblib.load(MODELS_DIR / "tree_preprocessor.pkl")
    X_mat = tree_prep.transform(X)
    return X, X_mat, model


def test_shap_explainer_initialization_and_output_shape(dataset_and_model):
    """Verify SHAP explainer initializes and computes expected output shapes."""
    X_df, X_mat, model = dataset_and_model
    feature_names = X_df.columns.tolist()

    # Small background and test sample for speed
    X_bg = X_mat[:20]
    X_sample = X_mat[:5]

    explainer = create_shap_explainer(model, X_bg)
    shap_vals, base_val = compute_shap_values(explainer, X_sample)

    assert shap_vals.shape == X_sample.shape, f"SHAP values shape {shap_vals.shape} != sample shape {X_sample.shape}"
    assert isinstance(base_val, float)

    # Test global importance
    importance_df = compute_global_shap_importance(shap_vals, feature_names, save_csv=False)
    assert len(importance_df) == len(feature_names)
    assert "mean_abs_shap" in importance_df.columns
    assert "mean_shap" in importance_df.columns


def test_explain_single_record(dataset_and_model):
    """Verify individual record explanation structure and non-causal keys."""
    X_df, X_mat, model = dataset_and_model
    feature_names = X_df.columns.tolist()

    X_bg = X_mat[:20]
    explainer = create_shap_explainer(model, X_bg)
    shap_vals, _ = compute_shap_values(explainer, X_mat[[0]])

    prob = float(model.predict_proba(X_mat[[0]])[0, 1])
    exp = explain_single_record(shap_vals[0], feature_names, X_df.iloc[0].values, prob, record_id=0)

    assert exp["record_id"] == 0
    assert "predicted_probability" in exp
    assert exp["risk_category"] in ["Low Risk", "Medium Risk", "High Risk"]
    assert "risk_increasing_factors" in exp
    assert "risk_reducing_factors" in exp
    assert "summary_explanation" in exp
