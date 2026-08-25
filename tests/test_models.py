"""
Unit tests for model loading, prediction consistency, and probability bounds (Segments 4, 5, 6).
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"


@pytest.fixture
def test_data():
    df = pd.read_csv(TEST_DATA_PATH)
    X = df.drop(columns=["Bankrupt?"])
    y = df["Bankrupt?"].values
    return X, y


@pytest.mark.parametrize("model_file,preproc_file", [
    ("logistic_model.pkl", "logistic_preprocessor.pkl"),
    ("random_forest_model.pkl", "tree_preprocessor.pkl"),
    ("xgboost_model.pkl", "tree_preprocessor.pkl"),
])
def test_model_loading_and_prediction(model_file, preproc_file, test_data):
    """Test that all three models load correctly, predict valid probabilities, and maintain shape."""
    X_test, y_test = test_data

    model_path = MODELS_DIR / model_file
    preproc_path = MODELS_DIR / preproc_file
    assert model_path.exists(), f"Model file {model_path} missing."
    assert preproc_path.exists(), f"Preprocessor file {preproc_path} missing."

    model = joblib.load(model_path)
    preproc = joblib.load(preproc_path)

    X_transformed = preproc.transform(X_test)
    probs = model.predict_proba(X_transformed)[:, 1]
    preds = model.predict(X_transformed)

    # Validate output length and dimensions
    assert len(probs) == len(X_test)
    assert len(preds) == len(X_test)

    # Validate probability bounds strictly between 0 and 1
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)

    # Validate binary predictions only 0 or 1
    assert set(np.unique(preds)).issubset({0, 1})


def test_final_model_selection_metadata():
    """Verify that final_model_selection.json exists and has valid structure."""
    selection_file = METRICS_DIR / "final_model_selection.json"
    assert selection_file.exists(), f"Selection file {selection_file} does not exist."

    with open(selection_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "selected_model" in data
    assert data["selected_model"] in ["Logistic Regression", "Random Forest", "XGBoost"]
    assert "selection_rationale" in data
    assert "test_metrics" in data
