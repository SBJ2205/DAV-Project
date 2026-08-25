"""
Unit tests for What-If Risk Trajectory Engine (Segment 8).
"""
from pathlib import Path
import joblib
import pandas as pd
import pytest

from src.trajectory.what_if import generate_what_if_trajectory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"


@pytest.fixture
def whatif_setup():
    df = pd.read_csv(TEST_DATA_PATH)
    X = df.drop(columns=["Bankrupt?"])
    model = joblib.load(MODELS_DIR / "random_forest_model.pkl")
    tree_prep = joblib.load(MODELS_DIR / "tree_preprocessor.pkl")
    return X.iloc[0], model, tree_prep


def test_whatif_trajectory_generation(whatif_setup):
    """Verify that sensitivity trajectory produces valid probabilities, correct shapes, and charts."""
    company_row, model, tree_prep = whatif_setup
    feature_name = company_row.index[0]  # First financial feature

    result = generate_what_if_trajectory(
        company_features=company_row,
        feature_name=feature_name,
        model=model,
        preprocessor=tree_prep,
        relative_percents=[-30, -15, 0, 15, 30],
    )

    assert result["feature_name"] == feature_name
    assert 0.0 <= result["current_risk"] <= 1.0
    assert isinstance(result["trajectory_df"], pd.DataFrame)
    assert len(result["trajectory_df"]) >= 5

    traj_df = result["trajectory_df"]
    assert "scenario_value" in traj_df.columns
    assert "predicted_risk" in traj_df.columns
    assert "is_current" in traj_df.columns

    # Probability bounds
    assert traj_df["predicted_risk"].min() >= 0.0
    assert traj_df["predicted_risk"].max() <= 1.0

    # Ensure current point is tagged
    assert traj_df["is_current"].sum() >= 1

    # Check chart object
    assert result["fig"] is not None
    assert "summary_text" in result
