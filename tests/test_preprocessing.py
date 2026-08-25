"""
Unit tests for data preprocessing and feature engineering (Segment 2).
"""
import numpy as np
import pandas as pd
import pytest

from src.data.load_data import load_raw_data
from src.data.preprocess import (
    clean_column_names,
    identify_constant_features,
    split_data,
    build_preprocessor,
    run_preprocessing_pipeline,
)


@pytest.fixture
def raw_df():
    return clean_column_names(load_raw_data())


def test_clean_column_names(raw_df):
    """Verify that column names do not have leading or trailing whitespaces."""
    for col in raw_df.columns:
        assert col == col.strip(), f"Column '{col}' has unstripped whitespace."


def test_identify_constant_features(raw_df):
    """Verify that constant/zero variance features are identified dynamically."""
    constant_cols = identify_constant_features(raw_df, target_col="Bankrupt?")
    assert isinstance(constant_cols, list)
    # 'Net Income Flag' is known to be constant across all records
    assert "Net Income Flag" in constant_cols
    for col in constant_cols:
        assert raw_df[col].nunique() <= 1 or raw_df[col].var() == 0


def test_stratified_split(raw_df):
    """Verify train/test split maintains target class ratio (stratification)."""
    X_train, X_test, y_train, y_test = split_data(raw_df, target_col="Bankrupt?", test_size=0.2, random_state=42)
    
    total_pos_rate = raw_df["Bankrupt?"].mean()
    train_pos_rate = y_train.mean()
    test_pos_rate = y_test.mean()

    # The positive class proportion should be very close across splits
    assert abs(train_pos_rate - total_pos_rate) < 0.005
    assert abs(test_pos_rate - total_pos_rate) < 0.005
    assert len(X_train) + len(X_test) == len(raw_df)
    assert len(y_train) + len(y_test) == len(raw_df)


def test_preprocessor_pipeline_no_nans_and_leakage_prevention(raw_df):
    """Verify preprocessing pipeline fits on train and transforms test without leakage or NaNs."""
    constant_cols = identify_constant_features(raw_df, target_col="Bankrupt?")
    df_clean = raw_df.drop(columns=constant_cols)

    X_train, X_test, y_train, y_test = split_data(df_clean, target_col="Bankrupt?", test_size=0.2, random_state=42)
    preprocessor = build_preprocessor(scale=True, imputation_strategy="median")

    # Fit on training data ONLY
    preprocessor.fit(X_train)
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Check output shapes
    assert X_train_trans.shape[0] == X_train.shape[0]
    assert X_test_trans.shape[0] == X_test.shape[0]
    assert X_train_trans.shape[1] == X_train.shape[1]
    assert X_test_trans.shape[1] == X_test.shape[1]

    # Check no NaN values
    assert not np.isnan(X_train_trans).any()
    assert not np.isnan(X_test_trans).any()


def test_end_to_end_preprocessing_pipeline():
    """Verify the full run_preprocessing_pipeline returns expected dictionary and files."""
    results = run_preprocessing_pipeline()
    assert "X_train" in results
    assert "X_test" in results
    assert "y_train" in results
    assert "y_test" in results
    assert len(results["removed_features"]) >= 1
    assert "Net Income Flag" in results["removed_features"]
    assert len(results["retained_features"]) > 90
