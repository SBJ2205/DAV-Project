"""
Data Preprocessing and Feature Engineering Module (Segment 2).
Provides leakage-free preprocessing pipelines for Logistic Regression and Tree models.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.load_data import load_raw_data
from src.utils.logger import setup_logger

logger = setup_logger("preprocess")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "data_quality"


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading and trailing whitespace from column names."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def identify_constant_features(df: pd.DataFrame, target_col: str = "Bankrupt?") -> List[str]:
    """
    Automatically identify features with only one unique value or zero variance.

    Args:
        df: DataFrame containing features.
        target_col: Target column name to exclude from check.

    Returns:
        List of column names that are constant/zero-variance.
    """
    features = [c for c in df.columns if c != target_col]
    constant_cols = []
    for col in features:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
        elif np.issubdtype(df[col].dtype, np.number) and df[col].var() == 0:
            constant_cols.append(col)
    logger.info(f"Identified {len(constant_cols)} constant/zero-variance feature(s): {constant_cols}")
    return constant_cols


def split_data(
    df: pd.DataFrame,
    target_col: str = "Bankrupt?",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a stratified train/test split to maintain class balance without data leakage.

    Args:
        df: Input DataFrame.
        target_col: Target column name.
        test_size: Fraction of data to reserve for test set.
        random_state: Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in DataFrame.")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # Validate target has binary values 0 and 1
    unique_targets = set(y.unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(f"Target contains unexpected values: {unique_targets}. Expected subset of {{0, 1}}.")

    logger.info(f"Splitting dataset: test_size={test_size}, random_state={random_state}, stratify=True")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Train shape: X_train={X_train.shape}, y_train={y_train.shape} (Positives: {y_train.sum()})")
    logger.info(f"Test shape: X_test={X_test.shape}, y_test={y_test.shape} (Positives: {y_test.sum()})")
    return X_train, X_test, y_train, y_test


def build_preprocessor(
    scale: bool = True,
    imputation_strategy: str = "median",
) -> Pipeline:
    """
    Build a sklearn preprocessing pipeline.

    Args:
        scale: If True, includes StandardScaler (for Logistic Regression).
        imputation_strategy: Imputation strategy for SimpleImputer ('median' recommended).

    Returns:
        sklearn Pipeline.
    """
    steps = [("imputer", SimpleImputer(strategy=imputation_strategy))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def run_preprocessing_pipeline(
    raw_data_path: Optional[Union[str, Path]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Union[pd.DataFrame, List[str], Pipeline]]:
    """
    Execute end-to-end preprocessing pipeline:
    1. Load raw data
    2. Clean column names
    3. Detect & drop constant features
    4. Stratified 80/20 train-test split
    5. Fit preprocessors exclusively on training data (Zero Data Leakage)
    6. Save processed datasets and artifacts

    Returns:
        Dictionary with processed data and artifacts.
    """
    logger.info("Executing end-to-end preprocessing pipeline...")
    df_raw = load_raw_data(raw_data_path)
    df_clean = clean_column_names(df_raw)

    target_col = "Bankrupt?"
    all_raw_cols = df_clean.columns.tolist()

    # Detect constant features
    constant_features = identify_constant_features(df_clean, target_col=target_col)
    
    # Track feature decisions
    feature_decisions = []
    for col in all_raw_cols:
        if col == target_col:
            continue
        if col in constant_features:
            feature_decisions.append({"feature": col, "status": "removed", "reason": "constant / zero variance"})
        else:
            feature_decisions.append({"feature": col, "status": "retained", "reason": "informative financial feature"})

    # Drop constant features
    df_filtered = df_clean.drop(columns=constant_features)
    retained_features = [c for c in df_filtered.columns if c != target_col]
    logger.info(f"Retained {len(retained_features)} features after removing {len(constant_features)} constant features.")

    # Stratified Split
    X_train, X_test, y_train, y_test = split_data(
        df_filtered, target_col=target_col, test_size=test_size, random_state=random_state
    )

    # Fit Logistic Regression Preprocessor (Impute + Scale) on X_train ONLY
    logistic_preprocessor = build_preprocessor(scale=True, imputation_strategy="median")
    logistic_preprocessor.fit(X_train)
    logger.info("Fitted Logistic preprocessor (median imputer + standard scaler) on X_train exclusively.")

    # Fit Tree Preprocessor (Impute ONLY, no scaling) on X_train ONLY
    tree_preprocessor = build_preprocessor(scale=False, imputation_strategy="median")
    tree_preprocessor.fit(X_train)
    logger.info("Fitted Tree preprocessor (median imputer) on X_train exclusively.")

    # Save processed train & test datasets
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df[target_col] = y_train.values
    test_df = X_test.copy()
    test_df[target_col] = y_test.values

    train_path = PROCESSED_DATA_DIR / "train.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info(f"Saved processed train dataset to: {train_path}")
    logger.info(f"Saved processed test dataset to: {test_path}")

    # Save feature decisions metadata
    decisions_df = pd.DataFrame(feature_decisions)
    decisions_path = OUTPUTS_DIR / "feature_preprocessing_metadata.csv"
    decisions_df.to_csv(decisions_path, index=False)
    logger.info(f"Saved feature preprocessing metadata to: {decisions_path}")

    # Save preprocessor artifact
    preprocessor_path = MODELS_DIR / "logistic_preprocessor.pkl"
    joblib.dump(logistic_preprocessor, preprocessor_path)
    logger.info(f"Saved preprocessor artifact to: {preprocessor_path}")

    tree_prep_path = MODELS_DIR / "tree_preprocessor.pkl"
    joblib.dump(tree_preprocessor, tree_prep_path)
    logger.info(f"Saved tree preprocessor artifact to: {tree_prep_path}")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "retained_features": retained_features,
        "removed_features": constant_features,
        "logistic_preprocessor": logistic_preprocessor,
        "tree_preprocessor": tree_preprocessor,
    }


if __name__ == "__main__":
    run_preprocessing_pipeline()
