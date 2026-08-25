"""
Script to run Segment 1 dataset inspection and generate data quality artifacts.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_raw_data
from src.utils.logger import setup_logger

logger = setup_logger("segment_1_inspection")


def inspect_dataset():
    logger.info("Starting Segment 1 Dataset Inspection...")
    df = load_raw_data()

    num_rows, num_cols = df.shape
    logger.info(f"Dataset Shape: {num_rows} rows, {num_cols} columns")

    # Clean column names (strip leading/trailing whitespace if any)
    df.columns = [c.strip() for c in df.columns]

    target_col = "Bankrupt?"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {df.columns.tolist()[:10]}...")

    features = [c for c in df.columns if c != target_col]
    logger.info(f"Target Column: '{target_col}', Total Features: {len(features)}")

    # Target Distribution
    target_counts = df[target_col].value_counts()
    target_props = df[target_col].value_counts(normalize=True) * 100
    logger.info(f"Target counts:\n{target_counts}")
    logger.info(f"Target distribution (%):\n{target_props}")

    # Duplicates & Missing
    dup_count = df.duplicated().sum()
    dup_pct = (dup_count / num_rows) * 100
    missing_total = df.isnull().sum().sum()
    logger.info(f"Duplicate Rows: {dup_count} ({dup_pct:.2f}%)")
    logger.info(f"Total Missing Values: {missing_total}")

    # Feature Metadata & Quality Report
    report_rows = []
    metadata_rows = []

    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        unique_cnt = series.nunique()
        missing_cnt = series.isnull().sum()
        missing_pct = (missing_cnt / num_rows) * 100
        is_constant = bool(unique_cnt <= 1)
        
        is_num = np.issubdtype(series.dtype, np.number)
        mean_val = float(series.mean()) if is_num else np.nan
        std_val = float(series.std()) if is_num else np.nan
        min_val = float(series.min()) if is_num else np.nan
        max_val = float(series.max()) if is_num else np.nan
        skew_val = float(series.skew()) if is_num else np.nan

        report_rows.append({
            "feature": col,
            "dtype": dtype,
            "missing_count": missing_cnt,
            "missing_percentage": missing_pct,
            "unique_count": unique_cnt,
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "skewness": skew_val,
            "constant_feature": is_constant,
        })

        if col != target_col and is_num:
            metadata_rows.append({
                "feature_name": col,
                "datatype": dtype,
                "unique_values": unique_cnt,
                "missing_count": missing_cnt,
                "missing_percentage": missing_pct,
                "mean": mean_val,
                "std": std_val,
                "min": min_val,
                "max": max_val,
            })

    report_df = pd.DataFrame(report_rows)
    metadata_df = pd.DataFrame(metadata_rows)

    out_dir = PROJECT_ROOT / "outputs" / "data_quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "data_quality_report.csv"
    metadata_path = out_dir / "feature_metadata.csv"

    report_df.to_csv(report_path, index=False)
    metadata_df.to_csv(metadata_path, index=False)

    logger.info(f"Saved Data Quality Report to: {report_path}")
    logger.info(f"Saved Feature Metadata to: {metadata_path}")

    constant_features = report_df[report_df["constant_feature"]]["feature"].tolist()
    logger.info(f"Constant/Zero-Variance Features found: {constant_features}")

    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "target_counts": target_counts.to_dict(),
        "target_props": target_props.to_dict(),
        "duplicates": dup_count,
        "missing_total": missing_total,
        "constant_features": constant_features,
    }


if __name__ == "__main__":
    inspect_dataset()
