"""
Unit tests for data loading and basic data validation (Segment 1).
"""
from pathlib import Path
import pytest
import pandas as pd

from src.data.load_data import load_raw_data, DEFAULT_RAW_DATA_PATH


def test_raw_data_file_exists():
    """Test that the raw dataset file exists at data/raw/data.csv."""
    assert DEFAULT_RAW_DATA_PATH.exists(), f"Raw data file does not exist at {DEFAULT_RAW_DATA_PATH}"
    assert DEFAULT_RAW_DATA_PATH.is_file(), f"Path {DEFAULT_RAW_DATA_PATH} is not a file."


def test_load_raw_data_returns_dataframe():
    """Test that load_raw_data returns a non-empty pandas DataFrame."""
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame), "load_raw_data() did not return a pandas DataFrame"
    assert not df.empty, "Loaded DataFrame is unexpectedly empty"


def test_dataset_dimensions():
    """Test that the dataset has expected rows and columns."""
    df = load_raw_data()
    n_rows, n_cols = df.shape
    assert n_rows > 6000, f"Expected >6000 rows, got {n_rows}"
    assert n_cols > 90, f"Expected >90 columns, got {n_cols}"


def test_target_column_exists():
    """Test that the target column 'Bankrupt?' is present."""
    df = load_raw_data()
    cleaned_cols = [c.strip() for c in df.columns]
    assert "Bankrupt?" in cleaned_cols, "Target column 'Bankrupt?' not found in dataset columns."


def test_load_raw_data_invalid_path_raises_error():
    """Test that loading from a non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_raw_data("non_existent_directory/non_existent_file.csv")
