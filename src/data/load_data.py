"""
Data loading module for Small Business Insolvency Early-Warning System.
"""
from pathlib import Path
from typing import Optional, Union
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("load_data")

DEFAULT_RAW_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "data.csv"


def load_raw_data(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load raw Company Bankruptcy Prediction dataset.

    Args:
        file_path: Path to the raw CSV file. Defaults to data/raw/data.csv.

    Returns:
        pd.DataFrame containing the raw dataset without any modifications.

    Raises:
        FileNotFoundError: If the specified raw data file does not exist.
        ValueError: If the file is empty or cannot be parsed as a DataFrame.
    """
    path = Path(file_path) if file_path is not None else DEFAULT_RAW_DATA_PATH

    if not path.exists():
        msg = f"Raw data file not found at: {path}. Please place the dataset at this path."
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info(f"Loading raw data from: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.error(f"Failed to read CSV at {path}: {e}")
        raise ValueError(f"Failed to read CSV at {path}: {e}") from e

    if df.empty:
        msg = f"The dataset at {path} is empty."
        logger.error(msg)
        raise ValueError(msg)

    logger.info(f"Successfully loaded dataset with shape: {df.shape} ({df.shape[0]} rows, {df.shape[1]} columns)")
    return df
