"""Data loading and validation utilities for Telecom Churn Case Study."""
import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    CUSTOMER_ID_COLS,
    DATE_COLS,
    MISSING_VALUE_THRESHOLD,
    MONTHS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TELECOM_DATA_FILE,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def load_telecom_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load telecom churn dataset from CSV.

    Args:
        filepath: Optional path to data file. Defaults to RAW_DATA_DIR/TELECOM_DATA_FILE.

    Returns:
        DataFrame with raw telecom data.

    Raises:
        FileNotFoundError: If data file does not exist.
    """
    if filepath is None:
        filepath = RAW_DATA_DIR / TELECOM_DATA_FILE

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}. "
            "Please place the telecom churn data CSV in data/raw/."
        )

    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def validate_data(df: pd.DataFrame) -> bool:
    """Validate that a DataFrame has expected telecom churn schema.

    Args:
        df: Input DataFrame to validate.

    Returns:
        True if validation passes.

    Raises:
        ValueError: If critical columns are missing or data is empty.
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")

    # Check at least one mobile_number or circle_id column
    id_cols_present = [c for c in CUSTOMER_ID_COLS if c in df.columns]
    if not id_cols_present:
        raise ValueError(
            f"No customer ID columns found. Expected one of: {CUSTOMER_ID_COLS}"
        )

    # Check for monthly columns
    for month in MONTHS:
        month_cols = [c for c in df.columns if c.endswith(f"_{month}")]
        if not month_cols:
            raise ValueError(
                f"No columns found for month {month}. "
                "Expected columns ending with _{month}."
            )

    logger.info("Data validation passed.")
    return True


def get_column_types(df: pd.DataFrame) -> Tuple[list, list, list]:
    """Segregate columns by data type.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (date_columns, customer_id_columns, numeric_columns).
    """
    date_cols = [c for c in DATE_COLS if c in df.columns]
    customer_id_cols = [c for c in CUSTOMER_ID_COLS if c in df.columns]
    num_cols = [
        c
        for c in df.columns
        if c not in date_cols
        and c not in customer_id_cols
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return date_cols, customer_id_cols, num_cols


def handle_missing_values(
    df: pd.DataFrame,
    threshold: float = MISSING_VALUE_THRESHOLD,
) -> pd.DataFrame:
    """Handle missing values: drop high-missing columns, impute the rest.

    Args:
        df: Input DataFrame.
        threshold: Drop columns with missing rate above this threshold.

    Returns:
        DataFrame with missing values handled.
    """
    df = df.copy()

    # Drop columns with too many missing values
    missing_rate = df.isnull().mean()
    high_missing = missing_rate[missing_rate > threshold].index.tolist()
    if high_missing:
        logger.info(
            f"Dropping {len(high_missing)} columns with >{threshold*100:.0f}% missing: "
            f"{high_missing[:5]}{'...' if len(high_missing) > 5 else ''}"
        )
        df = df.drop(columns=high_missing)

    # Impute remaining missing values with 0
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing > 0:
        logger.info(f"Imputing {remaining_missing} remaining missing values with 0")
        df = df.fillna(0)

    return df


def drop_single_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with only one unique value (no predictive power).

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with constant columns removed.
    """
    df = df.copy()
    single_value_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if single_value_cols:
        logger.info(
            f"Dropping {len(single_value_cols)} constant columns: {single_value_cols}"
        )
        df = df.drop(columns=single_value_cols)
    return df


def drop_date_and_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop date and customer ID columns not needed for modeling.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame without date and ID columns.
    """
    df = df.copy()
    cols_to_drop = [c for c in (DATE_COLS + CUSTOMER_ID_COLS) if c in df.columns]
    if cols_to_drop:
        logger.info(f"Dropping {len(cols_to_drop)} date/ID columns")
        df = df.drop(columns=cols_to_drop)
    return df


def clean_telecom_data(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: handle missing values, drop unusable columns.

    Args:
        df: Raw telecom DataFrame.

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    df = handle_missing_values(df)
    df = drop_single_value_columns(df)
    df = drop_date_and_id_columns(df)
    logger.info(f"After cleaning: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def get_missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary DataFrame of missing values per column.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with columns: ['missing_count', 'missing_pct'] sorted descending.
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    summary = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_pct": missing_pct,
        }
    )
    return summary[summary["missing_count"] > 0].sort_values(
        "missing_pct", ascending=False
    )


def save_processed_data(
    df: pd.DataFrame, filename: str = "telecom_churn_processed.csv"
) -> Path:
    """Save processed DataFrame to the processed data directory.

    Args:
        df: Processed DataFrame.
        filename: Output filename.

    Returns:
        Path to saved file.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / filename
    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")
    return output_path
