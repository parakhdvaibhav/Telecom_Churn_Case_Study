"""Statistical analysis and feature engineering for Telecom Churn Case Study."""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .config import (
    ACTION_MONTH,
    CHURN_MONTH,
    GOOD_MONTHS,
    HIGH_VALUE_PERCENTILE,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)


def calculate_churn_rate(df: pd.DataFrame, churn_col: str = TARGET_COL) -> float:
    """Calculate the churn rate as a percentage.

    Args:
        df: DataFrame containing churn column.
        churn_col: Name of the binary churn column.

    Returns:
        Churn rate as a float between 0 and 100.

    Raises:
        KeyError: If churn_col not in DataFrame.
    """
    if churn_col not in df.columns:
        raise KeyError(f"Column '{churn_col}' not found in DataFrame.")
    return df[churn_col].mean() * 100


def get_churn_distribution(
    df: pd.DataFrame, churn_col: str = TARGET_COL
) -> pd.Series:
    """Get the count and percentage distribution of churn vs non-churn.

    Args:
        df: DataFrame containing churn column.
        churn_col: Name of the binary churn column.

    Returns:
        Series with churn distribution counts.
    """
    if churn_col not in df.columns:
        raise KeyError(f"Column '{churn_col}' not found in DataFrame.")
    return df[churn_col].value_counts()


def filter_high_value_customers(
    df: pd.DataFrame,
    percentile: float = HIGH_VALUE_PERCENTILE,
) -> pd.DataFrame:
    """Filter to retain only high-value customers.

    High-value customers are defined as those whose average monthly recharge
    amount (months 6 and 7) is >= the 70th percentile.

    Args:
        df: Input DataFrame with recharge columns.
        percentile: Percentile threshold for high-value filtering.

    Returns:
        DataFrame filtered to high-value customers.
    """
    df = df.copy()

    # Calculate total data recharge amount for months 6 and 7
    for month in GOOD_MONTHS:
        rech_data_col = f"total_rech_data_{month}"
        av_rech_amt_col = f"av_rech_amt_data_{month}"
        if rech_data_col in df.columns and av_rech_amt_col in df.columns:
            df[f"total_data_rech_amt_{month}"] = (
                df[rech_data_col] * df[av_rech_amt_col]
            )
        else:
            df[f"total_data_rech_amt_{month}"] = 0

    # Calculate total amount recharge
    for month in GOOD_MONTHS:
        rech_amt_col = f"total_rech_amt_{month}"
        data_col = f"total_data_rech_amt_{month}"
        if rech_amt_col in df.columns:
            df[f"total_amt_{month}"] = df[rech_amt_col] + df.get(data_col, 0)
        else:
            df[f"total_amt_{month}"] = 0

    # Average of month 6 and 7
    df["average_amt_6_7"] = (df["total_amt_6"] + df["total_amt_7"]) / 2

    # Filter to >= percentile
    threshold = np.percentile(df["average_amt_6_7"], percentile)
    df_hv = df[df["average_amt_6_7"] >= threshold].copy()

    logger.info(
        f"High-value customers (>= {percentile}th percentile, "
        f"threshold={threshold:.2f}): {len(df_hv)} / {len(df)} "
        f"({len(df_hv)/len(df)*100:.1f}%)"
    )
    return df_hv


def tag_churners(df: pd.DataFrame) -> pd.DataFrame:
    """Tag churn customers based on business rules.

    A customer is tagged as churn (1) if both total call minutes and total
    data consumption in month 9 are 0.

    Args:
        df: DataFrame for high-value customers.

    Returns:
        DataFrame with 'churn' column added.
    """
    df = df.copy()

    # Total call minutes in month 9
    incoming_col = f"total_ic_mou_{CHURN_MONTH}"
    outgoing_col = f"total_og_mou_{CHURN_MONTH}"
    vol_2g_col = f"vol_2g_mb_{CHURN_MONTH}"
    vol_3g_col = f"vol_3g_mb_{CHURN_MONTH}"

    total_call_min_9 = 0
    if incoming_col in df.columns:
        total_call_min_9 = total_call_min_9 + df[incoming_col]
    if outgoing_col in df.columns:
        total_call_min_9 = total_call_min_9 + df[outgoing_col]

    total_data_9 = 0
    if vol_2g_col in df.columns:
        total_data_9 = total_data_9 + df[vol_2g_col]
    if vol_3g_col in df.columns:
        total_data_9 = total_data_9 + df[vol_3g_col]

    if isinstance(total_call_min_9, int):
        df[TARGET_COL] = 0
    else:
        df[TARGET_COL] = (
            (total_call_min_9 == 0) & (total_data_9 == 0)
        ).astype(int)

    churn_rate = calculate_churn_rate(df)
    logger.info(f"Churn rate after tagging: {churn_rate:.2f}%")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer key features for churn analysis.

    Creates:
    - total_call_min_8: Total call minutes in month 8
    - total_data_8: Total data consumption in month 8

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with engineered features added.
    """
    df = df.copy()

    # Total call minutes in action month (8)
    inc_col = f"total_ic_mou_{ACTION_MONTH}"
    og_col = f"total_og_mou_{ACTION_MONTH}"
    if inc_col in df.columns and og_col in df.columns:
        df["total_call_min_8"] = df[inc_col] + df[og_col]
    elif inc_col in df.columns:
        df["total_call_min_8"] = df[inc_col]
    elif og_col in df.columns:
        df["total_call_min_8"] = df[og_col]

    # Total data consumption in month 8
    v2g_col = f"vol_2g_mb_{ACTION_MONTH}"
    v3g_col = f"vol_3g_mb_{ACTION_MONTH}"
    if v2g_col in df.columns and v3g_col in df.columns:
        df["total_data_8"] = df[v2g_col] + df[v3g_col]
    elif v2g_col in df.columns:
        df["total_data_8"] = df[v2g_col]
    elif v3g_col in df.columns:
        df["total_data_8"] = df[v3g_col]

    return df


def compare_churn_groups(
    df: pd.DataFrame,
    feature: str,
    churn_col: str = TARGET_COL,
) -> Dict:
    """Compare a feature between churn and non-churn groups.

    Args:
        df: DataFrame containing feature and churn column.
        feature: Column name to analyze.
        churn_col: Name of the churn column.

    Returns:
        Dictionary with keys: 'churn_median', 'non_churn_median',
        'churn_mean', 'non_churn_mean', 'p_value', 'significant'.
    """
    if feature not in df.columns:
        raise KeyError(f"Feature '{feature}' not found in DataFrame.")
    if churn_col not in df.columns:
        raise KeyError(f"Churn column '{churn_col}' not found in DataFrame.")

    churn_vals = df[df[churn_col] == 1][feature].dropna()
    non_churn_vals = df[df[churn_col] == 0][feature].dropna()

    _, p_value = stats.mannwhitneyu(churn_vals, non_churn_vals, alternative="two-sided")

    return {
        "churn_median": churn_vals.median(),
        "non_churn_median": non_churn_vals.median(),
        "churn_mean": churn_vals.mean(),
        "non_churn_mean": non_churn_vals.mean(),
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def get_top_correlated_features(
    df: pd.DataFrame,
    target: str = TARGET_COL,
    n: int = 20,
) -> pd.Series:
    """Get the top N features most correlated with the target.

    Args:
        df: DataFrame with features and target.
        target: Target column name.
        n: Number of top features to return.

    Returns:
        Series of correlation values sorted by absolute value.
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)

    corr = df[numeric_cols + [target]].corr()[target].drop(target)
    return corr.reindex(corr.abs().sort_values(ascending=False).index).head(n)


def get_descriptive_stats(
    df: pd.DataFrame, group_col: str = TARGET_COL
) -> pd.DataFrame:
    """Get descriptive statistics grouped by churn status.

    Args:
        df: Input DataFrame.
        group_col: Column to group by.

    Returns:
        DataFrame with descriptive stats per group.
    """
    if group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' not found in DataFrame.")
    return df.groupby(group_col).describe().T


def train_test_split_stratified(
    df: pd.DataFrame,
    target: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets with stratification.

    Args:
        df: Input DataFrame.
        target: Target column name.
        test_size: Fraction of data for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    from sklearn.model_selection import train_test_split

    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame.")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples"
    )
    return X_train, X_test, y_train, y_test
