"""Unit tests for data_loader module."""
import numpy as np
import pandas as pd
import pytest

from src.config import CUSTOMER_ID_COLS, DATE_COLS
from src.data_loader import (
    clean_telecom_data,
    drop_date_and_id_columns,
    drop_single_value_columns,
    get_column_types,
    get_missing_value_summary,
    handle_missing_values,
    validate_data,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def minimal_df():
    """Minimal valid telecom DataFrame with month columns."""
    return pd.DataFrame(
        {
            "mobile_number": [111, 222, 333],
            "circle_id": [1, 2, 3],
            "arpu_6": [200.0, 300.0, 150.0],
            "arpu_7": [210.0, 280.0, 160.0],
            "arpu_8": [180.0, 250.0, 90.0],
            "arpu_9": [0.0, 230.0, 0.0],
            "total_ic_mou_6": [100.0, 200.0, 80.0],
            "total_ic_mou_7": [110.0, 190.0, 85.0],
            "total_ic_mou_8": [90.0, 170.0, 40.0],
            "total_ic_mou_9": [0.0, 160.0, 0.0],
            "date_of_last_rech_6": ["2014-06-01", "2014-06-15", "2014-06-20"],
        }
    )


@pytest.fixture
def df_with_missing():
    """DataFrame with missing values for testing imputation."""
    n = 10
    df = pd.DataFrame(
        {
            "mobile_number": range(n),
            "col_high_missing": [np.nan] * 8 + [1.0, 2.0],  # 80% missing -> drop
            "col_low_missing": [np.nan, 1.0] + [2.0] * 8,  # 10% missing -> impute
            "arpu_6": np.random.uniform(100, 500, n),
            "arpu_7": np.random.uniform(100, 500, n),
            "arpu_8": np.random.uniform(100, 500, n),
            "arpu_9": np.random.uniform(100, 500, n),
        }
    )
    return df


@pytest.fixture
def df_with_constants():
    """DataFrame with constant columns."""
    return pd.DataFrame(
        {
            "mobile_number": [1, 2, 3],
            "arpu_6": [100.0, 200.0, 300.0],
            "arpu_7": [100.0, 200.0, 300.0],
            "arpu_8": [100.0, 200.0, 300.0],
            "arpu_9": [100.0, 200.0, 300.0],
            "constant_col": [1, 1, 1],  # only one unique value
            "zero_col": [0, 0, 0],  # only one unique value
        }
    )


# ── validate_data ────────────────────────────────────────────────────────────


class TestValidateData:
    def test_valid_dataframe(self, minimal_df):
        assert validate_data(minimal_df) is True

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_data(pd.DataFrame())

    def test_missing_id_column_raises(self):
        df = pd.DataFrame(
            {
                "arpu_6": [100.0],
                "arpu_7": [100.0],
                "arpu_8": [100.0],
                "arpu_9": [100.0],
            }
        )
        with pytest.raises(ValueError, match="customer ID"):
            validate_data(df)

    def test_missing_month_column_raises(self):
        df = pd.DataFrame({"mobile_number": [1, 2], "some_col": [1, 2]})
        with pytest.raises(ValueError, match="month"):
            validate_data(df)

    def test_valid_with_only_mobile_number(self):
        df = pd.DataFrame(
            {
                "mobile_number": [1, 2],
                "arpu_6": [100.0, 200.0],
                "arpu_7": [100.0, 200.0],
                "arpu_8": [100.0, 200.0],
                "arpu_9": [100.0, 200.0],
            }
        )
        assert validate_data(df) is True


# ── get_column_types ─────────────────────────────────────────────────────────


class TestGetColumnTypes:
    def test_returns_three_lists(self, minimal_df):
        date_cols, id_cols, num_cols = get_column_types(minimal_df)
        assert isinstance(date_cols, list)
        assert isinstance(id_cols, list)
        assert isinstance(num_cols, list)

    def test_date_columns_detected(self, minimal_df):
        date_cols, _, _ = get_column_types(minimal_df)
        assert "date_of_last_rech_6" in date_cols

    def test_id_columns_detected(self, minimal_df):
        _, id_cols, _ = get_column_types(minimal_df)
        assert "mobile_number" in id_cols
        assert "circle_id" in id_cols

    def test_numeric_columns_excluded_id_and_date(self, minimal_df):
        _, _, num_cols = get_column_types(minimal_df)
        assert "mobile_number" not in num_cols
        assert "date_of_last_rech_6" not in num_cols


# ── handle_missing_values ────────────────────────────────────────────────────


class TestHandleMissingValues:
    def test_drops_high_missing_columns(self, df_with_missing):
        result = handle_missing_values(df_with_missing, threshold=0.70)
        assert "col_high_missing" not in result.columns

    def test_keeps_low_missing_columns(self, df_with_missing):
        result = handle_missing_values(df_with_missing, threshold=0.70)
        assert "col_low_missing" in result.columns

    def test_imputes_remaining_with_zero(self, df_with_missing):
        result = handle_missing_values(df_with_missing, threshold=0.70)
        assert result.isnull().sum().sum() == 0

    def test_no_missing_data_unchanged(self, minimal_df):
        df_clean = minimal_df.copy()
        # Remove date col to avoid issues
        df_clean = df_clean.drop(columns=["date_of_last_rech_6"])
        result = handle_missing_values(df_clean)
        assert result.shape == df_clean.shape

    def test_custom_threshold(self, df_with_missing):
        # With threshold=0.05, col_low_missing (10% missing) should also be dropped
        result = handle_missing_values(df_with_missing, threshold=0.05)
        assert "col_low_missing" not in result.columns


# ── drop_single_value_columns ────────────────────────────────────────────────


class TestDropSingleValueColumns:
    def test_removes_constant_columns(self, df_with_constants):
        result = drop_single_value_columns(df_with_constants)
        assert "constant_col" not in result.columns
        assert "zero_col" not in result.columns

    def test_keeps_varied_columns(self, df_with_constants):
        result = drop_single_value_columns(df_with_constants)
        assert "arpu_6" in result.columns
        assert "mobile_number" in result.columns

    def test_no_constants_unchanged(self, minimal_df):
        df_clean = minimal_df.drop(columns=["date_of_last_rech_6"])
        result = drop_single_value_columns(df_clean)
        assert result.shape == df_clean.shape


# ── drop_date_and_id_columns ─────────────────────────────────────────────────


class TestDropDateAndIdColumns:
    def test_removes_date_columns(self, minimal_df):
        result = drop_date_and_id_columns(minimal_df)
        assert "date_of_last_rech_6" not in result.columns

    def test_removes_id_columns(self, minimal_df):
        result = drop_date_and_id_columns(minimal_df)
        assert "mobile_number" not in result.columns
        assert "circle_id" not in result.columns

    def test_keeps_numeric_columns(self, minimal_df):
        result = drop_date_and_id_columns(minimal_df)
        assert "arpu_6" in result.columns


# ── get_missing_value_summary ────────────────────────────────────────────────


class TestGetMissingValueSummary:
    def test_returns_dataframe(self, df_with_missing):
        summary = get_missing_value_summary(df_with_missing)
        assert isinstance(summary, pd.DataFrame)

    def test_only_missing_columns_included(self, df_with_missing):
        summary = get_missing_value_summary(df_with_missing)
        assert "mobile_number" not in summary.index  # no missing in this col

    def test_sorted_descending(self, df_with_missing):
        summary = get_missing_value_summary(df_with_missing)
        pcts = summary["missing_pct"].values
        assert all(pcts[i] >= pcts[i + 1] for i in range(len(pcts) - 1))

    def test_no_missing_returns_empty(self, minimal_df):
        df_clean = minimal_df.drop(columns=["date_of_last_rech_6"])
        df_clean = df_clean.fillna(0)
        summary = get_missing_value_summary(df_clean)
        assert len(summary) == 0


# ── clean_telecom_data ────────────────────────────────────────────────────────


class TestCleanTelecomData:
    def test_returns_dataframe(self, minimal_df):
        result = clean_telecom_data(minimal_df)
        assert isinstance(result, pd.DataFrame)

    def test_no_missing_values(self, df_with_missing):
        result = clean_telecom_data(df_with_missing)
        assert result.isnull().sum().sum() == 0

    def test_id_columns_removed(self, minimal_df):
        result = clean_telecom_data(minimal_df)
        assert "mobile_number" not in result.columns

    def test_date_columns_removed(self, minimal_df):
        result = clean_telecom_data(minimal_df)
        assert "date_of_last_rech_6" not in result.columns
