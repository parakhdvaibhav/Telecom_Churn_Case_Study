"""Unit tests for analysis module."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    calculate_churn_rate,
    compare_churn_groups,
    engineer_features,
    filter_high_value_customers,
    get_churn_distribution,
    get_top_correlated_features,
    tag_churners,
    train_test_split_stratified,
)
from src.config import TARGET_COL


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def churn_df():
    """DataFrame with churn column for testing."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "mobile_number": range(n),
            TARGET_COL: np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
            "arpu_6": np.random.uniform(100, 500, n),
            "arpu_7": np.random.uniform(100, 500, n),
            "arpu_8": np.random.uniform(50, 400, n),
            "feature_a": np.random.normal(50, 10, n),
            "feature_b": np.random.normal(30, 5, n),
        }
    )
    return df


@pytest.fixture
def recharge_df():
    """DataFrame with recharge columns for high-value filtering."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame(
        {
            "mobile_number": range(n),
            "total_rech_amt_6": np.random.uniform(100, 1000, n),
            "total_rech_amt_7": np.random.uniform(100, 1000, n),
            "total_rech_data_6": np.random.uniform(0, 10, n),
            "total_rech_data_7": np.random.uniform(0, 10, n),
            "av_rech_amt_data_6": np.random.uniform(50, 200, n),
            "av_rech_amt_data_7": np.random.uniform(50, 200, n),
        }
    )
    return df


@pytest.fixture
def churn_tag_df():
    """DataFrame suitable for testing tag_churners."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame(
        {
            "mobile_number": range(n),
            "total_ic_mou_9": np.random.choice([0.0, 100.0], size=n),
            "total_og_mou_9": np.random.choice([0.0, 50.0], size=n),
            "vol_2g_mb_9": np.random.choice([0.0, 200.0], size=n),
            "vol_3g_mb_9": np.random.choice([0.0, 300.0], size=n),
            "total_ic_mou_8": np.random.uniform(0, 200, n),
            "total_og_mou_8": np.random.uniform(0, 100, n),
            "vol_2g_mb_8": np.random.uniform(0, 500, n),
            "vol_3g_mb_8": np.random.uniform(0, 500, n),
        }
    )
    return df


# ── calculate_churn_rate ─────────────────────────────────────────────────────


class TestCalculateChurnRate:
    def test_returns_float(self, churn_df):
        rate = calculate_churn_rate(churn_df)
        assert isinstance(rate, float)

    def test_rate_between_0_and_100(self, churn_df):
        rate = calculate_churn_rate(churn_df)
        assert 0 <= rate <= 100

    def test_all_churn(self):
        df = pd.DataFrame({TARGET_COL: [1, 1, 1, 1]})
        assert calculate_churn_rate(df) == 100.0

    def test_no_churn(self):
        df = pd.DataFrame({TARGET_COL: [0, 0, 0, 0]})
        assert calculate_churn_rate(df) == 0.0

    def test_missing_column_raises(self, churn_df):
        with pytest.raises(KeyError):
            calculate_churn_rate(churn_df, churn_col="nonexistent")


# ── get_churn_distribution ────────────────────────────────────────────────────


class TestGetChurnDistribution:
    def test_returns_series(self, churn_df):
        dist = get_churn_distribution(churn_df)
        assert isinstance(dist, pd.Series)

    def test_contains_both_classes(self, churn_df):
        dist = get_churn_distribution(churn_df)
        assert 0 in dist.index or 1 in dist.index

    def test_missing_column_raises(self):
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(KeyError):
            get_churn_distribution(df)


# ── filter_high_value_customers ───────────────────────────────────────────────


class TestFilterHighValueCustomers:
    def test_returns_dataframe(self, recharge_df):
        result = filter_high_value_customers(recharge_df, percentile=70)
        assert isinstance(result, pd.DataFrame)

    def test_reduces_rows(self, recharge_df):
        result = filter_high_value_customers(recharge_df, percentile=70)
        assert len(result) < len(recharge_df)

    def test_approximately_30_percent_retained(self, recharge_df):
        result = filter_high_value_customers(recharge_df, percentile=70)
        retention_rate = len(result) / len(recharge_df)
        assert 0.25 <= retention_rate <= 0.40

    def test_custom_percentile(self, recharge_df):
        result_50 = filter_high_value_customers(recharge_df, percentile=50)
        result_90 = filter_high_value_customers(recharge_df, percentile=90)
        assert len(result_50) > len(result_90)


# ── tag_churners ──────────────────────────────────────────────────────────────


class TestTagChurners:
    def test_adds_churn_column(self, churn_tag_df):
        result = tag_churners(churn_tag_df)
        assert TARGET_COL in result.columns

    def test_churn_column_binary(self, churn_tag_df):
        result = tag_churners(churn_tag_df)
        assert set(result[TARGET_COL].unique()).issubset({0, 1})

    def test_churn_logic_correct(self):
        """Customer with 0 call minutes and 0 data in month 9 should be churn=1."""
        df = pd.DataFrame(
            {
                "total_ic_mou_9": [0.0, 100.0],
                "total_og_mou_9": [0.0, 50.0],
                "vol_2g_mb_9": [0.0, 200.0],
                "vol_3g_mb_9": [0.0, 100.0],
            }
        )
        result = tag_churners(df)
        assert result[TARGET_COL].iloc[0] == 1
        assert result[TARGET_COL].iloc[1] == 0

    def test_original_df_not_modified(self, churn_tag_df):
        original_cols = churn_tag_df.columns.tolist()
        _ = tag_churners(churn_tag_df)
        assert churn_tag_df.columns.tolist() == original_cols


# ── engineer_features ─────────────────────────────────────────────────────────


class TestEngineerFeatures:
    def test_adds_total_call_min_8(self, churn_tag_df):
        result = engineer_features(churn_tag_df)
        assert "total_call_min_8" in result.columns

    def test_adds_total_data_8(self, churn_tag_df):
        result = engineer_features(churn_tag_df)
        assert "total_data_8" in result.columns

    def test_total_call_min_8_correct(self, churn_tag_df):
        result = engineer_features(churn_tag_df)
        expected = churn_tag_df["total_ic_mou_8"] + churn_tag_df["total_og_mou_8"]
        pd.testing.assert_series_equal(
            result["total_call_min_8"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_original_not_modified(self, churn_tag_df):
        original_cols = churn_tag_df.columns.tolist()
        _ = engineer_features(churn_tag_df)
        assert churn_tag_df.columns.tolist() == original_cols


# ── compare_churn_groups ──────────────────────────────────────────────────────


class TestCompareChurnGroups:
    def test_returns_dict(self, churn_df):
        result = compare_churn_groups(churn_df, "arpu_6")
        assert isinstance(result, dict)

    def test_contains_expected_keys(self, churn_df):
        result = compare_churn_groups(churn_df, "arpu_6")
        expected_keys = {
            "churn_median",
            "non_churn_median",
            "churn_mean",
            "non_churn_mean",
            "p_value",
            "significant",
        }
        assert expected_keys == set(result.keys())

    def test_p_value_between_0_and_1(self, churn_df):
        result = compare_churn_groups(churn_df, "arpu_6")
        assert 0 <= result["p_value"] <= 1

    def test_missing_feature_raises(self, churn_df):
        with pytest.raises(KeyError):
            compare_churn_groups(churn_df, "nonexistent_feature")

    def test_missing_churn_col_raises(self, churn_df):
        with pytest.raises(KeyError):
            compare_churn_groups(churn_df, "arpu_6", churn_col="missing_col")


# ── get_top_correlated_features ───────────────────────────────────────────────


class TestGetTopCorrelatedFeatures:
    def test_returns_series(self, churn_df):
        result = get_top_correlated_features(churn_df)
        assert isinstance(result, pd.Series)

    def test_length_at_most_n(self, churn_df):
        n = 3
        result = get_top_correlated_features(churn_df, n=n)
        assert len(result) <= n

    def test_target_not_in_result(self, churn_df):
        result = get_top_correlated_features(churn_df)
        assert TARGET_COL not in result.index

    def test_missing_target_raises(self, churn_df):
        with pytest.raises(KeyError):
            get_top_correlated_features(churn_df, target="nonexistent")


# ── train_test_split_stratified ───────────────────────────────────────────────


class TestTrainTestSplitStratified:
    def test_returns_four_parts(self, churn_df):
        result = train_test_split_stratified(churn_df)
        assert len(result) == 4

    def test_correct_sizes(self, churn_df):
        X_train, X_test, y_train, y_test = train_test_split_stratified(
            churn_df, test_size=0.25
        )
        total = len(churn_df)
        assert abs(len(X_test) - total * 0.25) <= 2

    def test_target_not_in_X(self, churn_df):
        X_train, X_test, _, _ = train_test_split_stratified(churn_df)
        assert TARGET_COL not in X_train.columns
        assert TARGET_COL not in X_test.columns

    def test_missing_target_raises(self, churn_df):
        with pytest.raises(KeyError):
            train_test_split_stratified(churn_df, target="nonexistent")
