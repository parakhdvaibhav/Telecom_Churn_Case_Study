"""Unit tests for visualizations module."""
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.config import TARGET_COL
from src.visualizations import (
    plot_churn_distribution,
    plot_correlation_heatmap,
    plot_feature_by_churn,
    plot_feature_importance,
    plot_median_comparison,
    plot_roc_curve,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def sample_df():
    """Sample DataFrame for visualization tests."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            TARGET_COL: np.random.choice([0, 1], size=n, p=[0.85, 0.15]),
            "arpu_6": np.random.uniform(100, 500, n),
            "arpu_7": np.random.uniform(100, 500, n),
            "arpu_8": np.random.uniform(50, 450, n),
            "total_ic_mou_8": np.random.uniform(0, 300, n),
            "vol_2g_mb_8": np.random.uniform(0, 1000, n),
        }
    )


# ── plot_churn_distribution ───────────────────────────────────────────────────


class TestPlotChurnDistribution:
    def test_returns_figure(self, sample_df):
        fig = plot_churn_distribution(sample_df)
        assert isinstance(fig, plt.Figure)

    def test_figure_has_two_axes(self, sample_df):
        fig = plot_churn_distribution(sample_df)
        assert len(fig.get_axes()) == 2

    def test_missing_column_raises(self, sample_df):
        with pytest.raises(KeyError):
            plot_churn_distribution(sample_df, churn_col="nonexistent")

    def test_saves_to_file(self, sample_df, tmp_path):
        save_path = str(tmp_path / "churn_dist.png")
        fig = plot_churn_distribution(sample_df, save_path=save_path)
        import os

        assert os.path.exists(save_path)


# ── plot_feature_by_churn ─────────────────────────────────────────────────────


class TestPlotFeatureByChurn:
    def test_returns_figure_box(self, sample_df):
        fig = plot_feature_by_churn(sample_df, "arpu_6", kind="box")
        assert isinstance(fig, plt.Figure)

    def test_returns_figure_hist(self, sample_df):
        fig = plot_feature_by_churn(sample_df, "arpu_6", kind="hist")
        assert isinstance(fig, plt.Figure)

    def test_returns_figure_violin(self, sample_df):
        fig = plot_feature_by_churn(sample_df, "arpu_6", kind="violin")
        assert isinstance(fig, plt.Figure)

    def test_missing_feature_raises(self, sample_df):
        with pytest.raises(KeyError):
            plot_feature_by_churn(sample_df, "nonexistent_feature")

    def test_missing_churn_col_raises(self, sample_df):
        with pytest.raises(KeyError):
            plot_feature_by_churn(sample_df, "arpu_6", churn_col="nonexistent")


# ── plot_median_comparison ────────────────────────────────────────────────────


class TestPlotMedianComparison:
    def test_returns_figure(self, sample_df):
        features = ["arpu_6", "arpu_7", "arpu_8"]
        fig = plot_median_comparison(sample_df, features)
        assert isinstance(fig, plt.Figure)

    def test_missing_churn_col_raises(self, sample_df):
        with pytest.raises(KeyError):
            plot_median_comparison(sample_df, ["arpu_6"], churn_col="missing")

    def test_no_valid_features_raises(self, sample_df):
        with pytest.raises(ValueError):
            plot_median_comparison(sample_df, ["nonexistent_1", "nonexistent_2"])

    def test_partial_features_ok(self, sample_df):
        """Should work even if some features don't exist in df."""
        fig = plot_median_comparison(sample_df, ["arpu_6", "nonexistent"])
        assert isinstance(fig, plt.Figure)


# ── plot_correlation_heatmap ──────────────────────────────────────────────────


class TestPlotCorrelationHeatmap:
    def test_returns_figure(self, sample_df):
        fig = plot_correlation_heatmap(sample_df)
        assert isinstance(fig, plt.Figure)

    def test_custom_n_features(self, sample_df):
        fig = plot_correlation_heatmap(sample_df, n_features=3)
        assert isinstance(fig, plt.Figure)

    def test_works_without_churn_col(self, sample_df):
        df_no_churn = sample_df.drop(columns=[TARGET_COL])
        fig = plot_correlation_heatmap(df_no_churn)
        assert isinstance(fig, plt.Figure)


# ── plot_feature_importance ───────────────────────────────────────────────────


class TestPlotFeatureImportance:
    def test_returns_figure(self):
        names = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]
        importances = [0.3, 0.25, 0.2, 0.15, 0.1]
        fig = plot_feature_importance(names, importances)
        assert isinstance(fig, plt.Figure)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            plot_feature_importance(["a", "b"], [0.5])

    def test_top_n_respected(self):
        names = [f"feat_{i}" for i in range(10)]
        importances = [i / 10.0 for i in range(10)]
        fig = plot_feature_importance(names, importances, top_n=5)
        assert isinstance(fig, plt.Figure)

    def test_saves_to_file(self, tmp_path):
        save_path = str(tmp_path / "feat_imp.png")
        names = ["a", "b", "c"]
        importances = [0.5, 0.3, 0.2]
        plot_feature_importance(names, importances, save_path=save_path)
        import os

        assert os.path.exists(save_path)


# ── plot_roc_curve ────────────────────────────────────────────────────────────


class TestPlotRocCurve:
    def test_returns_figure(self):
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr)
        fig = plot_roc_curve(fpr, tpr, auc_score=0.87)
        assert isinstance(fig, plt.Figure)

    def test_custom_model_name(self):
        fpr = np.array([0.0, 0.2, 0.5, 1.0])
        tpr = np.array([0.0, 0.6, 0.8, 1.0])
        fig = plot_roc_curve(fpr, tpr, auc_score=0.75, model_name="Random Forest")
        assert isinstance(fig, plt.Figure)

    def test_saves_to_file(self, tmp_path):
        save_path = str(tmp_path / "roc.png")
        fpr = np.array([0.0, 0.5, 1.0])
        tpr = np.array([0.0, 0.8, 1.0])
        plot_roc_curve(fpr, tpr, auc_score=0.80, save_path=save_path)
        import os

        assert os.path.exists(save_path)
