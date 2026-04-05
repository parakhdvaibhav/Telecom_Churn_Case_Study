"""Reusable plotting functions for Telecom Churn Case Study."""
import logging
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import CHURN_COLORS, CHURN_LABELS, FIGURE_DPI, TARGET_COL

logger = logging.getLogger(__name__)

# Set consistent style
sns.set_palette("tab10")
sns.set_style("whitegrid")


def plot_churn_distribution(
    df: pd.DataFrame,
    churn_col: str = TARGET_COL,
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot churn vs non-churn distribution as bar and pie charts.

    Args:
        df: DataFrame containing churn column.
        churn_col: Name of the churn column.
        figsize: Figure size tuple.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    if churn_col not in df.columns:
        raise KeyError(f"Column '{churn_col}' not found in DataFrame.")

    counts = df[churn_col].value_counts()
    labels = [CHURN_LABELS.get(i, str(i)) for i in counts.index]
    colors = [CHURN_COLORS.get(i, "#888888") for i in counts.index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=FIGURE_DPI)

    # Bar chart
    ax1.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=0.7)
    ax1.set_title("Churn Distribution (Count)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Customer Status")
    ax1.set_ylabel("Count")
    for i, (label, val) in enumerate(zip(labels, counts.values)):
        ax1.text(i, val + 50, f"{val:,}", ha="center", va="bottom", fontsize=10)

    # Pie chart
    ax2.pie(
        counts.values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
    )
    ax2.set_title("Churn Distribution (%)", fontsize=13, fontweight="bold")

    plt.suptitle("Customer Churn Analysis", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_feature_by_churn(
    df: pd.DataFrame,
    feature: str,
    churn_col: str = TARGET_COL,
    kind: str = "box",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a feature distribution segmented by churn status.

    Args:
        df: DataFrame containing feature and churn column.
        feature: Column name to plot.
        churn_col: Name of the churn column.
        kind: Plot type - 'box', 'violin', or 'hist'.
        figsize: Figure size tuple.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    if feature not in df.columns:
        raise KeyError(f"Feature '{feature}' not found in DataFrame.")
    if churn_col not in df.columns:
        raise KeyError(f"Column '{churn_col}' not found in DataFrame.")

    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)

    colors = list(CHURN_COLORS.values())

    if kind == "box":
        groups = [
            df[df[churn_col] == label][feature].dropna()
            for label in sorted(df[churn_col].unique())
        ]
        bp = ax.boxplot(groups, patch_artist=True, notch=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticklabels(
            [CHURN_LABELS.get(i, str(i)) for i in sorted(df[churn_col].unique())]
        )

    elif kind == "violin":
        data_list = []
        for label in sorted(df[churn_col].unique()):
            subset = df[df[churn_col] == label][[feature, churn_col]].copy()
            data_list.append(subset)
        plot_df = pd.concat(data_list)
        plot_df[churn_col] = plot_df[churn_col].map(CHURN_LABELS)
        sns.violinplot(
            data=plot_df,
            x=churn_col,
            y=feature,
            ax=ax,
            palette=list(CHURN_COLORS.values()),
        )

    elif kind == "hist":
        for label, color in CHURN_COLORS.items():
            subset = df[df[churn_col] == label][feature].dropna()
            ax.hist(
                subset,
                bins=30,
                alpha=0.6,
                color=color,
                label=CHURN_LABELS.get(label, str(label)),
            )
        ax.legend()

    ax.set_title(f"{feature} by Churn Status", fontsize=13, fontweight="bold")
    ax.set_xlabel("Churn Status" if kind in ("box", "violin") else feature)
    ax.set_ylabel(feature if kind in ("box", "violin") else "Frequency")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_median_comparison(
    df: pd.DataFrame,
    features: List[str],
    churn_col: str = TARGET_COL,
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot median values of multiple features for churn vs non-churn.

    Args:
        df: DataFrame containing features and churn column.
        features: List of feature column names.
        churn_col: Name of the churn column.
        figsize: Figure size tuple.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    if churn_col not in df.columns:
        raise KeyError(f"Column '{churn_col}' not found in DataFrame.")

    valid_features = [f for f in features if f in df.columns]
    if not valid_features:
        raise ValueError("No valid features found in DataFrame.")

    medians = df.groupby(churn_col)[valid_features].median()

    x = np.arange(len(valid_features))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    for i, label in enumerate(medians.index):
        ax.bar(
            x + i * width,
            medians.loc[label],
            width,
            label=CHURN_LABELS.get(label, str(label)),
            color=CHURN_COLORS.get(label, "#888888"),
            alpha=0.8,
        )

    ax.set_xlabel("Features")
    ax.set_ylabel("Median Value")
    ax.set_title(
        "Median Feature Values: Churn vs Non-Churn", fontsize=13, fontweight="bold"
    )
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(valid_features, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    n_features: int = 20,
    churn_col: str = TARGET_COL,
    figsize: tuple = (12, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot correlation heatmap for top N features.

    Args:
        df: DataFrame with numeric features.
        n_features: Number of top correlated features to include.
        churn_col: Target column to prioritize in correlation.
        figsize: Figure size tuple.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    numeric_df = df.select_dtypes(include=[np.number])

    if churn_col in numeric_df.columns:
        corr_with_target = numeric_df.corr()[churn_col].abs().sort_values(ascending=False)
        top_features = corr_with_target.head(n_features + 1).index.tolist()
    else:
        top_features = numeric_df.columns.tolist()[:n_features]

    corr_matrix = numeric_df[top_features].corr()

    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(
        f"Correlation Heatmap (Top {n_features} Features)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: List[float],
    top_n: int = 20,
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot feature importance bar chart.

    Args:
        feature_names: List of feature names.
        importances: Corresponding importance values.
        top_n: Number of top features to display.
        figsize: Figure size tuple.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    if len(feature_names) != len(importances):
        raise ValueError("feature_names and importances must have the same length.")

    fi_series = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    top_features = fi_series.tail(top_n)

    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_features)))
    top_features.plot(kind="barh", ax=ax, color=colors)

    ax.set_title(f"Top {top_n} Feature Importances", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    model_name: str = "Model",
    figsize: tuple = (8, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot ROC curve for a classification model.

    Args:
        fpr: False positive rates.
        tpr: True positive rates.
        auc_score: Area under ROC curve.
        model_name: Name of the model for the legend.
        figsize: Figure size tuple.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)

    ax.plot(
        fpr,
        tpr,
        color="#F44336",
        lw=2,
        label=f"{model_name} (AUC = {auc_score:.3f})",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        color="#888888",
        linestyle="--",
        lw=1,
        label="Random Classifier",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    return fig
