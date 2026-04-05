"""Convenience re-exports and EDA helper utilities for Telecom Churn Case Study."""
from .analysis import (
    calculate_churn_rate,
    compare_churn_groups,
    engineer_features,
    filter_high_value_customers,
    get_churn_distribution,
    get_descriptive_stats,
    get_top_correlated_features,
    tag_churners,
    train_test_split_stratified,
)
from .data_loader import (
    clean_telecom_data,
    drop_date_and_id_columns,
    drop_single_value_columns,
    get_column_types,
    get_missing_value_summary,
    handle_missing_values,
    load_telecom_data,
    save_processed_data,
    validate_data,
)
from .visualizations import (
    plot_churn_distribution,
    plot_correlation_heatmap,
    plot_feature_by_churn,
    plot_feature_importance,
    plot_median_comparison,
    plot_roc_curve,
)

__all__ = [
    # data_loader
    "load_telecom_data",
    "validate_data",
    "get_column_types",
    "handle_missing_values",
    "drop_single_value_columns",
    "drop_date_and_id_columns",
    "clean_telecom_data",
    "get_missing_value_summary",
    "save_processed_data",
    # analysis
    "calculate_churn_rate",
    "get_churn_distribution",
    "filter_high_value_customers",
    "tag_churners",
    "engineer_features",
    "compare_churn_groups",
    "get_top_correlated_features",
    "get_descriptive_stats",
    "train_test_split_stratified",
    # visualizations
    "plot_churn_distribution",
    "plot_feature_by_churn",
    "plot_median_comparison",
    "plot_correlation_heatmap",
    "plot_feature_importance",
    "plot_roc_curve",
]
