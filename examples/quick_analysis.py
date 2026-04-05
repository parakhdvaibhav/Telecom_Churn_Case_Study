"""
Quick Analysis Demo - Telecom Churn Case Study
==============================================
Demonstrates how to use the src/ utility modules for telecom churn analysis.

Usage:
    python examples/quick_analysis.py

Note: Place telecom_churn_data.csv in data/raw/ before running.
"""
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis import (
    calculate_churn_rate,
    compare_churn_groups,
    engineer_features,
    filter_high_value_customers,
    tag_churners,
)
from src.config import (
    HIGH_VALUE_PERCENTILE,
    MISSING_VALUE_THRESHOLD,
    REPORTS_DIR,
    TARGET_COL,
)
from src.data_loader import (
    drop_date_and_id_columns,
    drop_single_value_columns,
    get_column_types,
    get_missing_value_summary,
    handle_missing_values,
)
from src.visualizations import (
    plot_churn_distribution,
    plot_feature_by_churn,
    plot_median_comparison,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_demo_data(n: int = 500) -> pd.DataFrame:
    """Create a synthetic demo dataset for demonstration purposes."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "mobile_number": range(n),
            "circle_id": np.random.randint(1, 10, n),
            # Month 6 (good phase)
            "arpu_6": np.random.uniform(200, 600, n),
            "total_rech_amt_6": np.random.uniform(200, 800, n),
            "total_rech_data_6": np.random.uniform(1, 20, n),
            "av_rech_amt_data_6": np.random.uniform(50, 200, n),
            "total_ic_mou_6": np.random.uniform(100, 500, n),
            "total_og_mou_6": np.random.uniform(80, 400, n),
            "vol_2g_mb_6": np.random.uniform(100, 2000, n),
            "vol_3g_mb_6": np.random.uniform(0, 3000, n),
            # Month 7 (good phase)
            "arpu_7": np.random.uniform(200, 600, n),
            "total_rech_amt_7": np.random.uniform(200, 800, n),
            "total_rech_data_7": np.random.uniform(1, 20, n),
            "av_rech_amt_data_7": np.random.uniform(50, 200, n),
            "total_ic_mou_7": np.random.uniform(100, 500, n),
            "total_og_mou_7": np.random.uniform(80, 400, n),
            "vol_2g_mb_7": np.random.uniform(100, 2000, n),
            "vol_3g_mb_7": np.random.uniform(0, 3000, n),
            # Month 8 (action phase)
            "arpu_8": np.random.uniform(50, 550, n),
            "total_ic_mou_8": np.random.uniform(0, 450, n),
            "total_og_mou_8": np.random.uniform(0, 350, n),
            "vol_2g_mb_8": np.random.uniform(0, 1800, n),
            "vol_3g_mb_8": np.random.uniform(0, 2500, n),
            # Month 9 (churn phase) - some customers go silent
            "arpu_9": np.concatenate(
                [
                    np.zeros(int(n * 0.08)),
                    np.random.uniform(50, 500, n - int(n * 0.08)),
                ]
            ),
            "total_ic_mou_9": np.concatenate(
                [
                    np.zeros(int(n * 0.08)),
                    np.random.uniform(10, 400, n - int(n * 0.08)),
                ]
            ),
            "total_og_mou_9": np.concatenate(
                [
                    np.zeros(int(n * 0.08)),
                    np.random.uniform(5, 350, n - int(n * 0.08)),
                ]
            ),
            "vol_2g_mb_9": np.concatenate(
                [
                    np.zeros(int(n * 0.08)),
                    np.random.uniform(0, 1500, n - int(n * 0.08)),
                ]
            ),
            "vol_3g_mb_9": np.concatenate(
                [
                    np.zeros(int(n * 0.08)),
                    np.random.uniform(0, 2000, n - int(n * 0.08)),
                ]
            ),
        }
    )
    return df


def main():
    logger.info("=" * 60)
    logger.info("TELECOM CHURN CASE STUDY - QUICK ANALYSIS DEMO")
    logger.info("=" * 60)

    # 1. Create or load demo data
    logger.info("\n[1] Creating synthetic demo data...")
    df = create_demo_data(n=500)
    logger.info(f"    Dataset shape: {df.shape}")

    # 2. Basic exploration
    logger.info("\n[2] Basic data exploration...")
    date_cols, id_cols, num_cols = get_column_types(df)
    logger.info(f"    Numeric columns: {len(num_cols)}")
    logger.info(f"    ID columns: {id_cols}")

    # 3. Missing value summary
    missing_summary = get_missing_value_summary(df)
    if len(missing_summary) > 0:
        logger.info(f"\n[3] Missing values found in {len(missing_summary)} columns")
    else:
        logger.info("\n[3] No missing values found")

    # 4. Handle missing values
    df_clean = handle_missing_values(df, threshold=MISSING_VALUE_THRESHOLD)
    df_clean = drop_single_value_columns(df_clean)

    # 5. Filter high-value customers
    logger.info("\n[4] Filtering high-value customers...")
    df_hv = filter_high_value_customers(df_clean, percentile=HIGH_VALUE_PERCENTILE)

    # 6. Tag churners
    logger.info("\n[5] Tagging churners...")
    df_tagged = tag_churners(df_hv)
    churn_rate = calculate_churn_rate(df_tagged)
    logger.info(f"    Churn rate: {churn_rate:.2f}%")

    # 7. Engineer features
    df_features = engineer_features(df_tagged)

    # 8. Drop date/ID columns
    df_model = drop_date_and_id_columns(df_features)  # noqa: F841

    # 9. Comparative analysis
    logger.info("\n[6] Comparative analysis (churn vs non-churn)...")
    for feature in ["arpu_6", "arpu_7", "arpu_8"]:
        if feature in df_tagged.columns:
            stats = compare_churn_groups(df_tagged, feature)
            logger.info(
                f"    {feature}: churn_median={stats['churn_median']:.1f}, "
                f"non_churn_median={stats['non_churn_median']:.1f}, "
                f"significant={stats['significant']}"
            )

    # 10. Generate plots
    logger.info("\n[7] Generating visualizations...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Churn distribution
    fig1 = plot_churn_distribution(
        df_tagged,
        save_path=str(REPORTS_DIR / "churn_distribution.png"),
    )
    plt.close(fig1)
    logger.info("    Saved: reports/churn_distribution.png")

    # Median comparison
    arpu_features = [c for c in ["arpu_6", "arpu_7", "arpu_8"] if c in df_tagged.columns]
    if arpu_features:
        fig2 = plot_median_comparison(
            df_tagged,
            arpu_features,
            save_path=str(REPORTS_DIR / "arpu_comparison.png"),
        )
        plt.close(fig2)
        logger.info("    Saved: reports/arpu_comparison.png")

    logger.info("\n" + "=" * 60)
    logger.info("Demo complete! Reports saved to reports/ directory.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
