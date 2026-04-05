"""Centralized configuration for the Telecom Churn Case Study."""
from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# Data file names
TELECOM_DATA_FILE = "telecom_churn_data.csv"
DATA_DICT_FILE = "Data+Dictionary-+Telecom+Churn+Case+Study.xlsx"

# Analysis thresholds
MISSING_VALUE_THRESHOLD = 0.70  # Drop columns with >70% missing
HIGH_VALUE_PERCENTILE = 70  # 70th percentile for high-value customer filter
CORRELATION_THRESHOLD = 0.85  # High correlation threshold
PCA_VARIANCE_THRESHOLD = 0.96  # 96% explained variance for PCA

# Column definitions
DATE_COLS = [
    "date_of_last_rech_6",
    "date_of_last_rech_7",
    "date_of_last_rech_8",
    "date_of_last_rech_9",
    "date_of_last_rech_data_6",
    "date_of_last_rech_data_7",
    "date_of_last_rech_data_8",
    "date_of_last_rech_data_9",
]
CUSTOMER_ID_COLS = ["mobile_number", "circle_id"]

# Month suffixes
MONTHS = ["6", "7", "8", "9"]
GOOD_MONTHS = ["6", "7"]
ACTION_MONTH = "8"
CHURN_MONTH = "9"

# Target column
TARGET_COL = "churn"

# Engineered feature columns
ENGINEERED_FEATURE_COLUMNS = [
    "total_data_rech_amt_6",
    "total_data_rech_amt_7",
    "total_amt_6",
    "total_amt_7",
    "average_amt_6_7",
    "total_call_min_8",
    "total_data_8",
]

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.25
N_PCA_COMPONENTS = 18
CLASS_WEIGHT = "balanced"

# Visualization settings
FIGURE_DPI = 100
CHURN_COLORS = {0: "#2196F3", 1: "#F44336"}
CHURN_LABELS = {0: "Non-Churn", 1: "Churn"}
