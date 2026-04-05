# API Reference — Telecom Churn Case Study (`src` package)

## Module: `src.config`

Centralized configuration constants. Import directly:

```python
from src.config import RANDOM_STATE, TARGET_COL, HIGH_VALUE_PERCENTILE
```

| Constant | Type | Default | Description |
|----------|------|---------|-------------|
| `ROOT_DIR` | `Path` | — | Project root directory |
| `RAW_DATA_DIR` | `Path` | `data/raw/` | Raw data directory |
| `PROCESSED_DATA_DIR` | `Path` | `data/processed/` | Processed data directory |
| `REPORTS_DIR` | `Path` | `reports/` | Reports output directory |
| `TELECOM_DATA_FILE` | `str` | `telecom_churn_data.csv` | Default data filename |
| `MISSING_VALUE_THRESHOLD` | `float` | `0.70` | Drop threshold for missing values |
| `HIGH_VALUE_PERCENTILE` | `int` | `70` | Percentile for high-value filter |
| `CORRELATION_THRESHOLD` | `float` | `0.85` | High correlation threshold |
| `PCA_VARIANCE_THRESHOLD` | `float` | `0.96` | PCA variance target |
| `MONTHS` | `list` | `["6","7","8","9"]` | Month suffixes |
| `GOOD_MONTHS` | `list` | `["6","7"]` | Baseline months |
| `ACTION_MONTH` | `str` | `"8"` | Action month suffix |
| `CHURN_MONTH` | `str` | `"9"` | Churn labeling month suffix |
| `TARGET_COL` | `str` | `"churn"` | Target column name |
| `RANDOM_STATE` | `int` | `42` | Random seed |
| `TEST_SIZE` | `float` | `0.25` | Test split fraction |
| `N_PCA_COMPONENTS` | `int` | `18` | PCA components |

---

## Module: `src.data_loader`

### `load_telecom_data`

```python
def load_telecom_data(filepath: Optional[Path] = None) -> pd.DataFrame
```

Load telecom churn CSV from disk.

**Parameters:**
- `filepath` *(Optional[Path])*: Path to CSV. Defaults to `RAW_DATA_DIR/TELECOM_DATA_FILE`.

**Returns:** `pd.DataFrame`

**Raises:** `FileNotFoundError` if file does not exist.

---

### `validate_data`

```python
def validate_data(df: pd.DataFrame) -> bool
```

Validate DataFrame schema — checks for customer ID columns and monthly feature columns.

**Returns:** `True` on success.

**Raises:** `ValueError` if schema is invalid or DataFrame is empty.

---

### `get_column_types`

```python
def get_column_types(df: pd.DataFrame) -> Tuple[list, list, list]
```

Segregate columns by type.

**Returns:** `(date_columns, id_columns, numeric_columns)`

---

### `handle_missing_values`

```python
def handle_missing_values(df: pd.DataFrame, threshold: float = 0.70) -> pd.DataFrame
```

Drop high-missing columns; impute remainder with 0.

**Parameters:**
- `threshold`: Drop columns with missing rate > this value.

---

### `drop_single_value_columns`

```python
def drop_single_value_columns(df: pd.DataFrame) -> pd.DataFrame
```

Remove constant columns (only 1 unique value).

---

### `drop_date_and_id_columns`

```python
def drop_date_and_id_columns(df: pd.DataFrame) -> pd.DataFrame
```

Remove date and customer ID columns.

---

### `clean_telecom_data`

```python
def clean_telecom_data(df: pd.DataFrame) -> pd.DataFrame
```

Full cleaning pipeline: missing values → constant columns → date/ID columns.

---

### `get_missing_value_summary`

```python
def get_missing_value_summary(df: pd.DataFrame) -> pd.DataFrame
```

Returns a DataFrame with `missing_count` and `missing_pct` for columns with any missing values, sorted descending.

---

### `save_processed_data`

```python
def save_processed_data(df: pd.DataFrame, filename: str = "telecom_churn_processed.csv") -> Path
```

Save DataFrame to `data/processed/`. Returns the output path.

---

## Module: `src.analysis`

### `calculate_churn_rate`

```python
def calculate_churn_rate(df: pd.DataFrame, churn_col: str = "churn") -> float
```

Returns churn rate as a percentage (0–100).

**Raises:** `KeyError` if column not found.

---

### `get_churn_distribution`

```python
def get_churn_distribution(df: pd.DataFrame, churn_col: str = "churn") -> pd.Series
```

Returns value counts of churn vs non-churn.

---

### `filter_high_value_customers`

```python
def filter_high_value_customers(df: pd.DataFrame, percentile: float = 70) -> pd.DataFrame
```

Filter to customers with average recharge spend (months 6+7) >= given percentile.

---

### `tag_churners`

```python
def tag_churners(df: pd.DataFrame) -> pd.DataFrame
```

Add binary `churn` column: 1 if zero call minutes AND zero data in month 9.

---

### `engineer_features`

```python
def engineer_features(df: pd.DataFrame) -> pd.DataFrame
```

Add `total_call_min_8` (total call minutes in month 8) and `total_data_8` (total data in month 8).

---

### `compare_churn_groups`

```python
def compare_churn_groups(df: pd.DataFrame, feature: str, churn_col: str = "churn") -> dict
```

Statistical comparison of a feature between churn/non-churn groups using Mann-Whitney U test.

**Returns dict with keys:**
- `churn_median`, `non_churn_median`
- `churn_mean`, `non_churn_mean`
- `p_value`
- `significant` (bool, p < 0.05)

---

### `get_top_correlated_features`

```python
def get_top_correlated_features(df: pd.DataFrame, target: str = "churn", n: int = 20) -> pd.Series
```

Returns top N features by absolute Pearson correlation with target, sorted descending.

---

### `get_descriptive_stats`

```python
def get_descriptive_stats(df: pd.DataFrame, group_col: str = "churn") -> pd.DataFrame
```

Descriptive statistics (mean, std, quartiles) grouped by churn status.

---

### `train_test_split_stratified`

```python
def train_test_split_stratified(
    df: pd.DataFrame,
    target: str = "churn",
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
```

Stratified train/test split.

**Returns:** `(X_train, X_test, y_train, y_test)`

---

## Module: `src.visualizations`

All plot functions accept an optional `save_path: str` argument to save the figure to disk. All return a `matplotlib.figure.Figure` object.

### `plot_churn_distribution`

```python
def plot_churn_distribution(df, churn_col="churn", figsize=(12,5), save_path=None) -> Figure
```

Bar chart + pie chart of churn vs non-churn counts.

---

### `plot_feature_by_churn`

```python
def plot_feature_by_churn(df, feature, churn_col="churn", kind="box", figsize=(10,6), save_path=None) -> Figure
```

Distribution of `feature` segmented by churn status.

**`kind` options:** `"box"`, `"violin"`, `"hist"`

---

### `plot_median_comparison`

```python
def plot_median_comparison(df, features, churn_col="churn", figsize=(14,6), save_path=None) -> Figure
```

Grouped bar chart of median values for multiple features, split by churn.

---

### `plot_correlation_heatmap`

```python
def plot_correlation_heatmap(df, n_features=20, churn_col="churn", figsize=(12,10), save_path=None) -> Figure
```

Lower-triangle heatmap of Pearson correlations for top N features.

---

### `plot_feature_importance`

```python
def plot_feature_importance(feature_names, importances, top_n=20, figsize=(12,8), save_path=None) -> Figure
```

Horizontal bar chart of feature importance scores.

---

### `plot_roc_curve`

```python
def plot_roc_curve(fpr, tpr, auc_score, model_name="Model", figsize=(8,6), save_path=None) -> Figure
```

ROC curve with AUC annotation.

---

## Module: `src.eda_utils`

Convenience re-export of all public functions from `data_loader`, `analysis`, and `visualizations`.

```python
from src.eda_utils import (
    load_telecom_data,
    clean_telecom_data,
    filter_high_value_customers,
    tag_churners,
    engineer_features,
    plot_churn_distribution,
    # ... all other functions
)
```

See `__all__` in `src/eda_utils.py` for the complete list.
