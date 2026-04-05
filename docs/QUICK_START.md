# Quick Start Guide — Telecom Churn Case Study

## Prerequisites

- Python 3.8 or higher
- Git

---

## 1. Clone the Repository

```bash
git clone https://github.com/parakhdvaibhav/Telecom_Churn_Case_Study.git
cd Telecom_Churn_Case_Study
```

---

## 2. Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Add the Data

Place the raw data files in `data/raw/`:

```
data/raw/
├── telecom_churn_data.csv                          # Main dataset (99,999 rows)
└── Data+Dictionary-+Telecom+Churn+Case+Study.xlsx  # Data dictionary (optional)
```

> The data files are gitignored and must be obtained separately.

---

## 5. Run the Quick Analysis Demo

```bash
python examples/quick_analysis.py
```

This will:
- Load and validate data
- Filter high-value customers
- Tag churners
- Engineer features
- Run comparative analysis
- Save plots to `reports/`

---

## 6. Explore the Notebooks

Launch Jupyter:

```bash
jupyter notebook
```

Then open one of the three analysis notebooks:

| Notebook | Focus |
|----------|-------|
| `notebooks/01_data_exploration.ipynb` | Data loading, schema inspection, missing values, distributions |
| `notebooks/02_detailed_analysis.ipynb` | High-value filtering, churn tagging, bivariate analysis, feature engineering |
| `notebooks/03_churn_modeling_insights.ipynb` | PCA, model training, evaluation, feature importance |

---

## 7. Run the Tests

```bash
pytest tests/ -v
```

To see coverage:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 8. Using the `src` Package in Your Own Code

```python
from src.data_loader import load_telecom_data, clean_telecom_data
from src.analysis import filter_high_value_customers, tag_churners, engineer_features
from src.visualizations import plot_churn_distribution

# Load data
df = load_telecom_data("data/raw/telecom_churn_data.csv")

# Clean
df_clean = clean_telecom_data(df)

# Filter high-value customers
df_hv = filter_high_value_customers(df_clean, percentile=70)

# Tag churners
df_tagged = tag_churners(df_hv)

# Visualize
fig = plot_churn_distribution(df_tagged)
```

---

## 9. Project Structure

```
Telecom_Churn_Case_Study/
├── data/
│   ├── raw/          # Raw input data (gitignored)
│   └── processed/    # Cleaned/transformed data (gitignored)
├── docs/             # Documentation
├── examples/         # Runnable demo scripts
├── notebooks/        # Jupyter analysis notebooks
├── reports/          # Generated plots and reports (gitignored)
├── src/              # Reusable Python package
│   ├── __init__.py
│   ├── config.py     # Centralized configuration
│   ├── data_loader.py
│   ├── analysis.py
│   ├── visualizations.py
│   └── eda_utils.py  # Convenience re-exports
├── tests/            # Unit tests
├── .github/workflows/ # CI/CD pipelines
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## 10. Configuration

All key thresholds and settings are in `src/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MISSING_VALUE_THRESHOLD` | `0.70` | Drop columns with >70% missing |
| `HIGH_VALUE_PERCENTILE` | `70` | 70th percentile for high-value filter |
| `CORRELATION_THRESHOLD` | `0.85` | High correlation threshold |
| `PCA_VARIANCE_THRESHOLD` | `0.96` | 96% variance target for PCA |
| `N_PCA_COMPONENTS` | `18` | Number of PCA components |
| `TEST_SIZE` | `0.25` | Train/test split ratio |
| `RANDOM_STATE` | `42` | Random seed |
