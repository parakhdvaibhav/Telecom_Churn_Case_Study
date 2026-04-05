# Telecom Churn Case Study

![CI](https://github.com/parakhdvaibhav/Telecom_Churn_Case_Study/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A production-grade machine learning project predicting customer churn for a prepaid telecom operator in India. Built on 4 months of subscriber usage data (June–September 2014) with ~30,000 high-value customers after filtering.

---

## Executive Summary

Telecom operators face significant revenue loss from customer churn, particularly among high-value prepaid subscribers. This project builds a predictive model to identify at-risk customers **1 month before** they are likely to churn, enabling targeted retention interventions.

**Best model:** Random Forest with `class_weight='balanced'`  
**ROC-AUC:** ~0.87  
**Dataset:** 99,999 customers × 226 features × 4 months

---

## Key Findings

| # | Finding | Signal |
|---|---------|--------|
| 1 | **Declining ARPU** is the strongest churn predictor | >25% drop month-over-month |
| 2 | **Reduced call minutes in August** signals early disengagement | ~60% lower than non-churners |
| 3 | **Zero data usage in August** confirms service abandonment | Near-zero `total_data_8` |
| 4 | **Recharge frequency drops** 4–6 weeks before churn | No recharge in 15+ days |
| 5 | **On-net call ratio declines** indicate competitor SIM adoption | Shift to off-net calls |
| 6 | **Mid-tier high-value customers** (70th–85th percentile) have highest ROI for retention | Price-sensitive but high-revenue |

---

## Project Structure

```
Telecom_Churn_Case_Study/
├── data/
│   ├── raw/                    # Raw input data (gitignored)
│   └── processed/              # Cleaned/processed data (gitignored)
├── docs/
│   ├── DATA_DICTIONARY.md      # Feature descriptions for all 226 columns
│   ├── METHODOLOGY.md          # End-to-end pipeline documentation
│   ├── FINDINGS.md             # Key findings and business recommendations
│   ├── QUICK_START.md          # Setup and usage guide
│   └── API_REFERENCE.md        # Full API documentation for src/ package
├── examples/
│   └── quick_analysis.py       # Runnable demo script (synthetic data)
├── notebooks/
│   ├── 01_data_exploration.ipynb        # Data loading, EDA, distributions
│   ├── 02_detailed_analysis.ipynb       # High-value filtering, churn tagging, bivariate analysis
│   └── 03_churn_modeling_insights.ipynb # PCA, model training, evaluation, feature importance
├── reports/                    # Generated plots (gitignored)
├── src/
│   ├── __init__.py
│   ├── config.py               # Centralized configuration
│   ├── data_loader.py          # Data loading, validation, cleaning
│   ├── analysis.py             # Statistical analysis, feature engineering
│   ├── visualizations.py       # Reusable plotting functions
│   └── eda_utils.py            # Convenience re-exports
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py     # Unit tests for data_loader module
│   ├── test_analysis.py        # Unit tests for analysis module
│   └── test_visualizations.py  # Unit tests for visualizations module
├── .github/workflows/
│   ├── ci.yml                  # Lint + test CI pipeline (Python 3.8–3.11)
│   └── notebook-validation.yml # Notebook syntax validation
├── Telecom_Churn_Group+Assignment_Vaibhav.ipynb  # Original reference notebook
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Quick Start

### Prerequisites
- Python 3.8+

### Installation

```bash
git clone https://github.com/parakhdvaibhav/Telecom_Churn_Case_Study.git
cd Telecom_Churn_Case_Study
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Add Data

Place the data files in `data/raw/`:
```
data/raw/telecom_churn_data.csv
data/raw/Data+Dictionary-+Telecom+Churn+Case+Study.xlsx  (optional)
```

### Run Demo (Synthetic Data)

```bash
python examples/quick_analysis.py
```

### Run Notebooks

```bash
jupyter notebook
# Open notebooks/01_data_exploration.ipynb
```

### Run Tests

```bash
pytest tests/ -v
```

---

## Modeling Pipeline

```
Raw Data (99,999 × 226)
    │
    ▼ Clean (drop >70% missing, impute zeros, drop constants)
    │
    ▼ Filter High-Value Customers (≥ 70th percentile recharge, months 6+7)
    │                                   ↓ ~30,000 customers
    ▼ Tag Churners (zero calls AND zero data in month 9)
    │                                   ↓ ~8–9% churn rate
    ▼ Engineer Features (total_call_min_8, total_data_8)
    │
    ▼ Drop month-9 columns (prevent leakage)
    │
    ▼ Train/Test Split (75/25, stratified)
    │
    ▼ StandardScaler → PCA (18 components, 96% variance)
    │
    ▼ Train Models (LR, DT, RF, GBM, XGBoost)
    │
    ▼ Evaluate (ROC-AUC, Precision, Recall, F1)
    │
    ▼ Best: Random Forest (balanced) — AUC ~0.87
```

---

## Model Results

| Model | ROC-AUC |
|-------|---------|
| Logistic Regression (balanced) | ~0.78 |
| Decision Tree | ~0.72 |
| **Random Forest (balanced)** | **~0.87** ✓ |
| Gradient Boosting | ~0.84 |
| XGBoost | ~0.85 |

---

## Using the `src` Package

```python
from src.data_loader import load_telecom_data, clean_telecom_data
from src.analysis import filter_high_value_customers, tag_churners, engineer_features
from src.visualizations import plot_churn_distribution

df = load_telecom_data("data/raw/telecom_churn_data.csv")
df_clean = clean_telecom_data(df)
df_hv = filter_high_value_customers(df_clean, percentile=70)
df_tagged = tag_churners(df_hv)
fig = plot_churn_distribution(df_tagged)
```

See [docs/API_REFERENCE.md](docs/API_REFERENCE.md) for full documentation.

---

## Configuration

All thresholds are in `src/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MISSING_VALUE_THRESHOLD` | `0.70` | Drop columns with >70% missing |
| `HIGH_VALUE_PERCENTILE` | `70` | High-value customer threshold |
| `PCA_VARIANCE_THRESHOLD` | `0.96` | Target PCA variance |
| `N_PCA_COMPONENTS` | `18` | Number of PCA components |
| `RANDOM_STATE` | `42` | Reproducibility seed |

---

## Documentation

| Document | Description |
|----------|-------------|
| [DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) | All 226 features explained |
| [METHODOLOGY.md](docs/METHODOLOGY.md) | Full 8-step pipeline |
| [FINDINGS.md](docs/FINDINGS.md) | Key churn drivers and business recommendations |
| [QUICK_START.md](docs/QUICK_START.md) | Setup guide |
| [API_REFERENCE.md](docs/API_REFERENCE.md) | `src` package API docs |

---

## Requirements

Core: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `xgboost`, `imbalanced-learn`  
Dev: `pytest`, `pytest-cov`, `black`, `flake8`

See [requirements.txt](requirements.txt) for pinned versions.
