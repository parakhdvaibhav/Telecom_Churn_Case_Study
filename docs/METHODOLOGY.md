# Methodology — Telecom Churn Case Study

## Overview

This document describes the end-to-end machine-learning pipeline used to predict customer churn for a prepaid telecom operator in India. The study uses a usage-based churn definition on 4 months of subscriber data (June–September 2014).

---

## Step 1: Data Loading & Validation

**Input:** `telecom_churn_data.csv` — 99,999 rows, 226 columns

**Steps:**
1. Load CSV with `pandas.read_csv`.
2. Schema validation — confirm customer ID columns (`mobile_number`, `circle_id`) and all four month suffixes (`_6`, `_7`, `_8`, `_9`) exist.
3. Log dataset shape and column types.

**Key module:** `src/data_loader.py` → `load_telecom_data`, `validate_data`

---

## Step 2: Exploratory Data Analysis & Cleaning

### 2.1 Missing Value Treatment

- Compute missing rate per column.
- **Drop** columns with >70% missing values (threshold configurable via `MISSING_VALUE_THRESHOLD`).
- **Impute** remaining missing values with `0` (appropriate for telecom usage — no usage recorded = zero consumption).

### 2.2 Constant Column Removal

- Drop columns with only one unique value (zero predictive power).
- Typical candidates: indicator columns that are 100% zero in the dataset.

### 2.3 Date & ID Column Removal

- Drop `date_of_last_rech_{m}` and `date_of_last_rech_data_{m}` columns (used only for temporal reference, not modeling).
- Drop `mobile_number` and `circle_id` (identifiers with no predictive signal).

### 2.4 High Correlation Check

- Identify pairs of features with Pearson correlation > 0.85.
- These are candidates for removal in the PCA/feature selection step.

**Key module:** `src/data_loader.py` → `clean_telecom_data`, `handle_missing_values`

---

## Step 3: High-Value Customer Filtering

Only high-value customers are considered for churn prediction, as they represent the most revenue-critical segment.

**Definition:** A customer is "high-value" if their average total recharge spend across the good-phase months (June + July) is at or above the **70th percentile**.

**Formula:**
```
total_amt_6 = total_rech_amt_6 + (total_rech_data_6 × av_rech_amt_data_6)
total_amt_7 = total_rech_amt_7 + (total_rech_data_7 × av_rech_amt_data_7)
average_amt_6_7 = (total_amt_6 + total_amt_7) / 2

high_value = average_amt_6_7 >= percentile(average_amt_6_7, 70)
```

**Result:** ~30% of customers retained (~30,000 records).

**Key module:** `src/analysis.py` → `filter_high_value_customers`

---

## Step 4: Churn Tagging (Target Variable Creation)

Churn is defined using **usage-based business rules** applied to September (month 9) data:

```
churn = 1  if  (total_ic_mou_9 + total_og_mou_9 == 0)
              AND
              (vol_2g_mb_9 + vol_3g_mb_9 == 0)
churn = 0  otherwise
```

**Rationale:** A prepaid customer who makes no calls and uses no data in a given month has effectively churned (ported to another operator or stopped using the service).

**Churn rate:** ~8–9% in the high-value segment.

> **Important:** All month-9 columns are **dropped** from the feature set before modeling to prevent data leakage.

**Key module:** `src/analysis.py` → `tag_churners`

---

## Step 5: Feature Engineering

### 5.1 Engineered Features

| Feature | Description |
|---------|-------------|
| `total_call_min_8` | `total_ic_mou_8 + total_og_mou_8` — total call usage in action month |
| `total_data_8` | `vol_2g_mb_8 + vol_3g_mb_8` — total data usage in action month |

### 5.2 Dropped Features (Month 9)

All columns ending in `_9` are removed after churn labeling to avoid leakage.

**Key module:** `src/analysis.py` → `engineer_features`

---

## Step 6: Modeling Pipeline

### 6.1 Train/Test Split

- **Split ratio:** 75% train / 25% test
- **Stratification:** By churn label (to preserve ~8–9% churn rate in both sets)
- **Random state:** 42

### 6.2 Class Imbalance Handling

Two strategies are evaluated:
1. **`class_weight='balanced'`** — Scikit-learn built-in weight adjustment
2. **SMOTE** (`imbalanced-learn`) — Synthetic Minority Over-sampling Technique

### 6.3 Dimensionality Reduction (PCA)

After scaling (StandardScaler):
- PCA applied to retain **18 components** (capturing **~96% explained variance**).
- PCA is fit on training data only; applied to test data to prevent leakage.

### 6.4 Models Evaluated

| Model | Library | Variants |
|-------|---------|----------|
| Logistic Regression | scikit-learn | `class_weight='balanced'`, SMOTE |
| Decision Tree | scikit-learn | `class_weight='balanced'` |
| Random Forest | scikit-learn | `class_weight='balanced'`, SMOTE |
| Gradient Boosting | scikit-learn | Default |
| XGBoost | xgboost | `scale_pos_weight` |

### 6.5 Evaluation Metrics

| Metric | Rationale |
|--------|-----------|
| **ROC-AUC** | Primary metric — robust to class imbalance |
| Precision | Minimize false positives (costly intervention on non-churners) |
| Recall | Maximize true positive rate (identify churners before they leave) |
| F1-Score | Harmonic mean of precision and recall |
| Confusion Matrix | Visualize TP, TN, FP, FN breakdown |

---

## Step 7: Results & Best Model

| Model | ROC-AUC | Notes |
|-------|---------|-------|
| Logistic Regression (balanced) | ~0.78 | Baseline |
| Decision Tree | ~0.72 | Prone to overfitting |
| **Random Forest (balanced)** | **~0.87** | ✅ Best overall |
| Gradient Boosting | ~0.84 | Competitive, slower |
| XGBoost | ~0.85 | Best for speed/scale |

**Best model:** Random Forest with `class_weight='balanced'`
- Highest ROC-AUC (~0.87)
- Robust to overfitting via ensemble averaging
- Provides interpretable feature importances

---

## Step 8: Key Churn Indicators (Feature Importance)

Top predictors from Random Forest:

1. **Declining ARPU** from months 6/7 → 8 (revenue drop)
2. **Reduced call minutes** in month 8 (`total_call_min_8`)
3. **Reduced data usage** in month 8 (`total_data_8`)
4. **Drop in recharge frequency** in month 8
5. **Lower max recharge amount** in month 8
6. **Decline in on-net call minutes** (moving off the network)

---

## Reproducibility

All random seeds are fixed at `RANDOM_STATE = 42`. The full pipeline can be reproduced by:

```bash
pip install -r requirements.txt
python examples/quick_analysis.py
```

Or via the Jupyter notebooks in `notebooks/`.
