# Data Dictionary — Telecom Churn Case Study

This document describes all features in the telecom churn dataset (99,999 rows, 226 columns, 4 months: June=6, July=7, August=8, September=9).

---

## 1. Customer Identifiers

| Column | Type | Description |
|--------|------|-------------|
| `mobile_number` | int64 | Unique mobile subscriber identifier |
| `circle_id` | int64 | Telecom circle (geographic region) ID |

---

## 2. Monthly Usage Features

Features are suffixed `_6`, `_7`, `_8`, `_9` for the respective month.

### 2.1 Revenue & Recharge

| Column Pattern | Type | Description |
|----------------|------|-------------|
| `arpu_{m}` | float64 | Average Revenue Per User in month `m` (INR) |
| `total_rech_num_{m}` | float64 | Total number of recharges in month `m` |
| `total_rech_amt_{m}` | float64 | Total recharge amount in month `m` (INR) |
| `max_rech_amt_{m}` | float64 | Maximum single recharge amount in month `m` (INR) |
| `date_of_last_rech_{m}` | object | Date of last recharge in month `m` (YYYY-MM-DD) |
| `last_day_rch_amt_{m}` | float64 | Recharge amount on last recharge day in month `m` (INR) |

### 2.2 Data Recharge

| Column Pattern | Type | Description |
|----------------|------|-------------|
| `total_rech_data_{m}` | float64 | Total number of data recharges in month `m` |
| `av_rech_amt_data_{m}` | float64 | Average recharge amount for data packs in month `m` (INR) |
| `max_rech_data_{m}` | float64 | Maximum data recharge amount in month `m` (INR) |
| `date_of_last_rech_data_{m}` | object | Date of last data recharge in month `m` (YYYY-MM-DD) |
| `count_rech_2g_{m}` | float64 | Number of 2G data recharges in month `m` |
| `count_rech_3g_{m}` | float64 | Number of 3G data recharges in month `m` |

### 2.3 Call Minutes of Use (MOU)

| Column Pattern | Type | Description |
|----------------|------|-------------|
| `total_og_mou_{m}` | float64 | Total outgoing call minutes in month `m` |
| `total_ic_mou_{m}` | float64 | Total incoming call minutes in month `m` |
| `onnet_mou_{m}` | float64 | Outgoing on-net (same operator) call minutes in month `m` |
| `offnet_mou_{m}` | float64 | Outgoing off-net (different operator) call minutes in month `m` |
| `roam_og_mou_{m}` | float64 | Outgoing roaming call minutes in month `m` |
| `roam_ic_mou_{m}` | float64 | Incoming roaming call minutes in month `m` |
| `loc_og_mou_{m}` | float64 | Local outgoing call minutes in month `m` |
| `loc_ic_mou_{m}` | float64 | Local incoming call minutes in month `m` |
| `std_og_mou_{m}` | float64 | STD (long-distance) outgoing call minutes in month `m` |
| `std_ic_mou_{m}` | float64 | STD incoming call minutes in month `m` |
| `isd_og_mou_{m}` | float64 | ISD (international) outgoing call minutes in month `m` |
| `isd_ic_mou_{m}` | float64 | ISD incoming call minutes in month `m` |
| `spl_og_mou_{m}` | float64 | Special outgoing call minutes in month `m` |
| `spl_ic_mou_{m}` | float64 | Special incoming call minutes in month `m` |

### 2.4 Data Volumes

| Column Pattern | Type | Description |
|----------------|------|-------------|
| `vol_2g_mb_{m}` | float64 | 2G data consumption in month `m` (MB) |
| `vol_3g_mb_{m}` | float64 | 3G data consumption in month `m` (MB) |
| `monthly_2g_{m}` | float64 | Monthly 2G data pack indicator in month `m` |
| `monthly_3g_{m}` | float64 | Monthly 3G data pack indicator in month `m` |
| `sachet_2g_{m}` | float64 | Sachet (daily/weekly) 2G data pack count in month `m` |
| `sachet_3g_{m}` | float64 | Sachet 3G data pack count in month `m` |

### 2.5 Night-time & Weekend Usage

| Column Pattern | Type | Description |
|----------------|------|-------------|
| `night_pck_user_{m}` | float64 | Night pack user indicator (1=yes) in month `m` |
| `fb_user_{m}` | float64 | Facebook pack user indicator (1=yes) in month `m` |

---

## 3. Engineered Features

These features are created during the analysis pipeline from existing columns.

| Column | Formula | Description |
|--------|---------|-------------|
| `total_data_rech_amt_6` | `total_rech_data_6 × av_rech_amt_data_6` | Total data recharge spend in month 6 |
| `total_data_rech_amt_7` | `total_rech_data_7 × av_rech_amt_data_7` | Total data recharge spend in month 7 |
| `total_amt_6` | `total_rech_amt_6 + total_data_rech_amt_6` | Total recharge spend (voice + data) in month 6 |
| `total_amt_7` | `total_rech_amt_7 + total_data_rech_amt_7` | Total recharge spend (voice + data) in month 7 |
| `average_amt_6_7` | `(total_amt_6 + total_amt_7) / 2` | Average recharge spend across good-phase months (used for high-value filtering) |
| `total_call_min_8` | `total_ic_mou_8 + total_og_mou_8` | Total call minutes (in + out) in action month (August) |
| `total_data_8` | `vol_2g_mb_8 + vol_3g_mb_8` | Total data consumption in action month (August, MB) |

---

## 4. Target Variable

| Column | Type | Values | Description |
|--------|------|--------|-------------|
| `churn` | int64 | 0, 1 | Binary churn label: **1** = churned, **0** = retained |

**Churn Definition:** A high-value customer is labeled as churned (`churn=1`) if:
- `total_ic_mou_9 + total_og_mou_9 == 0` (zero call usage in September), **AND**
- `vol_2g_mb_9 + vol_3g_mb_9 == 0` (zero data usage in September)

All month-9 columns are **dropped** before model training, since they define the target and would cause data leakage.

---

## 5. Data Scope & Known Characteristics

| Property | Value |
|----------|-------|
| Total rows | 99,999 |
| Total columns (raw) | 226 |
| Time span | June–September 2014 |
| Geography | Indian telecom market |
| Churn rate (high-value) | ~8–9% |
| High-value filter | Average recharge ≥ 70th percentile (months 6–7) |
| Missing value strategy | Drop columns >70% missing; impute remainder with 0 |
| PCA components used | 18 (capturing ~96% variance) |

---

## 6. Month Naming Convention

| Value | Month | Phase |
|-------|-------|-------|
| `6` | June | Good phase (baseline behavior) |
| `7` | July | Good phase (baseline behavior) |
| `8` | August | Action phase (early churn signals) |
| `9` | September | Churn phase (labeling only — dropped from features) |
