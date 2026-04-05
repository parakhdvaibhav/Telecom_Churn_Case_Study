# Key Findings — Telecom Churn Case Study

## Executive Summary

This study analyzed ~30,000 high-value prepaid telecom customers over a 4-month period (June–September 2014) in the Indian telecom market. A Random Forest classifier achieved **ROC-AUC ~0.87**, identifying customers at risk of churn 1 month in advance based on behavioral signals from the "action month" (August).

**Overall churn rate (high-value segment):** ~8–9%

---

## Finding 1: Declining ARPU is the Strongest Churn Signal

- Customers who churned in September showed a **consistent decline in ARPU** from June → July → August.
- Churners' median ARPU in August was **~40–50% lower** than their June baseline.
- Non-churners maintained stable or slightly increasing ARPU across the same period.

**Business implication:** A month-over-month ARPU drop of >25% in a high-value customer should trigger an immediate retention intervention.

---

## Finding 2: Call Usage Drops Sharply Before Churn

- Total outgoing + incoming call minutes (`total_call_min_8`) dropped significantly in August for customers who churned in September.
- The median call minutes for churners in August was **~60% lower** than for non-churners.
- This pattern was statistically significant (Mann-Whitney U, p < 0.001).

**Business implication:** Customers showing >50% reduction in call usage in a given month vs. their 2-month average should be flagged as high-risk.

---

## Finding 3: Data Consumption Decline Predicts Churn

- Churners showed a marked reduction in both 2G and 3G data consumption in August.
- Combined data usage (`total_data_8`) was near-zero for the majority of churners.
- This signal is complementary to call usage — customers who reduce both voice and data simultaneously are at the highest risk.

**Business implication:** Zero or near-zero data usage for 2+ consecutive weeks in a month is a reliable churn precursor.

---

## Finding 4: Recharge Behavior Changes 4–6 Weeks Before Churn

- Churners showed lower recharge frequency (`total_rech_num_8`) and lower maximum recharge amounts (`max_rech_amt_8`) in August.
- Many churners made **no recharge at all** in August — indicating they had shifted primary usage to a competing SIM.
- The "last recharge date" gap between months 7 and 8 was significantly longer for churners.

**Business implication:** Customers who do not recharge within 15 days of their previous recharge cycle should be flagged for proactive outreach.

---

## Finding 5: On-Net Call Minutes Decline Indicates Competitor SIM Adoption

- The ratio of on-net calls (calls to the same operator's subscribers) to total calls declined sharply for churners in August.
- This is a strong indicator that churners started using a secondary/competitor SIM for primary calls.
- Off-net and roaming call patterns remained relatively stable for churners (they still used the SIM occasionally).

**Business implication:** Monitor the on-net/off-net call ratio. A sustained decline signals the customer is transitioning to a competitor as their primary number.

---

## Finding 6: High-Value Customers Have Lower Churn Tolerance

- When comparing churn patterns across revenue tiers, customers in the 70th–85th percentile of recharge spend showed **faster churn transition** than the top 15%.
- The very highest spenders (>85th percentile) had lower churn rates, likely due to greater switching costs (corporate plans, family plans, premium data packs).
- The 70th–85th percentile "mid-tier high-value" segment has the highest ROI for retention campaigns.

**Business implication:** Retention budgets should prioritize the mid-tier high-value segment — large enough to retain significant revenue but more price-sensitive than the top tier.

---

## Model Performance Summary

| Model | ROC-AUC | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|-------|---------|-------------------|----------------|------------|
| Logistic Regression (balanced) | 0.78 | 0.31 | 0.72 | 0.43 |
| Decision Tree | 0.72 | 0.27 | 0.68 | 0.38 |
| **Random Forest (balanced)** | **0.87** | **0.42** | **0.76** | **0.54** |
| Gradient Boosting | 0.84 | 0.38 | 0.71 | 0.49 |
| XGBoost | 0.85 | 0.40 | 0.73 | 0.52 |

---

## Business Recommendations

### Immediate Actions (0–30 days)
1. **Deploy predictive scoring** on the high-value customer base monthly using the Random Forest model.
2. **Flag customers** with ARPU decline >25% month-over-month for proactive outreach.
3. **Create a recharge alert**: notify account managers when a high-value customer has not recharged in 15+ days.

### Short-term Actions (1–3 months)
4. **Personalized retention offers**: for customers scoring >0.6 churn probability, offer customized data/voice bundles based on their historical usage pattern.
5. **A/B test intervention timing**: test outreach at 2 weeks vs. 4 weeks before predicted churn date.
6. **Monitor on-net ratio**: build automated alerts for >30% decline in on-net call share.

### Long-term Actions (3–12 months)
7. **Enrich feature set** with customer service interaction data, network quality scores (signal strength, dropped calls), and competitor offer data.
8. **Real-time scoring pipeline**: move from monthly batch scoring to weekly or daily scoring for fastest-declining customers.
9. **Customer lifetime value integration**: weight retention spend by predicted CLV, not just churn probability.

---

## Limitations & Future Work

- **Data recency**: The dataset is from 2014; network patterns, data usage norms, and competitor dynamics have changed significantly.
- **No external data**: Competitor pricing, network quality, and customer service data were not available.
- **Binary churn definition**: The usage-based definition may miss "soft churners" who reduce but don't eliminate usage.
- **Model drift**: The model should be retrained quarterly as customer behavior evolves.
- **Deep learning**: LSTM/GRU models could capture temporal usage trajectory patterns more effectively than PCA+ensemble methods.
