# Spec: Real-Time Fraud Detection Engine

**Spec ID:** CREDIT-001  
**Status:** Approved  
**Author:** Anshita Bhardwaj  
**Created:** 2026-04-26  

---

## Requirements (EARS Notation)

### Functional Requirements

**FR-001** — WHEN a payment transaction event arrives on the Kafka topic `transactions.raw`,
the system SHALL score the transaction for fraud within 100ms and publish the result to
`transactions.scored`.

**FR-002** — WHILE the fraud score is being computed, the system SHALL retrieve online features
from the Feast feature store using the `account_id` and `merchant_id` as entity keys.

**FR-003** — WHERE the fraud probability exceeds 0.75, the system SHALL automatically:
  (a) publish a HIGH-RISK alert to Kafka topic `alerts.fraud`,
  (b) log the decision to `audit.model_decisions`,
  (c) generate a SHAP explanation with top-5 feature contributions.

**FR-004** — IF the fraud probability is between 0.40 and 0.75, the system SHALL flag the
transaction as REVIEW_REQUIRED and route it to the human-in-the-loop queue.

**FR-005** — WHEN a user queries the `/explain/{txn_id}` endpoint, the system SHALL return
within 500ms: the fraud score, top-5 SHAP features, and at least 1 Dice-ML counterfactual.

**FR-006** — WHERE a transaction is flagged as fraudulent, the system SHALL generate a
counterfactual explanation: the minimum change to transaction attributes that would reduce
the fraud probability below 0.40.

### Non-Functional Requirements

**NFR-001** — Inference latency p99 < 100ms (online scoring path).
**NFR-002** — System must process 10,000 transactions per second at peak load.
**NFR-003** — Model must achieve AUC-ROC ≥ 0.92 on the holdout test set.
**NFR-004** — Demographic parity delta < 0.05 across all protected groups.
**NFR-005** — System availability ≥ 99.9% (< 8.7 hours downtime/year).
**NFR-006** — All predictions must be logged and retained for 7 years (regulatory).

---

## Design

### Data Flow
```
Transaction Event
    → Kafka [transactions.raw]
    → Fraud Scorer (FastAPI consumer)
        → Feast Online Store (Redis) — feature retrieval
        → XGBoost model — fraud probability
        → Isolation Forest — behavioral anomaly score
        → Composite risk score = 0.65 * fraud_prob + 0.35 * anomaly_score
    → Kafka [transactions.scored]
    → Audit log [audit.model_decisions]
    → Alert (if score > 0.75) → Kafka [alerts.fraud]
```

### Features Used
| Feature | Source | Window |
|---|---|---|
| `txn_amount_zscore` | Feast | 30d rolling per account |
| `txn_count_1h` | Feast | 1h rolling per account |
| `txn_count_24h` | Feast | 24h rolling per account |
| `merchant_fraud_rate_30d` | Feast | 30d rolling per merchant |
| `account_age_days` | PostgreSQL | static |
| `is_foreign_merchant` | derived | per transaction |
| `hour_of_day` | derived | per transaction |
| `day_of_week` | derived | per transaction |
| `amount_vs_account_avg` | Feast | 90d rolling per account |
| `velocity_score` | Feast | 5-min event rate per account |

### Acceptance Criteria
- [ ] `/score` endpoint responds in < 100ms for 95th percentile
- [ ] Fraud model AUC ≥ 0.92 on test set
- [ ] SHAP values generated for every prediction
- [ ] Counterfactual generated for every high-risk prediction
- [ ] Fairness gate passes (demographic parity delta < 0.05)
- [ ] All predictions logged to `audit.model_decisions`
- [ ] Kafka consumer lag < 1000 messages at steady state
