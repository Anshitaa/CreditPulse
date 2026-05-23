# CreditPulse — Model Risk Card

**Version:** 1.0.0  
**Generated:** 2026-05-23  
**Author:** Anshita Bhardwaj  
**Scored population:** 150,000 synthetic transactions (12.5% fraud prevalence)  
**Spec linkage:** CREDIT-001 (fraud scoring), CREDIT-002 (credit risk), CREDIT-003 (explainability)

---

## 1. Model Suite Overview

CreditPulse deploys three models in a layered risk pipeline:

| Model | Algorithm | Purpose | Latency Target |
|---|---|---|---|
| `fraud_detector` | XGBoost (gradient boosted trees) | Primary fraud classification | p99 < 100ms |
| `anomaly_detector` | Isolation Forest | Unsupervised outlier detection | p99 < 50ms |
| `credit_risk_scorer` | XGBoost regression | Credit risk probability (0–1) | p99 < 50ms |

**Composite Decision Logic:**
```
fraud_risk_score (0–100) = fraud_probability × 100
Decision: FRAUD if score > 75 | REVIEW if 40–75 | CLEAR if < 40
```

Counterfactual explanations (Dice-ML) and SHAP feature attribution are computed per transaction and stored in `audit.explanations`.

---

## 2. Training Data

| Attribute | Value |
|---|---|
| Generator | `data/synthetic_transactions.py` (deterministic seed) |
| Total transactions | 150,000 |
| Fraud transactions | 18,757 (12.5%) |
| Date range | 2023-01-01 – 2024-12-31 (2-year window) |
| Unique accounts | ~5,000 |
| Unique merchants | ~500 |
| Train / validation split | 80% / 20% (stratified) |
| Feature sources | Transaction attributes + engineered velocity features |

> **Note on synthetic data:** Labels are generated via additive risk scoring over
> account age, merchant category, transaction velocity, time-of-day, and amount
> deviation. The model achieves AUC-ROC of 0.68 — realistic for fraud detection
> on imbalanced data without feature leakage. Real-world fraud models typically
> achieve 0.65–0.85 AUC depending on feature richness.

---

## 3. Performance Metrics (Validation Set — 20% holdout)

### 3a. Fraud Detector (XGBoost Classifier)

| Metric | Value | Notes |
|---|---|---|
| ROC-AUC | **0.681** | Realistic; synthetic labels don't leak features |
| Average Precision | 0.220 | Reflects 12.5% class imbalance |
| F1 Score | 0.224 | Low due to imbalance; use PR-AUC for model selection |
| Decision threshold | 0.75 (score > 75 = FRAUD) | Tuned for precision at high-risk band |

> **Interview note:** Low F1 with reasonable AUC is expected in fraud detection. We optimize Average Precision (area under PR curve) rather than F1, which treats precision and recall symmetrically. High-precision at the FRAUD decision threshold matters more than global F1.

### 3b. Credit Risk Scorer (XGBoost Regressor)

| Metric | Value |
|---|---|
| R² | 0.964 |
| MAE | 0.025 |
| RMSE | 0.029 |

### 3c. Anomaly Detector (Isolation Forest)

| Metric | Value |
|---|---|
| Anomaly rate (training) | 2.0% |
| Mean normalized score | 0.796 |
| Std normalized score | 0.162 |

---

## 4. Decision Band Distribution

> **Note:** The following distribution reflects integration test traffic (mostly
> high-risk synthetic transactions). Production distribution will shift toward
> CLEAR as real low-risk transactions are scored. Raw data fraud prevalence is 12.5%.

| Decision | Count (integration tests) | Threshold |
|---|---|---|
| FRAUD | 224 | score > 75 |
| REVIEW | 18 | score 40–75 |
| CLEAR | 0 | score < 40 |

---

## 5. Fairness Audit (Fairlearn)

**Threshold:** All fairness metrics must be < 0.05 for gate to pass.  
**Protected groups evaluated:** `account_type`, `account_age_group`, `region_type`  
**Spec:** CREDIT-001 NFR-004

| Group | Metric | Value | Status |
|---|---|---|---|
| `account_type` | Demographic Parity Diff | 0.0200 | ✓ Pass |
| `account_type` | Equal Opportunity Diff | 0.0302 | ✓ Pass |
| `account_type` | Predictive Parity Diff | **0.0461** | ⚠ Borderline |
| `account_age_group` | Demographic Parity Diff | 0.0045 | ✓ Pass |
| `account_age_group` | Equal Opportunity Diff | 0.0065 | ✓ Pass |
| `account_age_group` | Predictive Parity Diff | **0.0552** | ⚠ Exceeds threshold |
| `region_type` | Demographic Parity Diff | 0.0106 | ✓ Pass |
| `region_type` | Equal Opportunity Diff | 0.0201 | ✓ Pass |
| `region_type` | Predictive Parity Diff | 0.0328 | ✓ Pass |

**Interpretation:**
- `account_age_group` predictive parity (0.055) slightly exceeds the 0.05 threshold — the model's precision is modestly lower for newer accounts. This is a known distributional artifact: new accounts have fewer historical velocity features, making them harder to classify accurately regardless of true fraud risk.
- Demographic parity and equal opportunity are within bounds for all groups, meaning the model does not systematically flag any group at higher rates or miss more fraud in any group.

**Recommended action:** Collect more labeled data for accounts aged < 90 days, or apply isotonic recalibration per account-age group before promoting to production.

---

## 6. Explainability

Every scored transaction includes:
- **SHAP top-5 features** with direction (`increases_risk` / `decreases_risk`)
- **Dice-ML counterfactual:** "Your score would be < 50 if you changed amount to $X or merchant to grocery"
- Stored in `audit.explanations` for regulatory audit trail

**Top features by mean |SHAP| across 242 scored transactions:**

| Rank | Feature | Direction |
|---|---|---|
| 1 | `amount_vs_avg_ratio` | Increases risk at high values |
| 2 | `txn_velocity_1h` | Increases risk at high values |
| 3 | `merchant_category` (wire_transfer) | Increases risk |
| 4 | `hour_of_day` (late night) | Increases risk |
| 5 | `is_foreign_merchant` | Increases risk when True |

**Regulatory compliance (RAG-indexed):**  
The agent has a FAISS index over 8 regulatory documents: PCI DSS, FCRA, Regulation E, CFPB adverse action guidance, BSA/AML, OCC model risk guidance (SR 11-7), Fair Housing Act, ECOA. Counterfactual explanations are designed to satisfy FCRA §615(a) adverse action notice requirements.

---

## 7. Drift Monitoring (PSI)

Monitoring cadence: weekly (Airflow DAG) + on-demand via `POST /governance/drift/run`.

| Feature | PSI | Status | Action |
|---|---|---|---|
| `amount` | 0.004 | STABLE | None |
| `txn_velocity_1h` | 0.002 | STABLE | None |
| `amount_vs_avg_ratio` | 0.003 | STABLE | None |
| `hour_of_day` | 0.001 | STABLE | None |
| `day_of_week` | 0.001 | STABLE | None |
| `is_foreign_merchant` | 0.002 | STABLE | None |
| `merchant_category` | 0.003 | STABLE | None |

**PSI interpretation:**
- < 0.10 = STABLE (no action)
- 0.10–0.20 = MONITOR (increase cadence)
- > 0.20 = RETRAIN (trigger retraining pipeline)

**Retrain trigger:** PSI > 0.20 on any feature OR fairness gate failure on 2 consecutive weekly runs.

---

## 8. Latency SLA (Spec: CREDIT-001 NFR-001)

Measured across 20 consecutive requests in integration test suite:

| Percentile | Inference Latency | Status |
|---|---|---|
| p50 | ~3ms | ✓ |
| p99 | < 10ms | ✓ |
| Target | < 100ms | ✓ PASS |

> End-to-end request latency (including DB audit write + Kafka publish) is higher (~50–200ms) but the model inference itself consistently meets the < 100ms SLA.

---

## 9. Operational Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Model staleness (concept drift) | Medium | High | Weekly PSI monitoring + auto-retrain trigger |
| Fairness regression post-retrain | Low | High | Fairlearn gate enforced as Kiro hook on every model save |
| SHAP explanation instability | Low | Medium | SHAP values locked to model version; versioned in MLflow |
| False negative fraud (miss) | Medium | High | Anomaly detector provides independent signal; REVIEW band for human review |
| Adversarial feature manipulation | Low | Medium | Velocity features computed server-side; client cannot spoof |
| Label noise in training data | Low | Medium | Synthetic labels derived from ground-truth risk factors; no annotator disagreement |

---

## 10. Governance & Compliance Lineage

| Artifact | Location |
|---|---|
| Model artifacts | `models/artifacts/*.pkl` |
| Training experiments | `mlruns/` (MLflow file-based) |
| Fairness reports | `audit.fairness_reports` (PostgreSQL) |
| Drift reports | `audit.drift_reports` (PostgreSQL) |
| Scoring decisions | `audit.model_decisions` (append-only) |
| SHAP explanations | `audit.explanations` (per-transaction) |
| Kiro specs | `.kiro/specs/` (EARS requirements) |
| Kiro hooks | `.kiro/hooks/` (fairness-gate, psi-check, auto-test) |
| This document | `docs/model_risk_card_v1.0.0.md` |

**Retraining process:**
1. Drift monitor exceeds PSI threshold → Airflow triggers `spark_features_dag`
2. Feature pipeline regenerates `mart.feature_store`
3. `models/fraud_detector.py --train --n-trials 20` (Optuna HPO)
4. Fairness gate hook fires automatically on model file save (Kiro)
5. If gate passes → MLflow registers new version as "Staging"
6. Human sign-off required to promote to "Production" (HITL)

---

## 11. Limitations & Known Gaps

1. **Synthetic data ceiling:** AUC will likely shift when deployed against real transaction data. Performance claims should be re-validated before production use.
2. **Account-age fairness:** `account_age_group` predictive parity (0.055) slightly exceeds threshold. Needs recalibration for new-account segment.
3. **Spark Streaming features:** Real-time velocity features (Spark → Feast → Redis) are architecturally implemented but the JDBC JAR for PySpark must be downloaded separately (`spark/jars/postgresql-42.7.3.jar`). Batch velocity features are used in the current demo.
4. **No cold-start handling:** Accounts with zero transaction history receive default feature values (`txn_velocity_1h=0`, `amount_vs_avg_ratio=1.0`). Production deployment should apply a separate new-account policy.
5. **Gemini free tier rate limit:** The LangChain ReAct agent uses Gemini Flash (15 RPM free tier). Production deployment should use a paid API key or Anthropic Claude for higher throughput.

---

*This risk card follows OCC SR 11-7 model risk management guidance. It should be reviewed and signed off by the Model Risk Officer before any production deployment.*
