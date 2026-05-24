# CreditPulse — Model Risk Card

**Version:** 1.1.0  
**Generated:** 2026-05-23  
**Author:** Anshita Bhardwaj  
**Scored population:** 590,540 real transactions — IEEE-CIS / Vesta Corporation (3.50% fraud prevalence)  
**Spec linkage:** CREDIT-001 (fraud scoring), CREDIT-002 (credit risk), CREDIT-003 (explainability)

---

## 1. Model Suite Overview

CreditPulse deploys three models in a layered risk pipeline:

| Model | Algorithm | Purpose | Latency Target |
|---|---|---|---|
| `fraud_detector_ieee` | XGBoost (gradient boosted trees) | Primary fraud classification — trained on real data | p99 < 100ms |
| `fraud_detector` | XGBoost | Synthetic baseline (AUC 0.681) — kept for comparison | p99 < 100ms |
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

### 2a. Production Model — IEEE-CIS (Real Data)

| Attribute | Value |
|---|---|
| Source | IEEE-CIS Fraud Detection Competition — Vesta Corporation real transaction data |
| Total transactions | 590,540 |
| Fraud transactions | 20,663 (3.50%) |
| Identity records joined | 144,233 |
| Features used | 318 (transaction + identity + Vesta-engineered V/C/D/M columns) |
| Train / val / test split | 72% / 13% / 15% (stratified by fraud label) |
| Optuna HPO trials | 30 |
| Best trial AUC (validation) | 0.9651 (trial 15) |

### 2b. Synthetic Baseline (kept for comparison)

| Attribute | Value |
|---|---|
| Generator | `data/synthetic_transactions.py` (deterministic seed 42) |
| Total transactions | 150,000 |
| Fraud rate | 12.5% |
| AUC-ROC | 0.681 |

> **Why both models exist:** The synthetic model demonstrates the architecture end-to-end and establishes a performance floor. The IEEE-CIS model validates that the same XGBoost + Optuna pipeline achieves industry-standard AUC (0.96+) on real data. Proactively stating both AUC values in interviews shows evaluative honesty, not weakness.

---

## 3. Performance Metrics — IEEE-CIS Model (holdout test set, 15%)

### 3a. Fraud Detector — IEEE-CIS (XGBoost, 590K real transactions)

| Metric | Value | Notes |
|---|---|---|
| **ROC-AUC** | **0.9686** | Real Vesta transaction data, 318 features |
| **Average Precision** | **0.8414** | PR-AUC; reflects 3.5% class imbalance |
| **F1 Score** | **0.7998** | At default 0.5 threshold |
| Best Optuna val AUC | 0.9651 | Trial 15 / 30 |
| Decision threshold | 0.75 (score > 75 = FRAUD) | Tuned for precision at high-risk band |
| Training time | ~40 min (30 Optuna trials, M2 Mac) | |
| MLflow run ID | `4f574ec80d9c436eac35eaad720a2278` | |

**Best hyperparameters (Optuna trial 15):**
```
n_estimators:       702
max_depth:          8
learning_rate:      0.185
subsample:          0.900
colsample_bytree:   0.731
min_child_weight:   7
scale_pos_weight:   10.49   ← handles 3.5% fraud imbalance
reg_alpha:          7.4e-6
reg_lambda:         0.0155
```

### 3b. Synthetic Baseline (for comparison)

| Metric | Value |
|---|---|
| ROC-AUC | 0.681 |
| Average Precision | 0.220 |
| F1 Score | 0.224 |

> **Interview talking point:** The 0.968 vs 0.681 gap is almost entirely explained by feature richness — 318 real Vesta-engineered features (velocity counts, timedeltas, device fingerprints) vs 7 synthetic features. AUC improvement from feature engineering, not model complexity.

### 3c. Credit Risk Scorer (XGBoost Regressor — synthetic)

| Metric | Value |
|---|---|
| R² | 0.964 |
| MAE | 0.025 |
| RMSE | 0.029 |

### 3d. Anomaly Detector (Isolation Forest)

| Metric | Value |
|---|---|
| Anomaly rate (training) | 2.0% |
| Mean normalized score | 0.796 |
| Std normalized score | 0.162 |

---

## 4. Decision Band Distribution

| Decision | Threshold | Expected production rate |
|---|---|---|
| FRAUD | score > 75 | ~3–5% (matches 3.5% fraud prevalence) |
| REVIEW | score 40–75 | ~10–15% (borderline cases) |
| CLEAR | score < 40 | ~80–85% |

---

## 5. Fairness Audit (Fairlearn) — Synthetic Model

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

> **Note on IEEE-CIS fairness:** The IEEE-CIS dataset does not include explicit demographic attributes. The fairness audit above applies to the synthetic model only. For a production deployment of the IEEE-CIS model, fairness would be audited using proxy variables (card network, billing region, email domain) pending collection of actual protected-class data per ECOA requirements.

**Recommended action:** Collect more labeled data for accounts aged < 90 days, or apply isotonic recalibration per account-age group before promoting to production.

---

## 6. Explainability

Every scored transaction includes:
- **SHAP top-5 features** with direction (`increases_risk` / `decreases_risk`)
- **Dice-ML counterfactual:** "Your score would be < 50 if you changed amount to $X or merchant to grocery"
- Stored in `audit.explanations` for regulatory audit trail

**Top features by mean |SHAP| — IEEE-CIS model (318 features, 1,000-sample test subset):**

| Rank | Feature | Type | Direction |
|---|---|---|---|
| 1 | `V258`, `V257`, `V201` | Vesta velocity/count features | Increases risk at high values |
| 2 | `TransactionAmt` / `log_amount` | Transaction amount | Increases risk at extremes |
| 3 | `C1`, `C2`, `C14` | Card association counts | Increases risk when high |
| 4 | `D1`, `D10` | Days since first/last transaction | Increases risk for new cards |
| 5 | `M6`, `M4` | Vesta match flags | Increases risk when unmatched |

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

> End-to-end request latency (including DB audit write + Kafka publish) is higher (~50–200ms) but model inference consistently meets the < 100ms SLA.

**WebSocket load test results (50 concurrent clients, 60 seconds):**

| Metric | Target | Status |
|---|---|---|
| POST /score p99 | < 100ms | Run `locust -f tests/load/locustfile_ws.py` to verify |
| WS broadcast p95 | < 200ms | Tested via `tests/load/locustfile_ws.py` |
| Concurrent WS connections | 50 | No message loss observed |

---

## 9. Operational Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Model staleness (concept drift) | Medium | High | Weekly PSI monitoring + auto-retrain trigger |
| Fairness regression post-retrain | Low | High | Fairlearn gate enforced as Kiro hook on every model save |
| SHAP explanation instability | Low | Medium | SHAP values locked to model version; versioned in MLflow |
| False negative fraud (miss) | Medium | High | Anomaly detector provides independent signal; REVIEW band for human review |
| Adversarial feature manipulation | Low | Medium | Velocity features computed server-side; client cannot spoof |
| V-feature opacity | Medium | Low | Top-5 SHAP reported; Vesta features documented as black-box inputs |

---

## 10. Governance & Compliance Lineage

| Artifact | Location |
|---|---|
| IEEE-CIS model | `models/artifacts/fraud_detector_ieee.pkl` |
| Synthetic baseline | `models/artifacts/fraud_detector.pkl` |
| Training experiments | `mlruns/` (MLflow, run ID: `4f574ec80d9c436eac35eaad720a2278`) |
| Fairness reports | `audit.fairness_reports` (PostgreSQL) |
| Drift reports | `audit.drift_reports` (PostgreSQL) |
| Scoring decisions | `audit.model_decisions` (append-only) |
| SHAP explanations | `audit.explanations` (per-transaction) |
| Kiro specs | `.kiro/specs/` (EARS requirements) |
| Kiro hooks | `.kiro/hooks/` (fairness-gate, psi-check, auto-test) |
| This document | `docs/model_risk_card_v1.0.0.md` |

**Retraining process:**
1. Drift monitor exceeds PSI threshold → Airflow triggers feature pipeline
2. `python data/load_ieee_cis.py --load-db` — refresh from source
3. `python models/fraud_detector.py --train --ieee-cis --n-trials 30`
4. Fairness gate hook fires automatically on model file save (Kiro)
5. If gate passes → MLflow registers new version as "Staging"
6. Human sign-off required to promote to "Production" (HITL)

---

## 11. Limitations & Known Gaps

1. **IEEE-CIS V-features are opaque:** The top SHAP features (`V258`, `V201`, etc.) are Vesta-proprietary engineered features. Their exact construction is not publicly documented. In production, the bank would have access to equivalent proprietary features from their own transaction history.
2. **No demographic fairness on IEEE-CIS:** Real protected-class attributes (age, race, gender) are absent from the dataset. Proxy-based fairness (billing region, card network) has been architecturally implemented and is ready to be swapped in.
3. **Synthetic fairness gate (0.055 > 0.05):** `account_age_group` predictive parity slightly exceeds threshold on the synthetic model. Documented as a known artifact of fewer velocity features for new accounts. Recalibration recommended before production.
4. **K8s on EKS:** Manifests written and tested on minikube. Not deployed to AWS EKS. Interview answer: *"Architected for EKS; local demo on minikube."*
5. **Gemini free-tier rate limit:** Agent uses Gemini Flash (15 RPM). Production should use a paid key or Claude API.

---

*This risk card follows OCC SR 11-7 model risk management guidance. It should be reviewed and signed off by the Model Risk Officer before any production deployment.*
