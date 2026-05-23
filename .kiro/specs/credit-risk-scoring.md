# Spec: Credit Risk Scoring

**Spec ID:** CREDIT-002  
**Status:** Approved  
**Author:** Anshita Bhardwaj  
**Created:** 2026-04-26  

---

## Requirements (EARS Notation)

**FR-001** — WHEN an account is created or a credit limit change is requested, the system
SHALL compute a credit risk score (0–100) within 2 seconds.

**FR-002** — WHERE the credit risk score exceeds 75, the system SHALL flag the account for
manual review and generate a full explanation report including SHAP values and counterfactuals.

**FR-003** — WHILE computing credit risk, the system SHALL combine three signals:
  (a) XGBoost credit risk probability (weight: 0.50)
  (b) Behavioral anomaly score from Isolation Forest (weight: 0.35)
  (c) External credit bureau signal (simulated) (weight: 0.15)

**FR-004** — IF a credit risk score changes by more than 15 points between weekly recalculations,
the system SHALL trigger a drift alert and log to `audit.drift_reports`.

**FR-005** — WHEN an account manager queries `/credit/{account_id}`, the system SHALL return
the current risk score, trend (last 12 weeks), top risk drivers, and recommended actions.

### Non-Functional Requirements
**NFR-001** — Credit risk scoring batch job must process all accounts in < 4 hours.
**NFR-002** — Credit risk model AUC ≥ 0.88.
**NFR-003** — Score stability: same account queried twice in 1 hour should return the same score.

---

## Risk Score Formula
```
Credit Risk Index (0–100) =
    0.50 × XGBoost_credit_risk_prob × 100
  + 0.35 × normalized_anomaly_score × 100
  + 0.15 × bureau_risk_signal × 100

Bands: 0–25 Very Low | 26–50 Low | 51–75 Medium | 76–100 High
```

### Acceptance Criteria
- [ ] Batch scoring completes all accounts in < 4 hours
- [ ] Credit risk model AUC ≥ 0.88 on holdout test set
- [ ] Score bands are calibrated: actual default rates match expected rates per band
- [ ] Weekly drift check runs automatically via Airflow DAG
- [ ] Score trend available via API for last 12 weeks
