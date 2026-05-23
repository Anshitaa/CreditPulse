# Fairness & Responsible AI Standards — CreditPulse

## Core Principle
CreditPulse makes decisions that affect people's financial lives. Every model must be audited
for fairness before deployment. Bias in fraud flagging can harm marginalized communities.

## Protected Attributes
The following attributes must NEVER be used as model features, directly or as proxies:
- race, ethnicity, national origin
- gender, gender identity
- religion
- age (unless legally required, e.g., minor account rules)
- disability status
- zip code (strong proxy for race — use region_type: urban/suburban/rural instead)

## Fairness Metrics to Compute (every model, every training run)
1. **Demographic Parity** — P(flag=1 | group=A) ≈ P(flag=1 | group=B)
   - Threshold: delta < 0.05
2. **Equal Opportunity** — True Positive Rate must not differ > 0.05 across groups
3. **Predictive Parity** — Precision must not differ > 0.05 across groups
4. **Individual Fairness** — Similar transactions should receive similar scores

## Fairness Gate (automated via Kiro hook)
- On every model file save: run `governance/fairness_gate.py`
- Gate blocks model promotion if any fairness metric fails threshold
- Override requires HITL sign-off logged to `audit.fairness_overrides`

## Counterfactual Fairness
- Use Dice-ML to verify: changing only protected attributes should NOT change predictions
- Log counterfactual consistency score to MLflow

## Explainability Requirements
- Every high-risk prediction (score > 75) MUST have:
  1. Top 5 SHAP features with human-readable labels
  2. At least 1 counterfactual: "Your score would be < 50 if [action]"
  3. Anchor explanation: minimum set of conditions that make this prediction robust

## Audit Trail
- Every prediction logged to `audit.model_decisions` with: txn_id, model_version, score,
  top_features, timestamp, channel (API/batch)
- Audit logs are immutable (append-only table)
- Retention: 7 years (regulatory requirement)
