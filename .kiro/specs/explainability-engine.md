# Spec: Explainability Engine

**Spec ID:** CREDIT-003  
**Status:** Approved  
**Author:** Anshita Bhardwaj  
**Created:** 2026-04-26  

---

## Requirements (EARS Notation)

**FR-001** — WHEN any prediction has a risk score > 50, the system SHALL compute and store:
  (a) SHAP values for all features (TreeExplainer for XGBoost)
  (b) Top-5 feature contributions in human-readable format
  (c) Global feature importance rank of each contributing feature

**FR-002** — WHERE a prediction score exceeds 75 (HIGH risk), the system SHALL additionally:
  (a) Generate a Dice-ML counterfactual: minimum change to reduce score below 50
  (b) Generate an anchor explanation: minimum set of conditions sufficient for this prediction
  (c) Find the 3 most similar low-risk transactions (prototypes)

**FR-003** — WHEN a user submits a "what-if" query via the frontend simulator, the system
SHALL respond within 1 second with the score change and updated top features for the
hypothetical transaction.

**FR-004** — WHERE the LangChain agent is asked "why was transaction X flagged?", the agent
SHALL retrieve the stored SHAP + counterfactual explanation and return a natural language
summary within 3 seconds.

### Design

#### Explanation Pipeline
```
Prediction → SHAP TreeExplainer → {shap_values, top_5_features}
           → Dice-ML (if score > 75) → {counterfactuals[]}
           → AnchorExplainer → {anchor_condition, precision, coverage}
           → Store in PostgreSQL [audit.explanations]
```

#### Counterfactual Output Format
```json
{
  "original_score": 88.4,
  "counterfactuals": [
    {
      "target_score": 42.1,
      "changes": [
        {"feature": "txn_amount", "original": 4500, "counterfactual": 847},
        {"feature": "merchant_category", "original": "wire_transfer", "counterfactual": "grocery"}
      ],
      "feasibility": "actionable"
    }
  ]
}
```

### Acceptance Criteria
- [ ] SHAP computed for every prediction where score > 50
- [ ] Counterfactual generated in < 1s for p95 of requests
- [ ] Anchor explanations have precision ≥ 0.80
- [ ] What-if simulator responds in < 1s
- [ ] Agent can retrieve and narrate any explanation by txn_id
