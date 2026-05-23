"""
CreditPulse — LangChain Agent Tools
Spec: CREDIT-003 FR-004

Four tools exposed to the ReAct agent:
1. explain_transaction   — SHAP + counterfactual for a txn_id
2. query_risk_scores     — Search high-risk transactions by criteria
3. get_drift_report      — Current PSI drift status
4. get_fairness_metrics  — Latest Fairlearn fairness report
"""

import json
import os
from typing import Optional

import psycopg2
import structlog
from langchain.tools import tool
from psycopg2.extras import RealDictCursor

logger = structlog.get_logger(__name__)

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://creditpulse:creditpulse@localhost:5435/creditpulse",
)


def _db_conn():
    return psycopg2.connect(DATABASE_URL)


@tool
def explain_transaction(txn_id: str) -> str:
    """Look up SHAP explanation and Dice-ML counterfactuals for a transaction ID.
    Use this when asked 'why was transaction X flagged?' or 'how can I reduce the risk for txn X?'
    Input: transaction ID (UUID string).
    Output: JSON with fraud_probability, top SHAP features, and counterfactuals.
    """
    try:
        conn = _db_conn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """SELECT e.txn_id, e.shap_values, e.top_features, e.counterfactuals, e.anchor,
                      r.fraud_probability, r.composite_risk_score, r.decision, r.scored_at
               FROM audit.explanations e
               JOIN mart.risk_scores r ON e.txn_id = r.txn_id
               WHERE e.txn_id = %s
               ORDER BY e.created_at DESC LIMIT 1""",
            (txn_id,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row:
            return f"No explanation found for transaction {txn_id}. It may not have been scored yet."
        return json.dumps(dict(row), default=str, indent=2)
    except Exception as e:
        return f"Error retrieving explanation: {e}"


@tool
def query_risk_scores(criteria: str) -> str:
    """Search the risk score database by natural language criteria.
    Supported criteria:
    - 'high risk' → returns transactions with composite_risk_score > 75
    - 'fraud decisions' → returns transactions flagged as FRAUD
    - 'review queue' → returns transactions in REVIEW status
    - 'account:<account_id>' → returns risk scores for a specific account
    - 'top 10 riskiest today' → top 10 by score in last 24 hours
    Output: JSON array of matching transactions with scores and decisions.
    """
    try:
        conn = _db_conn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        criteria_lower = criteria.lower()

        if "account:" in criteria_lower:
            account_id = criteria.split("account:")[1].strip()
            cur.execute(
                "SELECT * FROM mart.risk_scores WHERE account_id = %s ORDER BY scored_at DESC LIMIT 20",
                (account_id,),
            )
        elif "fraud" in criteria_lower:
            cur.execute(
                "SELECT * FROM mart.risk_scores WHERE decision = 'FRAUD' ORDER BY scored_at DESC LIMIT 20"
            )
        elif "review" in criteria_lower:
            cur.execute(
                "SELECT * FROM mart.risk_scores WHERE decision = 'REVIEW' ORDER BY composite_risk_score DESC LIMIT 20"
            )
        elif "top" in criteria_lower and "riskiest" in criteria_lower:
            cur.execute(
                "SELECT * FROM mart.risk_scores WHERE scored_at > NOW() - INTERVAL '24 hours' ORDER BY composite_risk_score DESC LIMIT 10"
            )
        else:
            # Default: high risk
            cur.execute(
                "SELECT * FROM mart.risk_scores WHERE composite_risk_score > 75 ORDER BY scored_at DESC LIMIT 20"
            )

        rows = cur.fetchall()
        cur.close()
        conn.close()
        if not rows:
            return "No transactions found matching those criteria."
        return json.dumps([dict(r) for r in rows], default=str, indent=2)
    except Exception as e:
        return f"Error querying risk scores: {e}"


@tool
def get_drift_report(feature_name: Optional[str] = None) -> str:
    """Get the latest PSI (Population Stability Index) drift report.
    Call this when asked about model health, data drift, or whether a retrain is needed.
    Optionally filter by feature_name (e.g., 'amount', 'txn_velocity_1h').
    PSI interpretation: < 0.10 = STABLE, 0.10–0.20 = MONITOR, > 0.20 = RETRAIN.
    Output: JSON with PSI scores per feature and recommendation.
    """
    try:
        conn = _db_conn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        if feature_name:
            cur.execute(
                "SELECT * FROM audit.drift_reports WHERE feature_name = %s ORDER BY computed_at DESC LIMIT 5",
                (feature_name,),
            )
        else:
            cur.execute(
                """SELECT DISTINCT ON (feature_name) * FROM audit.drift_reports
                   ORDER BY feature_name, computed_at DESC"""
            )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        if not rows:
            return "No drift reports found. Run governance/drift_monitor.py to generate one."
        result = {
            "drift_summary": [dict(r) for r in rows],
            "overall_recommendation": (
                "RETRAIN_REQUIRED" if any(r["drift_detected"] for r in rows)
                else "NO_ACTION"
            ),
        }
        return json.dumps(result, default=str, indent=2)
    except Exception as e:
        return f"Error retrieving drift report: {e}"


@tool
def get_fairness_metrics(model_name: str = "fraud_detector") -> str:
    """Get the latest Fairlearn fairness metrics for the fraud detection model.
    Call this when asked about bias, fairness, demographic parity, or equal opportunity.
    Reports demographic_parity_difference and equal_opportunity_difference across
    account age groups, region types, and account types.
    Threshold: all metrics should be < 0.05 per fairness-standards.md.
    Output: JSON with per-group fairness metrics and gate_passed status.
    """
    try:
        conn = _db_conn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT * FROM audit.fairness_reports WHERE model_name = %s ORDER BY computed_at DESC LIMIT 1",
            (model_name,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row:
            return f"No fairness report found for {model_name}. Run governance/fairness_gate.py first."
        return json.dumps(dict(row), default=str, indent=2)
    except Exception as e:
        return f"Error retrieving fairness metrics: {e}"
