"""
CreditPulse — /explain router
Spec: CREDIT-003 FR-001, FR-002, FR-003
"""

import os
from typing import Optional

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

logger = structlog.get_logger(__name__)
router = APIRouter()


class WhatIfRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "account_id": "acct-001",
            "amount": 847.0,
            "merchant_category": "grocery",
            "is_foreign_merchant": False,
            "hour_of_day": 14,
            "day_of_week": 2,
            "txn_velocity_1h": 1,
            "amount_vs_avg_ratio": 1.2,
        }
    })
    account_id: str
    amount: float
    merchant_category: str = "retail"
    is_foreign_merchant: bool = False
    hour_of_day: int = 12
    day_of_week: int = 0
    txn_velocity_1h: int = 0
    amount_vs_avg_ratio: float = 1.0


@router.get("/{txn_id}", summary="Get SHAP explanation + counterfactuals for a transaction")
async def get_explanation(txn_id: str):
    """
    Retrieve stored SHAP values and Dice-ML counterfactuals for a scored transaction.
    Spec: CREDIT-003 FR-001, FR-002 — WHERE score > 50, the system SHALL provide
    SHAP top-5 features and counterfactual: 'Your score would be < 50 if [action].'
    """
    import json
    import psycopg2
    from psycopg2.extras import RealDictCursor

    DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://creditpulse:creditpulse@localhost:5435/creditpulse")
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """SELECT e.*, r.fraud_probability, r.composite_risk_score, r.decision
               FROM audit.explanations e
               LEFT JOIN mart.risk_scores r ON e.txn_id = r.txn_id
               WHERE e.txn_id = %s ORDER BY e.created_at DESC LIMIT 1""",
            (txn_id,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row:
            raise HTTPException(status_code=404, detail=f"No explanation found for {txn_id}")
        return dict(row)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/counterfactual", summary="Generate Dice-ML counterfactuals for a transaction")
async def generate_counterfactual(txn: WhatIfRequest, request: Request):
    """
    Generate Dice-ML counterfactual explanations.
    Returns: 'To reduce risk below 40, change amount to $X OR change merchant to grocery.'
    Spec: CREDIT-003 FR-002 — WHERE score > 75, generate minimum change to reduce score < 50.
    Target: p95 < 1000ms.
    """
    cf_engine = request.app.state.cf_engine
    result = cf_engine.explain(txn.model_dump())
    return result


@router.post("/whatif", summary="What-if simulator: score a hypothetical transaction")
async def what_if(txn: WhatIfRequest, request: Request):
    """
    Interactive what-if simulator. Change transaction attributes and see how
    the fraud score changes in real time. Used by the React frontend simulator.
    Spec: CREDIT-003 FR-003 — WHEN user submits what-if query, respond within 1 second.
    """
    model = request.app.state.fraud_model
    label_encoder = request.app.state.label_encoder
    shap_explainer = request.app.state.shap_explainer

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    from models.fraud_detector import score_transaction as _score
    txn_dict = txn.model_dump()
    txn_dict["txn_id"] = "whatif-simulation"
    result = _score(txn_dict, model, label_encoder, shap_explainer)
    result["mode"] = "what_if_simulation"
    return result
