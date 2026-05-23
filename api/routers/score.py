"""
CreditPulse — /score router
Spec: CREDIT-001 FR-001, NFR-001 (< 100ms)
"""

import os
import time
import uuid
from datetime import datetime
from typing import Optional

import psycopg2
import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

logger = structlog.get_logger(__name__)
router = APIRouter()

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://creditpulse:creditpulse@localhost:5435/creditpulse")


class TransactionRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "txn_id": "abc-123",
            "account_id": "acct-001",
            "merchant_id": "merch-007",
            "amount": 4500.00,
            "merchant_category": "wire_transfer",
            "is_foreign_merchant": True,
            "hour_of_day": 2,
            "day_of_week": 6,
            "txn_velocity_1h": 8,
            "amount_vs_avg_ratio": 12.5,
        }
    })

    txn_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    account_id: str
    merchant_id: Optional[str] = None
    amount: float = Field(gt=0, le=100_000)
    merchant_category: str = "retail"
    is_foreign_merchant: bool = False
    hour_of_day: int = Field(ge=0, le=23, default=12)
    day_of_week: int = Field(ge=0, le=6, default=0)
    txn_velocity_1h: int = Field(ge=0, default=0)
    amount_vs_avg_ratio: float = Field(gt=0, default=1.0)


class ScoreResponse(BaseModel):
    txn_id: str
    fraud_probability: float
    fraud_risk_score: float
    decision: str  # FRAUD | REVIEW | CLEAR
    top_features: list[dict]
    inference_latency_ms: float
    spec_ref: str = "CREDIT-001"
    scored_at: str


@router.post("/", response_model=ScoreResponse, summary="Score a transaction for fraud risk")
async def score_transaction(txn: TransactionRequest, request: Request):
    """
    Score a transaction for fraud. Target latency: p99 < 100ms.

    Spec: CREDIT-001 FR-001 — WHEN a payment transaction event arrives,
    the system SHALL score it for fraud within 100ms.

    Decision thresholds:
    - FRAUD: score > 75 → block transaction
    - REVIEW: score 40–75 → human review required
    - CLEAR: score < 40 → approve
    """
    model = request.app.state.fraud_model
    label_encoder = request.app.state.label_encoder
    shap_explainer = request.app.state.shap_explainer

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run models/fraud_detector.py --train first.")

    from models.fraud_detector import score_transaction as _score

    result = _score(txn.model_dump(), model, label_encoder, shap_explainer)
    result["scored_at"] = datetime.utcnow().isoformat()
    result["spec_ref"] = "CREDIT-001"
    result["account_id"] = txn.account_id

    # Async audit log
    _log_to_audit(result)

    # Broadcast to WebSocket clients
    ws_manager = request.app.state.ws_manager
    await ws_manager.broadcast(result)

    # Spec: CREDIT-001 FR-003 — publish HIGH-RISK alert if score > 75
    if result["fraud_risk_score"] > 75:
        _publish_fraud_alert(result)

    return result


def _log_to_audit(result: dict) -> None:
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        import json
        now = datetime.utcnow()
        # Write to audit log
        cur.execute(
            """INSERT INTO audit.model_decisions
               (txn_id, model_version, score, decision, top_features, decided_at)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT DO NOTHING""",
            (
                result["txn_id"],
                "v1.0",
                result["fraud_risk_score"],
                result["decision"],
                json.dumps(result["top_features"]),
                now,
            ),
        )
        # Also write to mart.risk_scores for dashboard queries
        cur.execute(
            """INSERT INTO mart.risk_scores
               (txn_id, account_id, composite_risk_score, fraud_probability, decision, scored_at)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT (txn_id) DO NOTHING""",
            (
                result["txn_id"],
                result.get("account_id", "unknown"),
                result["fraud_risk_score"],
                result["fraud_probability"],
                result["decision"],
                now,
            ),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning("audit_log_failed", error=str(e))


def _publish_fraud_alert(result: dict) -> None:
    try:
        from confluent_kafka import Producer
        import json
        KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})
        producer.produce(
            topic="alerts.fraud",
            key=result["txn_id"].encode(),
            value=json.dumps(result, default=str).encode(),
        )
        producer.flush(timeout=1)
    except Exception as e:
        logger.warning("kafka_alert_failed", error=str(e))
