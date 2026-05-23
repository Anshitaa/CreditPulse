"""
CreditPulse — /risk router
Spec: CREDIT-002
"""

import os
from typing import Optional

import psycopg2
import structlog
from fastapi import APIRouter, HTTPException, Query
from psycopg2.extras import RealDictCursor

logger = structlog.get_logger(__name__)
router = APIRouter()

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://creditpulse:creditpulse@localhost:5435/creditpulse")


@router.get("/scores", summary="Query risk scores with filters")
async def get_risk_scores(
    min_score: float = Query(0, ge=0, le=100),
    max_score: float = Query(100, ge=0, le=100),
    decision: Optional[str] = Query(None, pattern="^(FRAUD|REVIEW|CLEAR)$"),
    hours_back: int = Query(24, ge=1, le=168),
    limit: int = Query(50, ge=1, le=500),
):
    """Paginated risk score queries. Filter by score range, decision, or time window."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        conditions = [
            "composite_risk_score BETWEEN %s AND %s",
            "scored_at > NOW() - INTERVAL '%s hours'",
        ]
        params = [min_score, max_score, hours_back]
        if decision:
            conditions.append("decision = %s")
            params.append(decision)
        where = " AND ".join(conditions)
        cur.execute(
            f"SELECT * FROM mart.risk_scores WHERE {where} ORDER BY composite_risk_score DESC LIMIT %s",
            params + [limit],
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return {"count": len(rows), "scores": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/account/{account_id}", summary="Get risk history for an account")
async def get_account_risk(account_id: str, weeks: int = Query(12, ge=1, le=52)):
    """
    Returns weekly risk score trend for an account (last N weeks).
    Spec: CREDIT-002 FR-005 — return score trend for last 12 weeks.
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """SELECT date_trunc('week', scored_at) AS week,
                      AVG(composite_risk_score) AS avg_score,
                      MAX(composite_risk_score) AS max_score,
                      COUNT(*) AS txn_count,
                      SUM(CASE WHEN decision = 'FRAUD' THEN 1 ELSE 0 END) AS fraud_count
               FROM mart.risk_scores
               WHERE account_id = %s AND scored_at > NOW() - INTERVAL '%s weeks'
               GROUP BY week ORDER BY week DESC""",
            (account_id, weeks),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        if not rows:
            raise HTTPException(status_code=404, detail=f"No risk history for account {account_id}")
        return {"account_id": account_id, "weeks": weeks, "trend": [dict(r) for r in rows]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary", summary="Get high-level risk summary stats")
async def get_risk_summary():
    """Dashboard summary: fraud rate, avg score, decision distribution today."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT
                COUNT(*) AS total_scored_today,
                ROUND(AVG(composite_risk_score)::numeric, 2) AS avg_risk_score,
                SUM(CASE WHEN decision = 'FRAUD' THEN 1 ELSE 0 END) AS fraud_count,
                SUM(CASE WHEN decision = 'REVIEW' THEN 1 ELSE 0 END) AS review_count,
                SUM(CASE WHEN decision = 'CLEAR' THEN 1 ELSE 0 END) AS clear_count,
                ROUND((SUM(CASE WHEN decision = 'FRAUD' THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) * 100)::numeric, 2) AS fraud_rate_pct
            FROM mart.risk_scores
            WHERE scored_at > NOW() - INTERVAL '24 hours'
        """)
        row = cur.fetchone()
        cur.close()
        conn.close()
        return dict(row) if row else {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
