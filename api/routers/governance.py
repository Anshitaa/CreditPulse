"""
CreditPulse — /governance router
Spec: CREDIT-002 FR-004, CREDIT-003
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


@router.get("/drift", summary="Get latest PSI drift report")
async def get_drift_report(feature: Optional[str] = Query(None)):
    """Latest PSI drift report. PSI > 0.20 = retrain required. Spec: CREDIT-002 FR-004."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        if feature:
            cur.execute(
                "SELECT * FROM audit.drift_reports WHERE feature_name = %s ORDER BY computed_at DESC LIMIT 5",
                (feature,),
            )
        else:
            cur.execute("""
                SELECT DISTINCT ON (feature_name) * FROM audit.drift_reports
                ORDER BY feature_name, computed_at DESC
            """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        drift_detected = any(r["drift_detected"] for r in rows)
        return {
            "drift_detected": drift_detected,
            "recommendation": "RETRAIN_REQUIRED" if drift_detected else "NO_ACTION",
            "features": [dict(r) for r in rows],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fairness", summary="Get latest fairness metrics")
async def get_fairness_metrics(model_name: str = Query("fraud_detector")):
    """Latest Fairlearn fairness report. Gate threshold: all metrics < 0.05."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT * FROM audit.fairness_reports WHERE model_name = %s ORDER BY computed_at DESC LIMIT 1",
            (model_name,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row:
            raise HTTPException(status_code=404, detail=f"No fairness report for {model_name}")
        return dict(row)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", summary="List registered MLflow models and stages")
async def list_models():
    """List all registered models from MLflow Model Registry."""
    try:
        import mlflow
        MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
        mlflow.set_tracking_uri(MLFLOW_URI)
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        models = client.search_registered_models()
        result = []
        for m in models:
            versions = client.search_model_versions(f"name='{m.name}'")
            result.append({
                "name": m.name,
                "versions": [{"version": v.version, "stage": v.current_stage, "run_id": v.run_id} for v in versions],
            })
        return {"models": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MLflow error: {e}")


@router.post("/drift/run", summary="Trigger PSI drift check on demand")
async def trigger_drift_check():
    """Trigger a PSI drift check immediately (normally runs via Airflow weekly)."""
    try:
        from governance.drift_monitor import DriftMonitor
        monitor = DriftMonitor()
        baseline, current = monitor.load_data(source="synthetic")
        report = monitor.compute_all_psi(baseline, current)
        monitor.log_to_db(report)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fairness/run", summary="Trigger fairness gate check on demand")
async def trigger_fairness_check():
    """Trigger a Fairlearn fairness evaluation immediately."""
    try:
        import numpy as np
        import pandas as pd
        from governance.fairness_gate import FairnessGate
        gate = FairnessGate()
        rng = np.random.default_rng(42)
        n = 2000
        df = pd.DataFrame({
            "amount": rng.lognormal(4.5, 1.0, n),
            "hour_of_day": rng.integers(0, 24, n),
            "day_of_week": rng.integers(0, 7, n),
            "txn_velocity_1h": rng.integers(0, 10, n),
            "amount_vs_avg_ratio": rng.lognormal(0, 0.8, n),
            "is_foreign_merchant": rng.integers(0, 2, n).astype(bool),
            "merchant_category": rng.choice(["grocery", "retail", "wire_transfer", "gambling"], n),
            "is_fraud": rng.integers(0, 2, n),
        })
        passed, report = gate.run_gate(df)
        return {"gate_passed": passed, "report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
