"""
CreditPulse — Behavioral Anomaly Detector
Spec: CREDIT-001 FR-003 (composite risk score component)

Isolation Forest trained on normal transaction behavior.
Anomaly score feeds into the composite risk index:
  composite_risk = 0.65 × fraud_prob + 0.35 × anomaly_score

Isolation Forest is unsupervised — no fraud labels needed.
It detects "unusual" transactions relative to the population,
complementing the supervised XGBoost fraud detector.

Usage:
    python models/anomaly.py --train
    python models/anomaly.py --score --txn-id abc-123
"""

import argparse
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import structlog
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger(__name__)

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
EXPERIMENT_NAME = "anomaly_detection"
MODEL_NAME = "anomaly_detector"
MODEL_DIR = Path("models/artifacts")

FEATURE_COLS = [
    "amount",
    "txn_velocity_1h",
    "amount_vs_avg_ratio",
    "hour_of_day",
    "day_of_week",
    "is_foreign_merchant",
]


def load_training_data() -> pd.DataFrame:
    parquet = Path("data/transactions.parquet")
    if parquet.exists():
        df = pd.read_parquet(parquet)
    else:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://creditpulse:creditpulse@localhost:5435/creditpulse")
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT amount, txn_velocity_1h, amount_vs_avg_ratio, hour_of_day, day_of_week, is_foreign_merchant FROM raw.transactions WHERE is_fraud = FALSE LIMIT 500000")
        df = pd.DataFrame(cur.fetchall())
        cur.close()
        conn.close()
    # Train only on NON-fraud transactions (normal behavior baseline)
    df = df[df["is_fraud"] == False].copy() if "is_fraud" in df.columns else df
    df["is_foreign_merchant"] = df["is_foreign_merchant"].astype(int)
    logger.info("loaded_normal_transactions", rows=len(df))
    return df[FEATURE_COLS].dropna()


def train() -> str:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_training_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    params = {
        "n_estimators": 200,
        "max_samples": "auto",
        "contamination": 0.02,   # expected ~2% anomalies
        "max_features": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }

    with mlflow.start_run(run_name="anomaly_detector_isolation_forest") as run:
        mlflow.log_params(params)
        mlflow.log_param("training_rows", len(df))
        mlflow.log_param("features", FEATURE_COLS)

        model = IsolationForest(**params)
        model.fit(X_scaled)

        # Evaluate: anomaly score distribution on training data
        raw_scores = model.score_samples(X_scaled)   # negative — more negative = more anomalous
        normalized = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
        anomaly_rate = (model.predict(X_scaled) == -1).mean()

        mlflow.log_metrics({
            "anomaly_rate_training": float(anomaly_rate),
            "mean_normalized_score": float(normalized.mean()),
            "std_normalized_score": float(normalized.std()),
        })
        logger.info("anomaly_model_trained", anomaly_rate=f"{anomaly_rate:.2%}")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_DIR / "anomaly_detector.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(MODEL_DIR / "anomaly_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        mlflow.sklearn.log_model(model, "anomaly_detector", registered_model_name=MODEL_NAME)
        mlflow.log_artifact(str(MODEL_DIR / "anomaly_detector.pkl"))
        mlflow.log_artifact(str(MODEL_DIR / "anomaly_scaler.pkl"))

        run_id = run.info.run_id
        logger.info("anomaly_training_complete", run_id=run_id)
        return run_id


def load_model() -> tuple[IsolationForest, StandardScaler]:
    with open(MODEL_DIR / "anomaly_detector.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "anomaly_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def score_anomaly(txn: dict) -> dict:
    """Score a transaction for behavioral anomaly. Returns normalized score 0–1 (1 = most anomalous)."""
    model, scaler = load_model()
    features = np.array([[
        txn.get("amount", 100),
        txn.get("txn_velocity_1h", 0),
        txn.get("amount_vs_avg_ratio", 1.0),
        txn.get("hour_of_day", 12),
        txn.get("day_of_week", 0),
        int(txn.get("is_foreign_merchant", False)),
    ]])
    X_scaled = scaler.transform(features)
    raw_score = float(model.score_samples(X_scaled)[0])
    is_anomaly = model.predict(X_scaled)[0] == -1

    # Normalize: Isolation Forest returns negative scores (more negative = more anomalous)
    # Map to 0–1 where 1 = most anomalous
    # Typical range: [-0.5, 0.5] → invert and clip
    normalized = float(np.clip(-raw_score * 2, 0, 1))

    return {
        "txn_id": txn.get("txn_id"),
        "anomaly_score_raw": round(raw_score, 4),
        "anomaly_score_normalized": round(normalized, 4),
        "is_anomaly": bool(is_anomaly),
        "anomaly_contribution_to_risk": round(normalized * 0.35 * 100, 2),  # 35% weight in composite
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()

    if args.train:
        run_id = train()
        print(f"Anomaly model trained. MLflow run_id: {run_id}")
    elif args.score:
        sample = {"txn_id": "test-001", "amount": 9999, "txn_velocity_1h": 12, "amount_vs_avg_ratio": 18.5, "hour_of_day": 3, "day_of_week": 6, "is_foreign_merchant": True}
        import json
        print(json.dumps(score_anomaly(sample), indent=2))


if __name__ == "__main__":
    main()
