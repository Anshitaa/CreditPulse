"""
CreditPulse — Fraud Detection Model
Spec: CREDIT-001 FR-001, NFR-003

XGBoost binary classifier for transaction fraud detection.
- Synthetic training: AUC ~0.68 (intentionally hard synthetic data, documented)
- IEEE-CIS training: AUC ~0.90+ (real Vesta Corporation transaction data)
- All predictions accompanied by SHAP TreeExplainer values
- Logs every training run to MLflow with params, metrics, and artifacts
- Optuna hyperparameter tuning (replaces manual grid search per ml-standards.md)

Usage:
    # Train on synthetic data (default):
    python models/fraud_detector.py --train

    # Train on real IEEE-CIS data (requires data/load_ieee_cis.py --load-db first):
    python models/fraud_detector.py --train --ieee-cis

    # Score a sample transaction:
    python models/fraud_detector.py --score --txn-parquet data/transactions.parquet
"""

import argparse
import os
import time
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import shap
import structlog
import xgboost as xgb
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

logger = structlog.get_logger(__name__)

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
EXPERIMENT_NAME = "fraud_detection"
MODEL_NAME = "fraud_detector"
MODEL_DIR = Path("models/artifacts")

FEATURE_COLS = [
    "amount",
    "hour_of_day",
    "day_of_week",
    "txn_velocity_1h",
    "amount_vs_avg_ratio",
    "is_foreign_merchant",
    "merchant_category_encoded",
]

FEATURE_DESCRIPTIONS = {
    "amount": "Transaction amount in USD",
    "hour_of_day": "Hour of day (0-23) when transaction occurred",
    "day_of_week": "Day of week (0=Monday, 6=Sunday)",
    "txn_velocity_1h": "Number of transactions by this account in the past hour",
    "amount_vs_avg_ratio": "Transaction amount divided by account's 90-day average transaction amount",
    "is_foreign_merchant": "1 if merchant is outside the account's home country",
    "merchant_category_encoded": "Merchant category label-encoded (high-risk categories have higher values)",
}


def load_training_data(parquet_path: str | None = None) -> pd.DataFrame:
    """Load synthetic transaction data from parquet or PostgreSQL."""
    if parquet_path and Path(parquet_path).exists():
        df = pd.read_parquet(parquet_path)
    else:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://creditpulse:creditpulse@localhost:5435/creditpulse")
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT t.txn_id, t.account_id, t.amount, t.merchant_category,
                   t.is_foreign_merchant, t.hour_of_day, t.day_of_week,
                   t.txn_velocity_1h, t.amount_vs_avg_ratio, t.is_fraud
            FROM raw.transactions t
            LIMIT 1000000
        """)
        df = pd.DataFrame(cur.fetchall())
        cur.close()
        conn.close()
    logger.info("loaded_training_data", rows=len(df), fraud_rate=f"{df['is_fraud'].mean():.2%}")
    return df


# ── IEEE-CIS feature set ───────────────────────────────────────────────────

IEEE_FEATURE_COLS = [
    "transaction_amt", "log_amount", "hour_of_day", "day_of_week",
    "email_match", "is_credit",
    "addr1", "addr2", "dist1",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "C11", "C12", "C13", "C14",
    "D1", "D2", "D3", "D4", "D5", "D10", "D11", "D15",
    "M1", "M2", "M3", "M4", "M5", "M6",
]

IEEE_FEATURE_DESCRIPTIONS = {
    "transaction_amt": "Transaction amount in USD (Vesta)",
    "log_amount": "Log(1 + amount) — reduces right-skew",
    "hour_of_day": "Hour of day (0–23) derived from TransactionDT",
    "day_of_week": "Day of week (0–6) derived from TransactionDT",
    "email_match": "1 if purchaser and recipient email domains match",
    "is_credit": "1 if card type is credit (vs debit)",
    "addr1": "Billing region code",
    "addr2": "Billing country code",
    "dist1": "Distance from home address to transaction location",
    "C1": "Count of addresses associated with the payment card",
    "C2": "Count of cards associated with the billing address",
    "D1": "Days since first transaction on this card",
    "D10": "Days since last transaction on this card",
    "M1": "Whether billing address matches the card issuer address",
    "M4": "Vesta match flag 4",
    "M6": "Vesta match flag 6 (high fraud signal)",
}


def load_ieee_cis_training_data() -> tuple[pd.DataFrame, list[str]]:
    """Load IEEE-CIS data from PostgreSQL (loaded by data/load_ieee_cis.py).

    Returns (df, feature_cols) where df has is_fraud label and all feature columns.
    """
    import json as _json
    import psycopg2
    from psycopg2.extras import RealDictCursor

    DATABASE_URL = os.environ.get(
        "DATABASE_URL", "postgresql://creditpulse:creditpulse@localhost:5435/creditpulse"
    )
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Check table exists
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema='raw' AND table_name='ieee_cis_transactions'
        )
    """)
    if not cur.fetchone()["exists"]:
        conn.close()
        raise RuntimeError(
            "raw.ieee_cis_transactions not found.\n"
            "Run: python data/load_ieee_cis.py --download --load-db"
        )

    cur.execute("SELECT COUNT(*) as cnt FROM raw.ieee_cis_transactions")
    count = cur.fetchone()["cnt"]
    logger.info("ieee_cis_table_found", rows=count)

    cur.execute("""
        SELECT transaction_id, is_fraud,
               transaction_amt, log_amount, hour_of_day, day_of_week,
               email_match, is_credit, addr1, addr2, dist1,
               features_json
        FROM raw.ieee_cis_transactions
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    records = []
    for row in rows:
        rec = dict(row)
        raw = rec.pop("features_json") or {}
        # psycopg2 auto-parses JSONB → dict; only call json.loads on str
        blob = raw if isinstance(raw, dict) else _json.loads(raw)
        rec.update(blob)
        records.append(rec)

    df = pd.DataFrame(records)

    # Replace -999 sentinel with NaN so XGBoost handles missings natively
    df.replace(-999, np.nan, inplace=True)

    # Build actual feature list (intersection of desired + available)
    # Include C/D/M/V columns from the JSON blob
    json_cols = [c for c in df.columns if c not in {
        "transaction_id", "is_fraud", "transaction_amt", "log_amount",
        "hour_of_day", "day_of_week", "email_match", "is_credit",
        "addr1", "addr2", "dist1",
    }]
    base_cols = ["transaction_amt", "log_amount", "hour_of_day", "day_of_week",
                 "email_match", "is_credit", "addr1", "addr2", "dist1"]
    all_feature_cols = [c for c in base_cols + json_cols if c in df.columns]

    # Fill remaining NaN with -999 (XGBoost handles internally)
    df[all_feature_cols] = df[all_feature_cols].fillna(-999)
    df["is_fraud"] = df["is_fraud"].astype(int)

    fraud_rate = df["is_fraud"].mean()
    logger.info("loaded_ieee_cis", rows=len(df), fraud_rate=f"{fraud_rate:.2%}",
                features=len(all_feature_cols))
    print(f"IEEE-CIS: {len(df):,} transactions | fraud rate: {fraud_rate:.2%} | features: {len(all_feature_cols)}")
    return df, all_feature_cols


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    le = LabelEncoder()
    df = df.copy()
    df["merchant_category_encoded"] = le.fit_transform(df["merchant_category"].fillna("unknown"))
    df["is_foreign_merchant"] = df["is_foreign_merchant"].astype(int)
    return df[FEATURE_COLS + ["is_fraud"]], le


def _optuna_objective(trial: optuna.Trial, X_train, y_train, X_val, y_val) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 10, 60),
        "tree_method": "hist",
        "eval_metric": "auc",
        "use_label_encoder": False,
        "random_state": 42,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])


def train(
    parquet_path: str | None = None,
    n_trials: int = 30,
    ieee_cis: bool = False,
) -> str:
    """Train XGBoost fraud detector with Optuna tuning. Returns MLflow run_id.

    Args:
        parquet_path: Path to synthetic data parquet (ignored if ieee_cis=True).
        n_trials: Number of Optuna trials.
        ieee_cis: If True, train on real IEEE-CIS data from PostgreSQL.
                  Saves to fraud_detector_ieee.pkl (separate from synthetic model).
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    label_encoder = None  # only used for synthetic categorical encoding

    if ieee_cis:
        df, feature_cols = load_ieee_cis_training_data()
        X = df[feature_cols]
        y = df["is_fraud"]
        model_artifact_name = "fraud_detector_ieee"
        model_pkl_name = "fraud_detector_ieee.pkl"
        run_name = "fraud_detector_ieee_cis_optuna"
        print(f"\nTraining on IEEE-CIS real data — expect AUC 0.88–0.93")
    else:
        df_raw = load_training_data(parquet_path)
        df, label_encoder = preprocess(df_raw)
        feature_cols = FEATURE_COLS
        X = df[FEATURE_COLS]
        y = df["is_fraud"].astype(int)
        model_artifact_name = "fraud_detector"
        model_pkl_name = "fraud_detector.pkl"
        run_name = "fraud_detector_optuna"
        print(f"\nTraining on synthetic data — AUC will be lower (~0.65–0.72) by design")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )

    # Optuna tuning uses the same objective for both data sources
    # For IEEE-CIS, scale_pos_weight is lower (3.5% fraud rate vs 12.5% synthetic)
    pos_weight_range = (5, 30) if ieee_cis else (10, 60)

    logger.info("starting_optuna_tuning", n_trials=n_trials, data_source="ieee_cis" if ieee_cis else "synthetic")
    study = optuna.create_study(
        direction="maximize",
        study_name=f"fraud_detector_{'ieee' if ieee_cis else 'synthetic'}_auc",
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight", pos_weight_range[0], pos_weight_range[1]
            ),
            "tree_method": "hist",
            "eval_metric": "auc",
            "use_label_encoder": False,
            "random_state": 42,
            "enable_categorical": False,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = {
        **study.best_params,
        "tree_method": "hist",
        "random_state": 42,
        "use_label_encoder": False,
        "enable_categorical": False,
    }
    logger.info("best_params", **best_params, best_auc=study.best_value)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(best_params)
        mlflow.log_param("n_optuna_trials", n_trials)
        mlflow.log_param("data_source", "ieee_cis" if ieee_cis else "synthetic")
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("fraud_rate_train", float(y_train.mean()))
        mlflow.log_param("n_features", len(feature_cols))

        # Final training on train+val with best params
        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(
            pd.concat([X_train, X_val]),
            pd.concat([y_train, y_val]),
            verbose=False,
        )

        # Evaluation
        y_pred_proba = final_model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        auc = roc_auc_score(y_test, y_pred_proba)
        ap = average_precision_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metrics({"auc_roc": auc, "avg_precision": ap, "f1_score": f1})
        logger.info("model_metrics", auc_roc=auc, avg_precision=ap, f1_score=f1)
        print(f"\n{'='*50}")
        print(f"AUC-ROC:         {auc:.4f}")
        print(f"Avg Precision:   {ap:.4f}")
        print(f"F1 Score:        {f1:.4f}")
        print(f"Data source:     {'IEEE-CIS (real)' if ieee_cis else 'Synthetic'}")
        print(f"{'='*50}\n")

        if not ieee_cis and auc < 0.70:
            logger.warning("synthetic_auc_low", auc=auc,
                           note="Expected ~0.65-0.72 for synthetic data — this is intentional")
        if ieee_cis and auc < 0.85:
            logger.warning("ieee_auc_below_expected", auc=auc,
                           note="IEEE-CIS should reach 0.88+ with enough Optuna trials")

        # SHAP values on test sample
        explainer = shap.TreeExplainer(final_model)
        shap_sample = X_test.sample(min(1000, len(X_test)), random_state=42)
        shap_values = explainer.shap_values(shap_sample)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = dict(zip(feature_cols, mean_abs_shap.tolist()))
        mlflow.log_dict(feature_importance, "shap_feature_importance.json")

        # Log model
        mlflow.xgboost.log_model(
            final_model,
            model_artifact_name,
            registered_model_name=model_artifact_name,
            input_example=X_test.head(3),
        )

        # Save artifacts
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        import pickle
        with open(MODEL_DIR / model_pkl_name, "wb") as f:
            pickle.dump(final_model, f)

        # Save feature column list alongside the model
        feature_meta = {
            "feature_cols": feature_cols,
            "data_source": "ieee_cis" if ieee_cis else "synthetic",
            "auc_roc": auc,
        }
        with open(MODEL_DIR / model_pkl_name.replace(".pkl", "_meta.json"), "w") as f:
            import json
            json.dump(feature_meta, f, indent=2)

        if label_encoder is not None:
            with open(MODEL_DIR / "label_encoder.pkl", "wb") as f:
                pickle.dump(label_encoder, f)
            mlflow.log_artifact(str(MODEL_DIR / "label_encoder.pkl"))

        mlflow.log_artifact(str(MODEL_DIR / model_pkl_name))

        run_id = run.info.run_id
        logger.info("training_complete", run_id=run_id, auc=auc,
                    model_file=str(MODEL_DIR / model_pkl_name))
        return run_id


def load_model() -> tuple[xgb.XGBClassifier, LabelEncoder, shap.TreeExplainer]:
    import pickle
    with open(MODEL_DIR / "fraud_detector.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    explainer = shap.TreeExplainer(model)
    return model, label_encoder, explainer


def score_transaction(
    txn: dict,
    model: xgb.XGBClassifier,
    label_encoder: LabelEncoder,
    explainer: shap.TreeExplainer,
) -> dict:
    """Score a single transaction. Returns score + SHAP explanation. Target: < 100ms."""
    t0 = time.perf_counter()

    category = txn.get("merchant_category", "unknown")
    try:
        cat_encoded = label_encoder.transform([category])[0]
    except ValueError:
        cat_encoded = 0

    features = np.array([[
        txn.get("amount", 0),
        txn.get("hour_of_day", 12),
        txn.get("day_of_week", 0),
        txn.get("txn_velocity_1h", 0),
        txn.get("amount_vs_avg_ratio", 1.0),
        int(txn.get("is_foreign_merchant", False)),
        cat_encoded,
    ]])
    feature_df = pd.DataFrame(features, columns=FEATURE_COLS)

    fraud_prob = float(model.predict_proba(feature_df)[0, 1])
    shap_vals = explainer.shap_values(feature_df)[0]
    top_features = sorted(
        zip(FEATURE_COLS, shap_vals.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:5]

    latency_ms = (time.perf_counter() - t0) * 1000
    if latency_ms > 100:
        logger.warning("slow_inference", latency_ms=round(latency_ms, 2), txn_id=txn.get("txn_id"))

    risk_score = fraud_prob * 100
    if risk_score >= 75:
        decision = "FRAUD"
    elif risk_score >= 40:
        decision = "REVIEW"
    else:
        decision = "CLEAR"

    return {
        "txn_id": txn.get("txn_id"),
        "fraud_probability": round(fraud_prob, 4),
        "fraud_risk_score": round(risk_score, 2),
        "decision": decision,
        "top_features": [
            {
                "feature": f,
                "shap_value": round(v, 4),
                "description": FEATURE_DESCRIPTIONS.get(f, f),
                "direction": "increases_risk" if v > 0 else "decreases_risk",
            }
            for f, v in top_features
        ],
        "inference_latency_ms": round(latency_ms, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--txn-parquet", type=str)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument(
        "--ieee-cis", action="store_true",
        help="Train on real IEEE-CIS data (run data/load_ieee_cis.py --load-db first)",
    )
    args = parser.parse_args()

    if args.train:
        run_id = train(args.txn_parquet, args.n_trials, ieee_cis=args.ieee_cis)
        suffix = " (IEEE-CIS real data)" if args.ieee_cis else " (synthetic data)"
        print(f"Training complete{suffix}. MLflow run_id: {run_id}")
    elif args.score:
        model, le, explainer = load_model()
        # Score a sample transaction
        sample_txn = {
            "txn_id": "sample-001",
            "amount": 4500.0,
            "merchant_category": "wire_transfer",
            "is_foreign_merchant": True,
            "hour_of_day": 2,
            "day_of_week": 6,
            "txn_velocity_1h": 8,
            "amount_vs_avg_ratio": 12.5,
        }
        result = score_transaction(sample_txn, model, le, explainer)
        import json
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
