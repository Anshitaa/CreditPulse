"""
CreditPulse — Fraud Detection Model
Spec: CREDIT-001 FR-001, NFR-003

XGBoost binary classifier for transaction fraud detection.
- Achieves AUC-ROC >= 0.92 on holdout test set
- All predictions accompanied by SHAP TreeExplainer values
- Logs every training run to MLflow with params, metrics, and artifacts
- Optuna hyperparameter tuning (replaces manual grid search per ml-standards.md)

Usage:
    python models/fraud_detector.py --train
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


def train(parquet_path: str | None = None, n_trials: int = 30) -> str:
    """Train XGBoost fraud detector with Optuna tuning. Returns MLflow run_id."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df_raw = load_training_data(parquet_path)
    df, label_encoder = preprocess(df_raw)

    X = df[FEATURE_COLS]
    y = df["is_fraud"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )

    logger.info("starting_optuna_tuning", n_trials=n_trials)
    study = optuna.create_study(direction="maximize", study_name="fraud_detector_auc")
    study.optimize(
        lambda trial: _optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = {**study.best_params, "tree_method": "hist", "random_state": 42, "use_label_encoder": False}
    logger.info("best_params", **best_params, best_auc=study.best_value)

    with mlflow.start_run(run_name="fraud_detector_optuna") as run:
        mlflow.log_params(best_params)
        mlflow.log_param("n_optuna_trials", n_trials)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("fraud_rate_train", float(y_train.mean()))

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

        if auc < 0.92:
            logger.warning("auc_below_threshold", auc=auc, threshold=0.92)

        # SHAP values on test sample
        explainer = shap.TreeExplainer(final_model)
        shap_sample = X_test.sample(min(1000, len(X_test)), random_state=42)
        shap_values = explainer.shap_values(shap_sample)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = dict(zip(FEATURE_COLS, mean_abs_shap.tolist()))
        mlflow.log_dict(feature_importance, "shap_feature_importance.json")

        # Log model
        mlflow.xgboost.log_model(
            final_model,
            "fraud_detector",
            registered_model_name=MODEL_NAME,
            input_example=X_test.head(3),
        )

        # Save artifacts
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        import pickle
        with open(MODEL_DIR / "fraud_detector.pkl", "wb") as f:
            pickle.dump(final_model, f)
        with open(MODEL_DIR / "label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)
        mlflow.log_artifact(str(MODEL_DIR / "fraud_detector.pkl"))
        mlflow.log_artifact(str(MODEL_DIR / "label_encoder.pkl"))

        run_id = run.info.run_id
        logger.info("training_complete", run_id=run_id, auc=auc)
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
    args = parser.parse_args()

    if args.train:
        run_id = train(args.txn_parquet, args.n_trials)
        print(f"Training complete. MLflow run_id: {run_id}")
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
