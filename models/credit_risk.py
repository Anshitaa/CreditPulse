"""
CreditPulse — Credit Risk Scorer
Spec: CREDIT-002 (credit risk scoring)

XGBoost regression model for credit risk scoring (0–100).
Composite formula:
  Credit Risk Index =
    0.50 × XGBoost_credit_risk_prob × 100
  + 0.35 × anomaly_score_normalized × 100
  + 0.15 × bureau_risk_signal × 100   (simulated external signal)

Risk bands:
  0–25:  Very Low   → auto-approve
  26–50: Low        → standard monitoring
  51–75: Medium     → enhanced monitoring
  76–100: High      → manual review required

Usage:
    python models/credit_risk.py --train
    python models/credit_risk.py --score-account acct-001
"""

import argparse
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import shap
import structlog
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = structlog.get_logger(__name__)

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
EXPERIMENT_NAME = "credit_risk"
MODEL_NAME = "credit_risk_scorer"
MODEL_DIR = Path("models/artifacts")

FEATURE_COLS = [
    "amount",
    "txn_velocity_1h",
    "amount_vs_avg_ratio",
    "is_foreign_merchant",
    "merchant_category_encoded",
    "account_age_days",
    "avg_monthly_spend",
    "hour_of_day",
    "day_of_week",
]

FEATURE_DESCRIPTIONS = {
    "amount": "Transaction amount in USD",
    "txn_velocity_1h": "Transactions in past hour",
    "amount_vs_avg_ratio": "Amount vs 90-day account average",
    "is_foreign_merchant": "Foreign merchant flag",
    "merchant_category_encoded": "Merchant risk category",
    "account_age_days": "Account age in days (older = lower risk)",
    "avg_monthly_spend": "Account average monthly spend",
    "hour_of_day": "Hour of transaction",
    "day_of_week": "Day of week",
}

RISK_BANDS = {
    (0, 25): "VERY_LOW",
    (26, 50): "LOW",
    (51, 75): "MEDIUM",
    (76, 100): "HIGH",
}


def _get_risk_band(score: float) -> str:
    for (lo, hi), label in RISK_BANDS.items():
        if lo <= score <= hi:
            return label
    return "UNKNOWN"


def _build_credit_target(df: pd.DataFrame) -> pd.Series:
    """
    Build a synthetic credit risk score target (0–1) from transaction features.
    In production: use actual default/delinquency labels from credit bureau data.
    """
    risk = (
        0.30 * (df["amount_vs_avg_ratio"].clip(0, 20) / 20)
        + 0.25 * (df["txn_velocity_1h"].clip(0, 15) / 15)
        + 0.20 * df["is_foreign_merchant"].astype(float)
        + 0.15 * (1 - (df["account_age_days"].clip(0, 3650) / 3650))
        + 0.10 * np.random.default_rng(42).uniform(0, 1, len(df))  # residual variance
    ).clip(0, 1)
    return risk


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
        cur.execute("""
            SELECT t.amount, t.txn_velocity_1h, t.amount_vs_avg_ratio, t.is_foreign_merchant,
                   t.merchant_category, t.hour_of_day, t.day_of_week,
                   a.age_days AS account_age_days, a.avg_monthly_spend
            FROM raw.transactions t
            JOIN raw.accounts a ON t.account_id = a.account_id
            LIMIT 500000
        """)
        df = pd.DataFrame(cur.fetchall())
        cur.close()
        conn.close()

    df["is_foreign_merchant"] = df["is_foreign_merchant"].astype(int)
    if "account_age_days" not in df.columns:
        df["account_age_days"] = 365
    if "avg_monthly_spend" not in df.columns:
        df["avg_monthly_spend"] = 1000.0

    le = LabelEncoder()
    df["merchant_category_encoded"] = le.fit_transform(df["merchant_category"].fillna("retail"))

    df["credit_risk_target"] = _build_credit_target(df)
    logger.info("loaded_data", rows=len(df), avg_risk=df["credit_risk_target"].mean().round(3))
    return df, le


def train(n_trials: int = 20) -> str:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df, le = load_training_data()
    X = df[FEATURE_COLS]
    y = df["credit_risk_target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "tree_method": "hist",
            "random_state": 42,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = model.predict(X_test)
        return -mean_absolute_error(y_test, preds)   # maximize negative MAE

    study = optuna.create_study(direction="maximize", study_name="credit_risk_mae")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_params = {**study.best_params, "tree_method": "hist", "random_state": 42}

    with mlflow.start_run(run_name="credit_risk_optuna") as run:
        mlflow.log_params(best_params)
        mlflow.log_param("n_trials", n_trials)

        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train, y_train, verbose=False)

        preds = model.predict(X_test).clip(0, 1)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)

        mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})
        logger.info("credit_risk_metrics", mae=round(mae, 4), rmse=round(rmse, 4), r2=round(r2, 4))

        # SHAP
        explainer = shap.TreeExplainer(model)
        shap_sample = X_test.sample(min(500, len(X_test)), random_state=42)
        shap_vals = explainer.shap_values(shap_sample)
        importance = dict(zip(FEATURE_COLS, np.abs(shap_vals).mean(axis=0).tolist()))
        mlflow.log_dict(importance, "shap_feature_importance.json")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_DIR / "credit_risk_scorer.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(MODEL_DIR / "credit_risk_le.pkl", "wb") as f:
            pickle.dump(le, f)

        mlflow.xgboost.log_model(model, "credit_risk_scorer", registered_model_name=MODEL_NAME)
        run_id = run.info.run_id
        logger.info("credit_risk_training_complete", run_id=run_id)
        return run_id


def load_model() -> tuple[xgb.XGBRegressor, LabelEncoder, shap.TreeExplainer]:
    with open(MODEL_DIR / "credit_risk_scorer.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "credit_risk_le.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le, shap.TreeExplainer(model)


def score_credit_risk(txn: dict) -> dict:
    """Compute credit risk score for a transaction/account. Spec: CREDIT-002 FR-001."""
    from models.anomaly import score_anomaly

    model, le, explainer = load_model()

    cat = txn.get("merchant_category", "retail")
    try:
        cat_enc = int(le.transform([cat])[0])
    except ValueError:
        cat_enc = 0

    features = np.array([[
        txn.get("amount", 100),
        txn.get("txn_velocity_1h", 0),
        txn.get("amount_vs_avg_ratio", 1.0),
        int(txn.get("is_foreign_merchant", False)),
        cat_enc,
        txn.get("account_age_days", 365),
        txn.get("avg_monthly_spend", 1000),
        txn.get("hour_of_day", 12),
        txn.get("day_of_week", 0),
    ]])
    feature_df = pd.DataFrame(features, columns=FEATURE_COLS)

    xgb_prob = float(model.predict(feature_df)[0].clip(0, 1))
    anomaly = score_anomaly(txn)
    bureau_signal = float(np.clip(np.random.default_rng(hash(txn.get("account_id", "")) % 2**31).uniform(0, 0.3), 0, 1))

    composite = (0.50 * xgb_prob + 0.35 * anomaly["anomaly_score_normalized"] + 0.15 * bureau_signal)
    credit_risk_score = round(composite * 100, 2)
    risk_band = _get_risk_band(credit_risk_score)

    shap_vals = explainer.shap_values(feature_df)[0]
    top_features = sorted(zip(FEATURE_COLS, shap_vals.tolist()), key=lambda x: abs(x[1]), reverse=True)[:5]

    return {
        "txn_id": txn.get("txn_id"),
        "account_id": txn.get("account_id"),
        "credit_risk_score": credit_risk_score,
        "risk_band": risk_band,
        "components": {
            "xgb_credit_risk_prob": round(xgb_prob, 4),
            "anomaly_score": anomaly["anomaly_score_normalized"],
            "bureau_signal": round(bureau_signal, 4),
            "weights": {"xgb": 0.50, "anomaly": 0.35, "bureau": 0.15},
        },
        "top_features": [
            {"feature": f, "shap_value": round(v, 4), "description": FEATURE_DESCRIPTIONS.get(f, f)}
            for f, v in top_features
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--score-account", type=str)
    parser.add_argument("--n-trials", type=int, default=20)
    args = parser.parse_args()

    if args.train:
        run_id = train(args.n_trials)
        print(f"Credit risk model trained. MLflow run_id: {run_id}")
    elif args.score_account:
        import json
        sample = {"account_id": args.score_account, "amount": 2500, "txn_velocity_1h": 3, "amount_vs_avg_ratio": 4.2, "is_foreign_merchant": False, "merchant_category": "online_retail", "account_age_days": 180, "avg_monthly_spend": 800, "hour_of_day": 10, "day_of_week": 1}
        print(json.dumps(score_credit_risk(sample), indent=2))


if __name__ == "__main__":
    main()
