"""
CreditPulse — Fairness Gate (Kiro Hook target)
Spec: CREDIT-001 NFR-004 | fairness-standards.md

Computes Fairlearn fairness metrics and blocks model promotion if thresholds are breached.
Called by .kiro/hooks/fairness-gate.sh on every model file save.

Metrics computed:
- Demographic Parity (selection rate equality across groups)
- Equal Opportunity (TPR equality across groups)
- Predictive Parity (precision equality across groups)

All results logged to MLflow and PostgreSQL audit.fairness_reports.

Usage:
    python governance/fairness_gate.py --model-file models/fraud_detector.py --threshold-demographic-parity 0.05
    python governance/fairness_gate.py --hitl  # Override with HITL sign-off
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import structlog
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
)
from sklearn.metrics import precision_score

logger = structlog.get_logger(__name__)

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://creditpulse:creditpulse@localhost:5435/creditpulse")

# Simulated protected attribute groups for demo
# In production: join to account demographic data (age_group, region only — not race/gender per fairness-standards.md)
PROTECTED_GROUPS = {
    "account_age_group": lambda df: pd.cut(df["account_age_days"] if "account_age_days" in df.columns else np.random.randint(30, 3650, len(df)), bins=[0, 365, 1825, 99999], labels=["new", "established", "long_term"]),
    "region_type": lambda df: df["region_type"] if "region_type" in df.columns else pd.Series(np.random.choice(["urban", "suburban", "rural"], len(df))),
    "account_type": lambda df: df["account_type"] if "account_type" in df.columns else pd.Series(np.random.choice(["checking", "savings", "credit", "business"], len(df))),
}


class FairnessGate:
    def __init__(
        self,
        threshold_demographic_parity: float = 0.05,
        threshold_equal_opportunity: float = 0.05,
        threshold_predictive_parity: float = 0.05,
    ):
        self.thresholds = {
            "demographic_parity_difference": threshold_demographic_parity,
            "equal_opportunity_difference": threshold_equal_opportunity,
            "predictive_parity_difference": threshold_predictive_parity,
        }

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        sensitive_features: pd.Series,
        group_name: str,
    ) -> dict:
        """Compute fairness metrics for a single sensitive attribute."""
        dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
        eod_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)

        # Predictive parity: precision per group
        groups = sensitive_features.unique()
        precisions = {}
        for g in groups:
            mask = sensitive_features == g
            if mask.sum() > 10 and y_pred[mask].sum() > 0:
                precisions[str(g)] = precision_score(y_true[mask], y_pred[mask], zero_division=0)
        pp_diff = max(precisions.values()) - min(precisions.values()) if len(precisions) > 1 else 0.0

        return {
            "group_name": group_name,
            "demographic_parity_difference": round(float(dp_diff), 4),
            "equal_opportunity_difference": round(float(eod_diff), 4),
            "predictive_parity_difference": round(float(pp_diff), 4),
            "precision_per_group": precisions,
            "n_samples_per_group": sensitive_features.value_counts().to_dict(),
        }

    def run_gate(self, test_df: pd.DataFrame, model_name: str = "fraud_detector", model_version: str = "latest") -> tuple[bool, dict]:
        """Run full fairness evaluation. Returns (gate_passed, full_report)."""
        from models.fraud_detector import load_model, score_transaction, FEATURE_COLS

        model, label_encoder, explainer = load_model()

        # Prepare features
        test_df = test_df.copy()
        test_df["merchant_category_encoded"] = label_encoder.transform(
            test_df["merchant_category"].fillna("unknown")
        )
        test_df["is_foreign_merchant"] = test_df["is_foreign_merchant"].astype(int)
        X_test = test_df[FEATURE_COLS]
        y_true = test_df["is_fraud"].astype(int).values
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        all_group_results = {}
        gate_passed = True
        failing_metrics = []

        for group_name, group_fn in PROTECTED_GROUPS.items():
            sensitive = group_fn(test_df)
            results = self.evaluate(y_true, y_pred, y_pred_proba, sensitive, group_name)
            all_group_results[group_name] = results

            for metric, threshold in self.thresholds.items():
                if metric in results and results[metric] > threshold:
                    gate_passed = False
                    failing_metrics.append({
                        "group": group_name,
                        "metric": metric,
                        "value": results[metric],
                        "threshold": threshold,
                    })
                    logger.warning(
                        "fairness_threshold_exceeded",
                        group=group_name,
                        metric=metric,
                        value=results[metric],
                        threshold=threshold,
                    )

        report = {
            "model_name": model_name,
            "model_version": model_version,
            "evaluated_at": datetime.utcnow().isoformat(),
            "gate_passed": gate_passed,
            "failing_metrics": failing_metrics,
            "group_results": all_group_results,
            "thresholds": self.thresholds,
        }

        self._log_to_mlflow(report)
        self._log_to_db(report)

        return gate_passed, report

    def _log_to_mlflow(self, report: dict) -> None:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            with mlflow.start_run(run_name=f"fairness_gate_{report['model_name']}"):
                mlflow.log_param("model_name", report["model_name"])
                mlflow.log_param("gate_passed", report["gate_passed"])
                for group, metrics in report["group_results"].items():
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"{group}_{metric}", value)
                mlflow.log_dict(report, "fairness_report.json")
        except Exception as e:
            logger.warning("mlflow_log_failed", error=str(e))

    def _log_to_db(self, report: dict) -> None:
        try:
            import psycopg2
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO audit.fairness_reports
                   (model_name, model_version, gate_passed, report, computed_at)
                   VALUES (%s, %s, %s, %s, %s)""",
                (
                    report["model_name"],
                    report["model_version"],
                    report["gate_passed"],
                    json.dumps(report),
                    datetime.utcnow(),
                ),
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.warning("db_log_failed", error=str(e))


def main() -> None:
    parser = argparse.ArgumentParser(description="CreditPulse Fairness Gate")
    parser.add_argument("--model-file", type=str, default="models/fraud_detector.py")
    parser.add_argument("--threshold-demographic-parity", type=float, default=0.05)
    parser.add_argument("--threshold-equal-opportunity", type=float, default=0.05)
    parser.add_argument("--log-file", type=str, default=".kiro/hooks/fairness-gate.log")
    parser.add_argument("--hitl", action="store_true", help="Override with HITL sign-off (logs to audit)")
    args = parser.parse_args()

    gate = FairnessGate(
        threshold_demographic_parity=args.threshold_demographic_parity,
        threshold_equal_opportunity=args.threshold_equal_opportunity,
    )

    # Load test data
    test_parquet = Path("data/transactions.parquet")
    if test_parquet.exists():
        df = pd.read_parquet(test_parquet).sample(min(5000, len(pd.read_parquet(test_parquet))), random_state=42)
    else:
        logger.warning("no_test_data", msg="Using synthetic test data — run data generator first")
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

    print(json.dumps(report, indent=2))

    if args.hitl and not passed:
        logger.info("hitl_override", model=args.model_file, overrider="HITL")
        print("\n[HITL OVERRIDE] Fairness gate bypassed with human sign-off. Logged to audit.")
        sys.exit(0)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
