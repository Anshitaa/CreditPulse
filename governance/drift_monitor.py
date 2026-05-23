"""
CreditPulse — PSI Drift Monitor (Kiro Hook target)
Spec: CREDIT-002 FR-004

Computes Population Stability Index (PSI) for all model features.
PSI thresholds:
  < 0.10 → No significant change (STABLE)
  0.10–0.20 → Some change (MONITOR)
  > 0.20 → Major shift (RETRAIN)

Called by .kiro/hooks/psi-check.sh before every commit.
Also runs weekly via Airflow DAG (airflow/dags/drift_monitoring_dag.py).
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://creditpulse:creditpulse@localhost:5435/creditpulse")

FEATURE_COLS = [
    "amount",
    "txn_velocity_1h",
    "amount_vs_avg_ratio",
    "hour_of_day",
    "day_of_week",
]


def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Compute Population Stability Index between two distributions.

    PSI = sum((actual% - expected%) * ln(actual% / expected%))
    """
    expected_pcts, bins = np.histogram(expected, bins=n_bins)
    actual_pcts, _ = np.histogram(actual, bins=bins)

    expected_pcts = expected_pcts / len(expected)
    actual_pcts = actual_pcts / len(actual)

    # Avoid division by zero and log(0) with small epsilon
    epsilon = 1e-6
    expected_pcts = np.where(expected_pcts == 0, epsilon, expected_pcts)
    actual_pcts = np.where(actual_pcts == 0, epsilon, actual_pcts)

    psi = np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts))
    return float(psi)


def interpret_psi(psi: float) -> str:
    if psi < 0.10:
        return "STABLE"
    elif psi < 0.20:
        return "MONITOR"
    else:
        return "RETRAIN"


class DriftMonitor:
    def __init__(self, baseline_days: int = 30, comparison_days: int = 7):
        self.baseline_days = baseline_days
        self.comparison_days = comparison_days

    def load_data(self, source: str = "db") -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load baseline and current window data."""
        if source == "db":
            import psycopg2
            conn = psycopg2.connect(DATABASE_URL)
            now = datetime.utcnow()
            baseline_start = now - timedelta(days=self.baseline_days + self.comparison_days)
            baseline_end = now - timedelta(days=self.comparison_days)
            current_start = now - timedelta(days=self.comparison_days)

            baseline_df = pd.read_sql(
                "SELECT amount, txn_velocity_1h, amount_vs_avg_ratio, hour_of_day, day_of_week "
                "FROM raw.transactions WHERE created_at BETWEEN %s AND %s",
                conn,
                params=(baseline_start, baseline_end),
            )
            current_df = pd.read_sql(
                "SELECT amount, txn_velocity_1h, amount_vs_avg_ratio, hour_of_day, day_of_week "
                "FROM raw.transactions WHERE created_at >= %s",
                conn,
                params=(current_start,),
            )
            conn.close()
        else:
            # Synthetic drift simulation for demo
            rng = np.random.default_rng(42)
            n = 10_000
            baseline_df = pd.DataFrame({
                "amount": rng.lognormal(4.5, 1.0, n),
                "txn_velocity_1h": rng.integers(0, 10, n),
                "amount_vs_avg_ratio": rng.lognormal(0, 0.8, n),
                "hour_of_day": rng.integers(0, 24, n),
                "day_of_week": rng.integers(0, 7, n),
            })
            # Introduce drift: shift amount distribution up
            rng2 = np.random.default_rng(99)
            current_df = pd.DataFrame({
                "amount": rng2.lognormal(5.0, 1.2, n),  # drifted up
                "txn_velocity_1h": rng2.integers(0, 12, n),
                "amount_vs_avg_ratio": rng2.lognormal(0.3, 0.9, n),
                "hour_of_day": rng2.integers(0, 24, n),
                "day_of_week": rng2.integers(0, 7, n),
            })

        return baseline_df, current_df

    def compute_all_psi(self, baseline_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
        results = {}
        drift_detected = False
        for feat in FEATURE_COLS:
            if feat not in baseline_df.columns or feat not in current_df.columns:
                continue
            psi = compute_psi(
                baseline_df[feat].dropna().values,
                current_df[feat].dropna().values,
            )
            status = interpret_psi(psi)
            if status == "RETRAIN":
                drift_detected = True
            results[feat] = {
                "psi": round(psi, 4),
                "status": status,
                "baseline_mean": round(float(baseline_df[feat].mean()), 4),
                "current_mean": round(float(current_df[feat].mean()), 4),
                "drift_detected": status != "STABLE",
            }
            logger.info("psi_computed", feature=feat, psi=round(psi, 4), status=status)

        return {
            "computed_at": datetime.utcnow().isoformat(),
            "baseline_rows": len(baseline_df),
            "current_rows": len(current_df),
            "drift_detected": drift_detected,
            "features": results,
            "recommendation": "RETRAIN_REQUIRED" if drift_detected else "NO_ACTION",
        }

    def log_to_db(self, report: dict) -> None:
        try:
            import psycopg2
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            for feat, metrics in report["features"].items():
                cur.execute(
                    """INSERT INTO audit.drift_reports
                       (feature_name, psi_score, drift_detected, report, computed_at)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (feat, metrics["psi"], metrics["drift_detected"], json.dumps(metrics), datetime.utcnow()),
                )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.warning("db_log_failed", error=str(e))


def main() -> None:
    parser = argparse.ArgumentParser(description="CreditPulse PSI Drift Monitor")
    parser.add_argument("--mode", choices=["psi", "full"], default="psi")
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--output-format", choices=["json", "text"], default="text")
    parser.add_argument("--log-file", type=str)
    parser.add_argument("--source", choices=["db", "synthetic"], default="synthetic")
    args = parser.parse_args()

    monitor = DriftMonitor()
    baseline_df, current_df = monitor.load_data(source=args.source)
    report = monitor.compute_all_psi(baseline_df, current_df)
    monitor.log_to_db(report)

    if args.output_format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(f"\nDrift Monitor Report — {report['computed_at']}")
        print(f"Baseline rows: {report['baseline_rows']:,} | Current rows: {report['current_rows']:,}")
        print(f"Overall drift detected: {report['drift_detected']} | Recommendation: {report['recommendation']}")
        print("\nFeature PSI Scores:")
        for feat, metrics in report["features"].items():
            marker = "⚠ " if metrics["drift_detected"] else "✓ "
            print(f"  {marker}{feat}: PSI={metrics['psi']} ({metrics['status']})")

    if report["drift_detected"]:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
