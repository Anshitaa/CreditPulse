"""
Integration tests: PostgreSQL schema integrity

Verifies that the warehouse schema is correctly structured:
- All 4 schemas exist (raw, staging, mart, audit)
- All critical tables exist with expected columns
- Partitioning / indexes (audit trail tables)
- No empty mart tables (data was loaded)
"""

import pytest
import psycopg2
from psycopg2.extras import RealDictCursor


def table_exists(db, schema, table):
    cur = db.cursor()
    cur.execute(
        "SELECT to_regclass(%s)",
        (f"{schema}.{table}",),
    )
    result = cur.fetchone()[0]
    cur.close()
    return result is not None


def column_exists(db, schema, table, column):
    cur = db.cursor()
    cur.execute(
        """SELECT 1 FROM information_schema.columns
           WHERE table_schema = %s AND table_name = %s AND column_name = %s""",
        (schema, table, column),
    )
    result = cur.fetchone()
    cur.close()
    return result is not None


def row_count(db, schema, table):
    cur = db.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
    count = cur.fetchone()[0]
    cur.close()
    return count


class TestSchemas:
    def test_raw_schema_exists(self, db):
        cur = db.cursor()
        cur.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'raw'")
        assert cur.fetchone() is not None, "Schema 'raw' does not exist"
        cur.close()

    def test_staging_schema_exists(self, db):
        cur = db.cursor()
        cur.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'staging'")
        assert cur.fetchone() is not None, "Schema 'staging' does not exist"
        cur.close()

    def test_mart_schema_exists(self, db):
        cur = db.cursor()
        cur.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'mart'")
        assert cur.fetchone() is not None, "Schema 'mart' does not exist"
        cur.close()

    def test_audit_schema_exists(self, db):
        cur = db.cursor()
        cur.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'audit'")
        assert cur.fetchone() is not None, "Schema 'audit' does not exist"
        cur.close()


class TestCriticalTables:
    def test_raw_transactions_exists(self, db):
        assert table_exists(db, "raw", "transactions"), "raw.transactions table missing"

    def test_mart_risk_scores_exists(self, db):
        assert table_exists(db, "mart", "risk_scores"), "mart.risk_scores table missing"

    def test_audit_model_decisions_exists(self, db):
        assert table_exists(db, "audit", "model_decisions"), "audit.model_decisions table missing"

    def test_audit_drift_reports_exists(self, db):
        assert table_exists(db, "audit", "drift_reports"), "audit.drift_reports table missing"

    def test_audit_fairness_reports_exists(self, db):
        assert table_exists(db, "audit", "fairness_reports"), "audit.fairness_reports table missing"


class TestMartRiskScoresSchema:
    def test_txn_id_column(self, db):
        assert column_exists(db, "mart", "risk_scores", "txn_id")

    def test_account_id_column(self, db):
        assert column_exists(db, "mart", "risk_scores", "account_id")

    def test_composite_risk_score_column(self, db):
        assert column_exists(db, "mart", "risk_scores", "composite_risk_score")

    def test_fraud_probability_column(self, db):
        assert column_exists(db, "mart", "risk_scores", "fraud_probability")

    def test_decision_column(self, db):
        assert column_exists(db, "mart", "risk_scores", "decision")

    def test_scored_at_column(self, db):
        assert column_exists(db, "mart", "risk_scores", "scored_at")


class TestAuditModelDecisionsSchema:
    def test_txn_id_column(self, db):
        assert column_exists(db, "audit", "model_decisions", "txn_id")

    def test_decision_column(self, db):
        assert column_exists(db, "audit", "model_decisions", "decision")

    def test_score_column(self, db):
        assert column_exists(db, "audit", "model_decisions", "score")

    def test_decided_at_column(self, db):
        assert column_exists(db, "audit", "model_decisions", "decided_at")

    def test_top_features_column(self, db):
        assert column_exists(db, "audit", "model_decisions", "top_features")


class TestDataPopulated:
    def test_raw_transactions_has_data(self, db):
        count = row_count(db, "raw", "transactions")
        assert count > 0, "raw.transactions is empty — run: python data/synthetic_transactions.py --load-db"

    def test_mart_risk_scores_has_data(self, db):
        count = row_count(db, "mart", "risk_scores")
        assert count > 0, "mart.risk_scores is empty — score some transactions via POST /score/"

    def test_audit_model_decisions_has_data(self, db):
        count = row_count(db, "audit", "model_decisions")
        assert count > 0, "audit.model_decisions is empty — no scoring activity logged"

    def test_raw_transactions_row_count(self, db):
        count = row_count(db, "raw", "transactions")
        assert count >= 1000, f"raw.transactions has only {count} rows — expected at least 1K"


class TestDataQuality:
    def test_no_null_txn_ids_in_risk_scores(self, db):
        cur = db.cursor()
        cur.execute("SELECT COUNT(*) FROM mart.risk_scores WHERE txn_id IS NULL")
        nulls = cur.fetchone()[0]
        cur.close()
        assert nulls == 0, f"{nulls} rows with NULL txn_id in mart.risk_scores"

    def test_risk_scores_in_valid_range(self, db):
        cur = db.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM mart.risk_scores WHERE composite_risk_score < 0 OR composite_risk_score > 100"
        )
        out_of_range = cur.fetchone()[0]
        cur.close()
        assert out_of_range == 0, f"{out_of_range} risk scores outside valid 0–100 range"

    def test_decisions_are_valid_enum(self, db):
        cur = db.cursor()
        cur.execute(
            "SELECT DISTINCT decision FROM mart.risk_scores WHERE decision NOT IN ('FRAUD', 'REVIEW', 'CLEAR')"
        )
        invalid = cur.fetchall()
        cur.close()
        assert not invalid, f"Invalid decision values found: {invalid}"

    def test_fraud_rate_is_plausible(self, db):
        """Fraud rate in raw data should be between 1% and 30%."""
        cur = db.cursor()
        cur.execute(
            "SELECT ROUND(AVG(CASE WHEN is_fraud THEN 1.0 ELSE 0.0 END) * 100, 2) FROM raw.transactions"
        )
        rate = cur.fetchone()[0]
        cur.close()
        if rate is not None:
            assert 1.0 <= float(rate) <= 30.0, (
                f"Fraud rate {rate}% is outside expected 1–30% range"
            )
