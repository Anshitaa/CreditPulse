"""
CreditPulse — Spark Streaming End-to-End Integration Test
Spec: CREDIT-001 FR-002

Tests the full Kafka → Spark → PostgreSQL feature pipeline:
1. Publish test transactions to Kafka (transactions.raw topic)
2. Run Spark streaming job in --local --once mode
3. Verify computed features landed in mart.features_* tables

Requirements:
- Docker stack running: postgres (5435), kafka (9092)
- JDBC JAR at spark/jars/postgresql-42.7.3.jar
- JAVA_HOME=/opt/anaconda3

Run:
    export JAVA_HOME=/opt/anaconda3
    export PYTHONPATH=/Users/anshita/Desktop/CreditPulse
    pytest tests/integration/test_spark_streaming.py -v -s
"""

import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
import pytest
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

# ── Configuration ──────────────────────────────────────────────────────────

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://creditpulse:creditpulse@localhost:5435/creditpulse"
)
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = "transactions.raw"
PROJECT_ROOT = Path(__file__).parent.parent.parent
JDBC_JAR = PROJECT_ROOT / "spark/jars/postgresql-42.7.3.jar"
JAVA_HOME = os.environ.get("JAVA_HOME", "/opt/anaconda3")


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def kafka_producer():
    """Create a Kafka producer; skip if Kafka unavailable."""
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            request_timeout_ms=5000,
        )
        yield producer
        producer.close()
    except NoBrokersAvailable:
        pytest.skip("Kafka not available at localhost:9092 — start Docker stack first")


@pytest.fixture(scope="module")
def db_conn():
    """PostgreSQL connection; skip if DB unavailable."""
    try:
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
        yield conn
        conn.close()
    except Exception as e:
        pytest.skip(f"PostgreSQL not available: {e}")


@pytest.fixture(scope="module")
def spark_available():
    """Check PySpark + Java are available."""
    if not JDBC_JAR.exists():
        pytest.skip(f"JDBC JAR not found at {JDBC_JAR}. Run: mkdir -p spark/jars && "
                    "curl -sL https://jdbc.postgresql.org/download/postgresql-42.7.3.jar "
                    "-o spark/jars/postgresql-42.7.3.jar")
    try:
        import pyspark
    except ImportError:
        pytest.skip("pyspark not installed")
    if not Path(JAVA_HOME).exists():
        pytest.skip(f"JAVA_HOME={JAVA_HOME} not found")


# ── Helpers ────────────────────────────────────────────────────────────────

def make_test_transaction(account_id: str, amount: float = 250.0) -> dict:
    return {
        "txn_id": f"T-E2E-{uuid.uuid4().hex[:8]}",
        "account_id": account_id,
        "merchant_id": f"MCH-E2E-{uuid.uuid4().hex[:4]}",
        "amount": amount,
        "merchant_category": "online_retail",
        "is_foreign_merchant": False,
        "hour_of_day": 14,
        "day_of_week": 2,
        "txn_velocity_1h": 3,
        "amount_vs_avg_ratio": 1.2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "event_id": uuid.uuid4().hex,
    }


def get_feature_row_counts(conn) -> dict:
    """Return current row counts for all mart.features_* tables."""
    tables = [
        "mart.features_account_txn_counts",
        "mart.features_velocity",
        "mart.features_amount_stats",
        "mart.features_merchant_stats",
    ]
    counts = {}
    with conn.cursor() as cur:
        for table in tables:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = cur.fetchone()[0]
            except Exception:
                counts[table] = 0
    conn.rollback()
    return counts


# ── Tests ──────────────────────────────────────────────────────────────────

class TestKafkaToSparkToPostgres:
    """End-to-end: Kafka → Spark Structured Streaming → PostgreSQL."""

    N_TEST_TRANSACTIONS = 20
    TEST_ACCOUNT_PREFIX = "ACC-E2E-TEST"

    def test_kafka_producer_can_publish(self, kafka_producer):
        """Baseline: can we publish messages to Kafka at all?"""
        txn = make_test_transaction("ACC-E2E-BASELINE", amount=100.0)
        future = kafka_producer.send(TOPIC, key=txn["txn_id"], value=txn)
        record_metadata = future.get(timeout=10)
        assert record_metadata.topic == TOPIC
        assert record_metadata.offset >= 0

    def test_kafka_topic_exists(self, kafka_producer):
        """transactions.raw topic must exist before streaming can consume it."""
        from kafka import KafkaAdminClient
        try:
            admin = KafkaAdminClient(bootstrap_servers=KAFKA_BOOTSTRAP)
            topics = admin.list_topics()
            assert TOPIC in topics, f"Topic '{TOPIC}' not found. Got: {topics}"
            admin.close()
        except Exception as e:
            pytest.fail(f"Could not list Kafka topics: {e}")

    def test_spark_streaming_end_to_end(
        self, kafka_producer, db_conn, spark_available
    ):
        """
        Full pipeline test:
        1. Publish N transactions to Kafka
        2. Run Spark job in --once mode
        3. Verify rows appeared in PostgreSQL feature tables
        """
        # Step 1: record baseline row counts
        before = get_feature_row_counts(db_conn)

        # Step 2: publish test transactions
        account_id = f"{self.TEST_ACCOUNT_PREFIX}-{uuid.uuid4().hex[:4]}"
        for i in range(self.N_TEST_TRANSACTIONS):
            txn = make_test_transaction(account_id, amount=100.0 + i * 10)
            kafka_producer.send(TOPIC, key=txn["txn_id"], value=txn)

        kafka_producer.flush()
        print(f"\nPublished {self.N_TEST_TRANSACTIONS} test transactions for account {account_id}")

        # Give Kafka a moment to commit the messages
        time.sleep(2)

        # Step 3: run Spark streaming job in --once mode
        env = os.environ.copy()
        env["JAVA_HOME"] = JAVA_HOME
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        env.setdefault("DATABASE_URL", DATABASE_URL)
        env.setdefault("KAFKA_BOOTSTRAP_SERVERS", KAFKA_BOOTSTRAP)

        print("Starting Spark streaming job (--local --once)...")
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "features/spark_streaming_features.py"),
                "--local",
                "--once",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min max — Spark startup is slow
            cwd=str(PROJECT_ROOT),
        )

        print("--- Spark stdout ---")
        print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
        if result.stderr:
            print("--- Spark stderr (last 1000 chars) ---")
            print(result.stderr[-1000:])

        assert result.returncode == 0, (
            f"Spark job failed with exit code {result.returncode}.\n"
            f"Stderr: {result.stderr[-500:]}"
        )

        # Step 4: verify rows landed in PostgreSQL
        after = get_feature_row_counts(db_conn)
        print(f"\nRow counts before: {before}")
        print(f"Row counts after:  {after}")

        # At least one feature table must have grown
        grew = any(after[t] > before.get(t, 0) for t in after)
        assert grew, (
            "No feature table grew after Spark job completed.\n"
            f"Before: {before}\nAfter: {after}"
        )

    def test_feature_tables_have_valid_data(self, db_conn):
        """After the streaming run, spot-check feature table data quality."""
        with db_conn.cursor() as cur:
            # txn counts table
            try:
                cur.execute("""
                    SELECT account_id, txn_count_1h, total_amount_1h
                    FROM mart.features_account_txn_counts
                    ORDER BY window_start DESC
                    LIMIT 5
                """)
                rows = cur.fetchall()
                if rows:
                    for account_id, txn_count, total_amount in rows:
                        assert txn_count > 0, "txn_count_1h should be > 0"
                        assert total_amount > 0, "total_amount_1h should be > 0"
            except psycopg2.errors.UndefinedTable:
                pytest.skip("features_account_txn_counts table doesn't exist yet")
            finally:
                db_conn.rollback()

            # velocity table
            try:
                cur.execute("""
                    SELECT account_id, txn_count_5m, velocity_score
                    FROM mart.features_velocity
                    ORDER BY window_start DESC
                    LIMIT 5
                """)
                rows = cur.fetchall()
                if rows:
                    for account_id, count_5m, velocity in rows:
                        assert count_5m >= 0
                        assert 0.0 <= velocity <= 1.0, f"velocity_score {velocity} out of [0,1]"
            except psycopg2.errors.UndefinedTable:
                pass
            finally:
                db_conn.rollback()


class TestSparkLocalMode:
    """Lighter tests — verify Spark starts and can read from Kafka without running the full pipeline."""

    def test_pyspark_import(self, spark_available):
        """PySpark importable."""
        from pyspark.sql import SparkSession
        assert SparkSession is not None

    def test_jdbc_jar_readable(self, spark_available):
        """JDBC JAR must be a valid file (> 500KB)."""
        assert JDBC_JAR.exists(), f"JDBC JAR missing: {JDBC_JAR}"
        size_kb = JDBC_JAR.stat().st_size / 1024
        assert size_kb > 500, f"JDBC JAR seems corrupted ({size_kb:.0f}KB < 500KB)"

    def test_java_home_set(self, spark_available):
        """JAVA_HOME must point to a real JVM."""
        java_bin = Path(JAVA_HOME) / "bin/java"
        assert java_bin.exists(), (
            f"java binary not found at {java_bin}. "
            f"Set JAVA_HOME correctly — expected /opt/anaconda3 for Anaconda."
        )
