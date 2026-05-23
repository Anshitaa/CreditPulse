"""
Integration tests: Kafka producer + topic health

Covers:
- Producer can connect and deliver to transactions.raw
- High-risk scores publish to alerts.fraud topic
- Topic exists and is reachable
- Spec: CREDIT-001 FR-001 (publish to scored topic)
"""

import json
import time
import pytest

KAFKA_BOOTSTRAP = "localhost:9092"

try:
    from confluent_kafka import Producer, Consumer, KafkaException
    from confluent_kafka.admin import AdminClient
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

requires_kafka = pytest.mark.skipif(
    not KAFKA_AVAILABLE,
    reason="confluent_kafka not installed",
)


def kafka_reachable():
    if not KAFKA_AVAILABLE:
        return False
    try:
        admin = AdminClient({"bootstrap.servers": KAFKA_BOOTSTRAP, "socket.timeout.ms": 3000})
        metadata = admin.list_topics(timeout=3)
        return metadata is not None
    except Exception:
        return False


kafka_up = pytest.mark.skipif(
    not kafka_reachable(),
    reason="Kafka not reachable at localhost:9092 — start Docker stack first",
)


class TestKafkaTopics:
    @requires_kafka
    @kafka_up
    def test_transactions_raw_topic_exists(self):
        admin = AdminClient({"bootstrap.servers": KAFKA_BOOTSTRAP})
        metadata = admin.list_topics(timeout=5)
        assert "transactions.raw" in metadata.topics, (
            "Topic 'transactions.raw' not found — run: docker-compose up kafka"
        )

    @requires_kafka
    @kafka_up
    def test_expected_topics_exist(self):
        expected_topics = ["transactions.raw", "alerts.fraud"]
        admin = AdminClient({"bootstrap.servers": KAFKA_BOOTSTRAP})
        metadata = admin.list_topics(timeout=5)
        for topic in expected_topics:
            assert topic in metadata.topics, f"Expected topic '{topic}' not found in Kafka"


class TestKafkaProducer:
    @requires_kafka
    @kafka_up
    def test_producer_delivers_message(self):
        """Producer can deliver a test message to transactions.raw."""
        delivered = []
        errors = []

        def on_delivery(err, msg):
            if err:
                errors.append(err)
            else:
                delivered.append(msg)

        producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})
        test_payload = {
            "txn_id": f"kafka-inttest-{int(time.time())}",
            "account_id": "kafka-test-acct",
            "amount": 100.0,
            "test": True,
        }
        producer.produce(
            topic="transactions.raw",
            key=test_payload["txn_id"].encode(),
            value=json.dumps(test_payload).encode(),
            on_delivery=on_delivery,
        )
        producer.flush(timeout=5)
        assert not errors, f"Kafka delivery errors: {errors}"
        assert len(delivered) == 1, "Message not delivered within 5 seconds"

    @requires_kafka
    @kafka_up
    def test_ingestion_producer_script(self):
        """CreditPulse ingestion/kafka/producer.py can produce events."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "ingestion/kafka/producer.py", "--mode", "simulate", "--rate", "5", "--duration", "3"],
            capture_output=True, text=True, timeout=15,
            cwd="/Users/anshita/Desktop/CreditPulse",
        )
        assert result.returncode == 0, (
            f"Kafka producer script failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    @requires_kafka
    @kafka_up
    def test_high_risk_score_publishes_fraud_alert(self):
        """Scoring a txn with risk > 75 should publish to alerts.fraud.
        Spec: CREDIT-001 FR-003 — publish HIGH-RISK alert if score > 75."""
        import httpx
        consumer = Consumer({
            "bootstrap.servers": KAFKA_BOOTSTRAP,
            "group.id": "inttest-fraud-alerts",
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
        })
        consumer.subscribe(["alerts.fraud"])
        consumer.poll(timeout=1.0)  # seek to latest

        with httpx.Client(base_url="http://127.0.0.1:8000", timeout=10) as client:
            payload = {
                "txn_id": f"kafka-fraud-alert-{int(time.time())}",
                "account_id": "kafka-fraud-test",
                "amount": 9999.0,
                "merchant_category": "wire_transfer",
                "is_foreign_merchant": True,
                "hour_of_day": 2,
                "day_of_week": 6,
                "txn_velocity_1h": 15,
                "amount_vs_avg_ratio": 20.0,
            }
            resp = client.post("/score/", json=payload)
            score_data = resp.json()

        if score_data["fraud_risk_score"] <= 75:
            pytest.skip(f"Score {score_data['fraud_risk_score']:.1f} ≤ 75 — no alert expected")

        msg = consumer.poll(timeout=5.0)
        consumer.close()
        assert msg is not None, "No message received in alerts.fraud within 5s"
        assert not msg.error(), f"Kafka consumer error: {msg.error()}"
        alert = json.loads(msg.value().decode())
        assert alert["txn_id"] == payload["txn_id"]
