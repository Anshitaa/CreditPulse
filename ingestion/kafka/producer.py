"""
CreditPulse — Idempotent Kafka Transaction Producer
Spec: CREDIT-001 FR-001

Streams synthetic transaction events to Kafka topic `transactions.raw`.
Supports replay mode (rewind to watermark) and real-time simulation mode.

Usage:
    python ingestion/kafka/producer.py --mode simulate --rate 100  # 100 txn/s
    python ingestion/kafka/producer.py --mode replay --from-parquet data/transactions.parquet
"""

import argparse
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import structlog
from confluent_kafka import Producer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.json_schema import JSONSerializer
from confluent_kafka.serialization import MessageField, SerializationContext

logger = structlog.get_logger(__name__)

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
SCHEMA_REGISTRY_URL = os.environ.get("SCHEMA_REGISTRY_URL", "http://localhost:8081")
TOPIC = "transactions.raw"

TRANSACTION_SCHEMA = json.dumps({
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Transaction",
    "type": "object",
    "properties": {
        "txn_id": {"type": "string"},
        "account_id": {"type": "string"},
        "merchant_id": {"type": "string"},
        "amount": {"type": "number"},
        "merchant_category": {"type": "string"},
        "is_foreign_merchant": {"type": "boolean"},
        "hour_of_day": {"type": "integer"},
        "day_of_week": {"type": "integer"},
        "txn_velocity_1h": {"type": "integer"},
        "amount_vs_avg_ratio": {"type": "number"},
        "created_at": {"type": "string", "format": "date-time"},
        "event_id": {"type": "string"},  # idempotency key
    },
    "required": ["txn_id", "account_id", "amount", "created_at", "event_id"],
})


def _delivery_callback(err, msg):
    if err:
        logger.error("delivery_failed", error=str(err), topic=msg.topic())
    else:
        logger.debug("delivered", topic=msg.topic(), partition=msg.partition(), offset=msg.offset())


def create_producer() -> Producer:
    return Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "enable.idempotence": True,          # exactly-once delivery
        "acks": "all",
        "retries": 5,
        "max.in.flight.requests.per.connection": 5,
        "compression.type": "snappy",
        "linger.ms": 10,                     # micro-batch for throughput
        "batch.size": 65536,
    })


def produce_event(producer: Producer, txn: dict) -> None:
    """Produce a single transaction event with idempotency key."""
    payload = {**txn, "event_id": str(uuid.uuid4())}
    producer.produce(
        topic=TOPIC,
        key=txn["account_id"].encode("utf-8"),  # partition by account for ordering
        value=json.dumps(payload, default=str).encode("utf-8"),
        on_delivery=_delivery_callback,
    )


def simulate_realtime(rate_per_second: int, duration_seconds: int | None = None) -> None:
    """Generate and stream synthetic transactions in real time."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from data.synthetic_transactions import generate_accounts, generate_merchants, generate_transactions
    import numpy as np

    producer = create_producer()
    rng = np.random.default_rng(int(time.time()))

    accounts = generate_accounts(1_000, rng)
    merchants = generate_merchants(100, rng)

    logger.info("starting_simulation", rate_per_second=rate_per_second, topic=TOPIC)
    interval = 1.0 / rate_per_second
    start = time.monotonic()
    count = 0

    try:
        while True:
            if duration_seconds and (time.monotonic() - start) > duration_seconds:
                break
            txn = generate_transactions(accounts, merchants, 1, rng, datetime.utcnow())[0]
            produce_event(producer, {
                "txn_id": txn.txn_id,
                "account_id": txn.account_id,
                "merchant_id": txn.merchant_id,
                "amount": txn.amount,
                "merchant_category": txn.merchant_category,
                "is_foreign_merchant": txn.is_foreign_merchant,
                "hour_of_day": txn.hour_of_day,
                "day_of_week": txn.day_of_week,
                "txn_velocity_1h": txn.txn_velocity_1h,
                "amount_vs_avg_ratio": txn.amount_vs_avg_ratio,
                "created_at": txn.created_at.isoformat(),
            })
            count += 1
            if count % 1000 == 0:
                producer.poll(0)
                logger.info("produced", count=count, elapsed_s=round(time.monotonic() - start, 1))
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("shutting_down", total_produced=count)
    finally:
        producer.flush(timeout=30)


def replay_from_parquet(parquet_path: str, rate_per_second: int = 1000) -> None:
    """Replay historical transactions from Parquet file."""
    producer = create_producer()
    df = pd.read_parquet(parquet_path)
    logger.info("replaying", rows=len(df), path=parquet_path, rate=rate_per_second)

    interval = 1.0 / rate_per_second
    for i, row in df.iterrows():
        produce_event(producer, row.to_dict())
        if i % 10_000 == 0:
            producer.poll(0)
            logger.info("replayed", count=i, total=len(df))
        time.sleep(interval)

    producer.flush(timeout=30)
    logger.info("replay_complete", total=len(df))


def main() -> None:
    parser = argparse.ArgumentParser(description="CreditPulse Kafka Transaction Producer")
    parser.add_argument("--mode", choices=["simulate", "replay"], default="simulate")
    parser.add_argument("--rate", type=int, default=100, help="Transactions per second")
    parser.add_argument("--duration", type=int, default=None, help="Run for N seconds (simulate mode)")
    parser.add_argument("--from-parquet", type=str, help="Parquet file path (replay mode)")
    args = parser.parse_args()

    if args.mode == "simulate":
        simulate_realtime(args.rate, args.duration)
    elif args.mode == "replay":
        if not args.from_parquet:
            raise ValueError("--from-parquet required for replay mode")
        replay_from_parquet(args.from_parquet, args.rate)


if __name__ == "__main__":
    main()
