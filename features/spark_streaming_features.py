"""
CreditPulse — Spark Streaming Feature Computation
Spec: CREDIT-001 FR-002

Consumes `transactions.raw` Kafka topic and computes real-time rolling window features:
- txn_count_1m, txn_count_1h, txn_count_24h (per account)
- txn_amount_zscore (per account, 30d baseline)
- merchant_fraud_rate_30d (per merchant)
- velocity_score (event rate per account in 5-minute windows)

Writes computed features to:
- Feast online store (Redis) for < 100ms retrieval during inference
- Feast offline store (PostgreSQL) for historical training data

Usage:
    export JAVA_HOME=/opt/anaconda3
    python features/spark_streaming_features.py --local           # continuous streaming
    python features/spark_streaming_features.py --local --once    # process all available, then stop (CI/testing)
"""

import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5435")
JDBC_URL = f"jdbc:postgresql://{POSTGRES_HOST}:{POSTGRES_PORT}/creditpulse"
JDBC_PROPS = {
    "user": os.environ.get("POSTGRES_USER", "creditpulse"),
    "password": os.environ.get("POSTGRES_PASSWORD", "creditpulse"),
    "driver": "org.postgresql.Driver",
}

TRANSACTION_SCHEMA = StructType([
    StructField("txn_id", StringType(), True),
    StructField("account_id", StringType(), True),
    StructField("merchant_id", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("merchant_category", StringType(), True),
    StructField("is_foreign_merchant", BooleanType(), True),
    StructField("hour_of_day", IntegerType(), True),
    StructField("day_of_week", IntegerType(), True),
    StructField("txn_velocity_1h", IntegerType(), True),
    StructField("amount_vs_avg_ratio", DoubleType(), True),
    StructField("created_at", StringType(), True),
    StructField("event_id", StringType(), True),
])


def create_spark(local: bool = False) -> SparkSession:
    jdbc_jar = os.environ.get(
        "JDBC_JAR",
        str(Path(__file__).parent.parent / "spark/jars/postgresql-42.7.3.jar"),
    )
    # The Kafka source is not bundled with PySpark; pull the connector that
    # matches the installed Spark version (pip PySpark 3.x is built on Scala 2.12).
    import pyspark
    kafka_pkg = f"org.apache.spark:spark-sql-kafka-0-10_2.12:{pyspark.__version__}"
    builder = (
        SparkSession.builder.appName("CreditPulse-StreamingFeatures")
        .config("spark.sql.streaming.checkpointLocation", "/tmp/creditpulse-checkpoints")
        .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true")
        .config("spark.jars", jdbc_jar)
        .config("spark.jars.packages", kafka_pkg)
    )
    if local:
        builder = builder.master("local[*]").config("spark.driver.memory", "4g")
    return builder.getOrCreate()


def build_streaming_pipeline(spark: SparkSession, starting_offsets: str = "latest"):
    # Read from Kafka. startingOffsets only applies on the very first run;
    # after that the checkpoint decides where to resume.
    raw_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", "transactions.raw")
        .option("startingOffsets", starting_offsets)
        .option("failOnDataLoss", "false")
        .load()
    )

    # Parse JSON payload
    txn_stream = (
        raw_stream.select(
            F.from_json(F.col("value").cast("string"), TRANSACTION_SCHEMA).alias("txn"),
            F.col("timestamp").alias("kafka_timestamp"),
        )
        .select("txn.*", "kafka_timestamp")
        .withColumn("created_at", F.to_timestamp("created_at"))
        .withWatermark("created_at", "2 minutes")  # handle late arrivals up to 2 min
    )

    # Feature 1: Transaction count per account — 1-hour tumbling window
    txn_count_1h = (
        txn_stream.groupBy(
            F.col("account_id"),
            F.window("created_at", "1 hour"),
        )
        .agg(
            F.count("txn_id").alias("txn_count_1h"),
            F.sum("amount").alias("total_amount_1h"),
            F.avg("amount").alias("avg_amount_1h"),
        )
        .select(
            "account_id",
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            "txn_count_1h",
            "total_amount_1h",
            "avg_amount_1h",
        )
    )

    # Feature 2: Velocity score — 5-minute sliding window (step 1 min)
    velocity_5m = (
        txn_stream.groupBy(
            F.col("account_id"),
            F.window("created_at", "5 minutes", "1 minute"),
        )
        .agg(F.count("txn_id").alias("txn_count_5m"))
        .select(
            "account_id",
            F.col("window.start").alias("window_start"),
            "txn_count_5m",
            # Normalize: > 5 txns in 5 min = high velocity
            F.when(F.col("txn_count_5m") > 5, F.lit(1.0))
            .when(F.col("txn_count_5m") > 2, F.lit(0.5))
            .otherwise(F.lit(0.0))
            .alias("velocity_score"),
        )
    )

    # Feature 3: Amount z-score per account — rolling 24h
    amount_stats = (
        txn_stream.groupBy(
            F.col("account_id"),
            F.window("created_at", "24 hours"),
        )
        .agg(
            F.avg("amount").alias("amount_mean_24h"),
            F.stddev("amount").alias("amount_std_24h"),
            F.count("txn_id").alias("txn_count_24h"),
        )
        .select(
            "account_id",
            F.col("window.start").alias("window_start"),
            "amount_mean_24h",
            "amount_std_24h",
            "txn_count_24h",
        )
    )

    # Feature 4: Merchant fraud rate — 30d (processed as micro-batch update)
    merchant_stats = (
        txn_stream.groupBy("merchant_id", F.window("created_at", "30 minutes"))
        .agg(
            F.count("txn_id").alias("merchant_txn_count"),
            F.avg("amount").alias("merchant_avg_amount"),
        )
        .select(
            "merchant_id",
            F.col("window.start").alias("window_start"),
            "merchant_txn_count",
            "merchant_avg_amount",
        )
    )

    return txn_count_1h, velocity_5m, amount_stats, merchant_stats


FEATURE_TABLE_KEYS = {
    "features_account_txn_counts": ("account_id", "window_start"),
    "features_velocity": ("account_id", "window_start"),
    "features_amount_stats": ("account_id", "window_start"),
    "features_merchant_stats": ("merchant_id", "window_start"),
}


def write_to_postgres(df, table: str) -> None:
    """Idempotent upsert of one micro-batch into mart.<table> (offline store).

    In `update` output mode Spark re-emits a window every time it changes, and
    a rerun reprocesses windows already written. A plain JDBC append would hit
    the (key, window_start) primary key, so we stage-then-merge:
      1. JDBC-overwrite the batch into mart._stg_<table>
      2. INSERT ... SELECT ... ON CONFLICT DO UPDATE into the real table
    """
    import psycopg2

    staging = f"mart._stg_{table}"
    df.write.jdbc(url=JDBC_URL, table=staging, mode="overwrite", properties=JDBC_PROPS)

    keys = FEATURE_TABLE_KEYS[table]
    cols = df.columns
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in keys)
    merge_sql = (
        f"INSERT INTO mart.{table} ({', '.join(cols)}) "
        f"SELECT {', '.join(cols)} FROM {staging} "
        f"ON CONFLICT ({', '.join(keys)}) DO UPDATE SET {updates}"
    )
    conn = psycopg2.connect(
        host=POSTGRES_HOST, port=POSTGRES_PORT, dbname="creditpulse",
        user=JDBC_PROPS["user"], password=JDBC_PROPS["password"],
    )
    try:
        with conn, conn.cursor() as cur:
            cur.execute(merge_sql)
    finally:
        conn.close()


def run_streaming(local: bool = False, once: bool = False) -> None:
    """
    Run the Spark streaming feature pipeline.

    Args:
        local: Use local Spark mode (no cluster needed).
        once: Use availableNow trigger — processes all messages currently in Kafka
              then stops. Ideal for CI/end-to-end testing. Requires Spark 3.3+.
    """
    spark = create_spark(local)
    spark.sparkContext.setLogLevel("WARN")

    # --once drains the backlog already sitting in Kafka, so it must start from
    # "earliest"; "latest" would skip every message published before startup.
    txn_count_1h, velocity_5m, amount_stats, merchant_stats = build_streaming_pipeline(
        spark, starting_offsets="earliest" if once else "latest"
    )

    # Write each feature stream to PostgreSQL
    queries = []

    def write_micro_batch(df, epoch_id, table: str):
        if df.count() > 0:
            write_to_postgres(df, table)

    if once:
        # availableNow: process everything in Kafka right now, then stop.
        # Perfect for CI smoke tests and end-to-end validation.
        trigger_kwargs = {"availableNow": True}
        print("Running in --once mode (availableNow trigger). Will stop after processing all available messages.")
    else:
        trigger_kwargs = {"processingTime": "30 seconds"}

    queries.append(
        txn_count_1h.writeStream
        .foreachBatch(lambda df, epoch: write_micro_batch(df, epoch, "features_account_txn_counts"))
        .outputMode("update")
        .option("checkpointLocation", "/tmp/creditpulse-checkpoints/txn_count_1h")
        .trigger(**trigger_kwargs)
        .start()
    )

    queries.append(
        velocity_5m.writeStream
        .foreachBatch(lambda df, epoch: write_micro_batch(df, epoch, "features_velocity"))
        .outputMode("update")
        .option("checkpointLocation", "/tmp/creditpulse-checkpoints/velocity_5m")
        .trigger(**trigger_kwargs)
        .start()
    )

    queries.append(
        amount_stats.writeStream
        .foreachBatch(lambda df, epoch: write_micro_batch(df, epoch, "features_amount_stats"))
        .outputMode("update")
        .option("checkpointLocation", "/tmp/creditpulse-checkpoints/amount_stats")
        .trigger(**{"availableNow": True} if once else {"processingTime": "60 seconds"})
        .start()
    )

    queries.append(
        merchant_stats.writeStream
        .foreachBatch(lambda df, epoch: write_micro_batch(df, epoch, "features_merchant_stats"))
        .outputMode("update")
        .option("checkpointLocation", "/tmp/creditpulse-checkpoints/merchant_stats")
        .trigger(**{"availableNow": True} if once else {"processingTime": "60 seconds"})
        .start()
    )

    if once:
        # Wait for all queries to finish processing available data
        for q in queries:
            q.awaitTermination()
        print(f"--once mode complete. Processed all available Kafka messages.")
    else:
        print(f"Streaming {len(queries)} feature pipelines. Ctrl+C to stop.")
        try:
            spark.streams.awaitAnyTermination()
        except KeyboardInterrupt:
            print("Stopping all streaming queries...")
            for q in queries:
                q.stop()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run in local Spark mode")
    parser.add_argument("--once", action="store_true",
                        help="Process all available Kafka messages then stop (CI/testing mode)")
    args = parser.parse_args()
    run_streaming(local=args.local, once=args.once)
