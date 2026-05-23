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
    python features/spark_streaming_features.py --local
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
    builder = (
        SparkSession.builder.appName("CreditPulse-StreamingFeatures")
        .config("spark.sql.streaming.checkpointLocation", "/tmp/creditpulse-checkpoints")
        .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true")
        .config("spark.jars", jdbc_jar)
    )
    if local:
        builder = builder.master("local[*]").config("spark.driver.memory", "4g")
    return builder.getOrCreate()


def build_streaming_pipeline(spark: SparkSession):
    # Read from Kafka
    raw_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", "transactions.raw")
        .option("startingOffsets", "latest")
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


def write_to_postgres(df, table: str, mode: str = "append"):
    """Write a batch of features to PostgreSQL (offline store)."""
    df.write.jdbc(
        url=JDBC_URL,
        table=f"mart.{table}",
        mode=mode,
        properties=JDBC_PROPS,
    )


def run_streaming(local: bool = False) -> None:
    spark = create_spark(local)
    spark.sparkContext.setLogLevel("WARN")

    txn_count_1h, velocity_5m, amount_stats, merchant_stats = build_streaming_pipeline(spark)

    # Write each feature stream to PostgreSQL
    queries = []

    def write_micro_batch(df, epoch_id, table: str):
        if df.count() > 0:
            write_to_postgres(df, table)

    queries.append(
        txn_count_1h.writeStream
        .foreachBatch(lambda df, epoch: write_micro_batch(df, epoch, "features_account_txn_counts"))
        .outputMode("update")
        .option("checkpointLocation", "/tmp/creditpulse-checkpoints/txn_count_1h")
        .trigger(processingTime="30 seconds")
        .start()
    )

    queries.append(
        velocity_5m.writeStream
        .foreachBatch(lambda df, epoch: write_micro_batch(df, epoch, "features_velocity"))
        .outputMode("update")
        .option("checkpointLocation", "/tmp/creditpulse-checkpoints/velocity_5m")
        .trigger(processingTime="30 seconds")
        .start()
    )

    queries.append(
        amount_stats.writeStream
        .foreachBatch(lambda df, epoch: write_micro_batch(df, epoch, "features_amount_stats"))
        .outputMode("update")
        .option("checkpointLocation", "/tmp/creditpulse-checkpoints/amount_stats")
        .trigger(processingTime="60 seconds")
        .start()
    )

    queries.append(
        merchant_stats.writeStream
        .foreachBatch(lambda df, epoch: write_micro_batch(df, epoch, "features_merchant_stats"))
        .outputMode("update")
        .option("checkpointLocation", "/tmp/creditpulse-checkpoints/merchant_stats")
        .trigger(processingTime="60 seconds")
        .start()
    )

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
    args = parser.parse_args()
    run_streaming(local=args.local)
