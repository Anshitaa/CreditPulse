"""
CreditPulse — Synthetic Transaction Generator
Spec: CREDIT-001 (fraud detection data source)

Generates realistic financial transaction data with:
- 1M+ transactions across 50K accounts and 5K merchants
- ~2% fraud rate with realistic fraud patterns
- Temporal patterns (hour-of-day, day-of-week effects)
- Merchant risk profiles (wire transfers, gambling = higher fraud)
- Behavioral anomalies (velocity spikes, amount outliers)

Usage:
    python data/synthetic_transactions.py --rows 1000000 --load-db
    python data/synthetic_transactions.py --rows 50000 --fast  # quick demo
"""

import argparse
import json
import os
import random
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://creditpulse:creditpulse@localhost:5435/creditpulse",
)

MERCHANT_CATEGORIES = {
    "grocery": 0.005,
    "restaurant": 0.008,
    "gas_station": 0.012,
    "retail": 0.015,
    "online_retail": 0.025,
    "travel": 0.020,
    "entertainment": 0.018,
    "atm_withdrawal": 0.035,
    "peer_transfer": 0.040,
    "wire_transfer": 0.080,
    "gambling": 0.090,
    "cryptocurrency": 0.095,
    "money_service": 0.085,
}

REGIONS = ["northeast", "southeast", "midwest", "southwest", "west", "international"]

ACCOUNT_TYPES = ["checking", "savings", "credit", "business"]


@dataclass
class Account:
    account_id: str
    account_type: str
    age_days: int
    avg_monthly_spend: float
    region: str
    is_high_risk: bool


@dataclass
class Merchant:
    merchant_id: str
    category: str
    base_fraud_rate: float
    is_foreign: bool
    avg_txn_amount: float


@dataclass
class Transaction:
    txn_id: str
    account_id: str
    merchant_id: str
    amount: float
    merchant_category: str
    is_foreign_merchant: bool
    hour_of_day: int
    day_of_week: int
    txn_velocity_1h: int
    amount_vs_avg_ratio: float
    is_fraud: bool
    fraud_reason: str | None
    created_at: datetime


def generate_accounts(n: int, rng: np.random.Generator) -> list[Account]:
    accounts = []
    for _ in range(n):
        account_type = rng.choice(ACCOUNT_TYPES, p=[0.40, 0.25, 0.25, 0.10])
        avg_spend = float(rng.lognormal(mean=6.5, sigma=1.2))  # ~$665 avg
        accounts.append(
            Account(
                account_id=str(uuid.uuid4()),
                account_type=account_type,
                age_days=int(rng.integers(30, 3650)),
                avg_monthly_spend=avg_spend,
                region=str(rng.choice(REGIONS)),
                is_high_risk=bool(rng.random() < 0.05),
            )
        )
    return accounts


def generate_merchants(n: int, rng: np.random.Generator) -> list[Merchant]:
    categories = list(MERCHANT_CATEGORIES.keys())
    merchants = []
    for _ in range(n):
        category = str(rng.choice(categories))
        merchants.append(
            Merchant(
                merchant_id=str(uuid.uuid4()),
                category=category,
                base_fraud_rate=MERCHANT_CATEGORIES[category],
                is_foreign=bool(rng.random() < 0.15),
                avg_txn_amount=float(rng.lognormal(mean=4.5, sigma=1.5)),
            )
        )
    return merchants


def _compute_fraud_probability(
    account: Account,
    merchant: Merchant,
    amount: float,
    hour: int,
    velocity_1h: int,
    amount_ratio: float,
) -> tuple[float, str | None]:
    """Additive risk scoring to achieve ~2% fraud rate without blowup."""
    risk = merchant.base_fraud_rate

    # Account risk factors
    if account.is_high_risk:
        risk += 0.15
    if account.age_days < 90:
        risk += 0.05  # new accounts riskier

    # Transaction factors
    if merchant.is_foreign:
        risk += 0.03
    if amount > account.avg_monthly_spend * 0.5:  # large single txn
        risk += 0.04
    if amount_ratio > 5.0:  # way above average
        risk += 0.08
    if velocity_1h > 5:  # velocity spike
        risk += 0.10
    if hour in {0, 1, 2, 3, 4}:  # late night
        risk += 0.02

    # Cap at 0.95
    risk = min(risk, 0.95)

    if random.random() < risk:
        reasons = []
        if merchant.base_fraud_rate > 0.05:
            reasons.append(f"high_risk_merchant_{merchant.category}")
        if velocity_1h > 5:
            reasons.append("velocity_spike")
        if amount_ratio > 5.0:
            reasons.append("amount_anomaly")
        if merchant.is_foreign:
            reasons.append("foreign_merchant")
        return risk, "|".join(reasons) if reasons else "pattern_match"
    return risk, None


def generate_transactions(
    accounts: list[Account],
    merchants: list[Merchant],
    n_transactions: int,
    rng: np.random.Generator,
    start_date: datetime,
) -> list[Transaction]:
    transactions = []
    account_velocity: dict[str, list[datetime]] = {a.account_id: [] for a in accounts}

    print(f"Generating {n_transactions:,} transactions...")
    for i in range(n_transactions):
        if i % 100_000 == 0 and i > 0:
            print(f"  {i:,} / {n_transactions:,}")

        account = accounts[int(rng.integers(0, len(accounts)))]
        merchant = merchants[int(rng.integers(0, len(merchants)))]

        # Temporal patterns
        days_offset = float(rng.uniform(0, 365))
        txn_time = start_date + timedelta(days=days_offset)
        hour = txn_time.hour
        dow = txn_time.weekday()

        # Amount: lognormal around account average
        amount = float(rng.lognormal(
            mean=np.log(max(merchant.avg_txn_amount, 1)),
            sigma=0.8,
        ))
        amount = round(min(max(amount, 0.50), 50_000), 2)

        # Velocity: count recent transactions
        now = txn_time
        recent = [t for t in account_velocity[account.account_id] if (now - t).total_seconds() < 3600]
        velocity_1h = len(recent)
        account_velocity[account.account_id].append(now)
        if len(account_velocity[account.account_id]) > 50:
            account_velocity[account.account_id] = account_velocity[account.account_id][-50:]

        # Amount ratio vs account monthly average
        avg_txn = account.avg_monthly_spend / 30
        amount_ratio = amount / max(avg_txn, 1.0)

        _, fraud_reason = _compute_fraud_probability(
            account, merchant, amount, hour, velocity_1h, amount_ratio
        )
        is_fraud = fraud_reason is not None

        transactions.append(
            Transaction(
                txn_id=str(uuid.uuid4()),
                account_id=account.account_id,
                merchant_id=merchant.merchant_id,
                amount=amount,
                merchant_category=merchant.category,
                is_foreign_merchant=merchant.is_foreign,
                hour_of_day=hour,
                day_of_week=dow,
                txn_velocity_1h=velocity_1h,
                amount_vs_avg_ratio=round(amount_ratio, 4),
                is_fraud=is_fraud,
                fraud_reason=fraud_reason,
                created_at=txn_time,
            )
        )

    fraud_rate = sum(1 for t in transactions if t.is_fraud) / len(transactions)
    print(f"Generated {len(transactions):,} transactions | Fraud rate: {fraud_rate:.2%}")
    return transactions


def load_to_db(
    accounts: list[Account],
    merchants: list[Merchant],
    transactions: list[Transaction],
) -> None:
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    print("Loading accounts...")
    execute_values(
        cur,
        "INSERT INTO raw.accounts (account_id, account_type, age_days, avg_monthly_spend, region, is_high_risk) VALUES %s ON CONFLICT DO NOTHING",
        [(a.account_id, a.account_type, a.age_days, a.avg_monthly_spend, a.region, a.is_high_risk) for a in accounts],
    )
    print("Loading merchants...")
    execute_values(
        cur,
        "INSERT INTO raw.merchants (merchant_id, category, base_fraud_rate, is_foreign, avg_txn_amount) VALUES %s ON CONFLICT DO NOTHING",
        [(m.merchant_id, m.category, m.base_fraud_rate, m.is_foreign, m.avg_txn_amount) for m in merchants],
    )
    print("Loading transactions (batches of 10K)...")
    batch_size = 10_000
    for i in range(0, len(transactions), batch_size):
        batch = transactions[i : i + batch_size]
        execute_values(
            cur,
            """INSERT INTO raw.transactions
               (txn_id, account_id, merchant_id, amount, merchant_category,
                is_foreign_merchant, hour_of_day, day_of_week, txn_velocity_1h,
                amount_vs_avg_ratio, is_fraud, fraud_reason, created_at)
               VALUES %s ON CONFLICT DO NOTHING""",
            [
                (
                    t.txn_id, t.account_id, t.merchant_id, t.amount, t.merchant_category,
                    t.is_foreign_merchant, t.hour_of_day, t.day_of_week, t.txn_velocity_1h,
                    t.amount_vs_avg_ratio, t.is_fraud, t.fraud_reason, t.created_at,
                )
                for t in batch
            ],
        )
        if (i // batch_size) % 10 == 0:
            print(f"  Loaded {i + len(batch):,} / {len(transactions):,}")
    conn.commit()
    cur.close()
    conn.close()
    print("Database load complete.")


def save_to_parquet(
    accounts: list[Account],
    merchants: list[Merchant],
    transactions: list[Transaction],
    output_dir: str = "data/",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame([asdict(a) for a in accounts]).to_parquet(f"{output_dir}/accounts.parquet", index=False)
    pd.DataFrame([asdict(m) for m in merchants]).to_parquet(f"{output_dir}/merchants.parquet", index=False)
    pd.DataFrame([asdict(t) for t in transactions]).to_parquet(f"{output_dir}/transactions.parquet", index=False)
    print(f"Saved parquet files to {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic CreditPulse transaction data")
    parser.add_argument("--rows", type=int, default=1_000_000, help="Number of transactions")
    parser.add_argument("--accounts", type=int, default=50_000, help="Number of unique accounts")
    parser.add_argument("--merchants", type=int, default=5_000, help="Number of unique merchants")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--fast", action="store_true", help="Quick demo: 50K transactions")
    parser.add_argument("--load-db", action="store_true", help="Load data into PostgreSQL")
    parser.add_argument("--save-parquet", action="store_true", help="Save to Parquet files")
    args = parser.parse_args()

    if args.fast:
        args.rows = 50_000
        args.accounts = 5_000
        args.merchants = 500

    rng = np.random.default_rng(args.seed)
    start_date = datetime(2024, 1, 1)

    print(f"Generating {args.accounts:,} accounts, {args.merchants:,} merchants, {args.rows:,} transactions...")
    accounts = generate_accounts(args.accounts, rng)
    merchants = generate_merchants(args.merchants, rng)
    transactions = generate_transactions(accounts, merchants, args.rows, rng, start_date)

    if args.load_db:
        load_to_db(accounts, merchants, transactions)
    if args.save_parquet:
        save_to_parquet(accounts, merchants, transactions)
    if not args.load_db and not args.save_parquet:
        print("Tip: use --load-db or --save-parquet to persist the data.")


if __name__ == "__main__":
    main()
