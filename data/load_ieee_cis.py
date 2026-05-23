"""
CreditPulse — IEEE-CIS Fraud Detection Dataset Loader
Dataset: https://www.kaggle.com/c/ieee-fraud-detection

Downloads and ETLs the Vesta Corporation transaction data into PostgreSQL.
- 590,540 transactions (train_transaction.csv)
- 144,233 identity records (train_identity.csv)
- isFraud label: 3.5% fraud rate (realistic for financial systems)

Usage:
    # Download via Kaggle API (requires ~/.kaggle/kaggle.json):
    python data/load_ieee_cis.py --download --load-db

    # If already downloaded:
    python data/load_ieee_cis.py --load-db --data-dir data/ieee_cis/

    # Just download, don't load:
    python data/load_ieee_cis.py --download

Feature strategy:
    We select ~50 interpretable features from the 434 available columns.
    V-features (Vesta-engineered) are kept as opaque but included because
    they dominate the AUC signal. Top-30 by SHAP importance are exposed
    in the model risk card.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://creditpulse:creditpulse@localhost:5435/creditpulse",
)

DATA_DIR = Path("data/ieee_cis")

# ── Feature selection ──────────────────────────────────────────────────────
# Selected for: interpretability + coverage of key fraud signals
# Full feature list: https://www.kaggle.com/c/ieee-fraud-detection/data

TRANSACTION_FEATURES = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",           # product type: W, H, C, S, R
    "card1",               # card info (anonymized)
    "card2",
    "card3",
    "card4",               # card network: visa, mastercard, amex, discover
    "card5",
    "card6",               # credit vs debit
    "addr1",               # billing region code
    "addr2",               # billing country code
    "dist1",               # distance (home to transaction)
    "dist2",
    "P_emaildomain",       # purchaser email domain
    "R_emaildomain",       # recipient email domain
    # C-features: count fields (how many addresses, cards, etc. associated)
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "C11", "C12", "C13", "C14",
    # D-features: timedelta fields (days between events)
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9",
    "D10", "D11", "D12", "D13", "D14", "D15",
    # M-features: match fields (T/F)
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    # V-features: Vesta-engineered (top-50 by variance; rest dropped)
    *[f"V{i}" for i in [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        12, 13, 14, 15, 17, 19, 20,
        29, 30, 33, 34, 35, 36, 37, 38,
        44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
        56, 57, 58, 61, 62, 63, 64, 65, 67, 69, 70,
        75, 76, 78, 79, 80, 81, 82, 83, 87, 90, 91, 94,
        95, 96, 97, 98, 99, 100,
        126, 127, 128, 129, 130, 131,
        139, 140, 143, 145,
        150, 151, 152, 153, 158, 159, 160, 161, 162,
        165, 166, 167, 168, 169, 170,
        187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198,
        201, 202, 207, 209, 210, 211, 212, 213, 214, 215, 216, 217,
        218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229,
        230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
        243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254,
        255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266,
        267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278,
        279, 280, 281, 282, 283, 284, 285, 287, 289, 291,
        292, 293, 294, 295, 296, 297, 298, 299, 300,
        306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317,
        318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329,
        330, 331, 332, 333, 334, 335, 336, 337, 338, 339,
    ]],
]

IDENTITY_FEATURES = [
    "TransactionID",
    "DeviceType",          # desktop vs mobile
    "DeviceInfo",          # device string (browser/OS)
    *[f"id_{i:02d}" for i in range(1, 39)],  # id_01 to id_38
]


# ── Download ───────────────────────────────────────────────────────────────

def download_dataset(data_dir: Path = DATA_DIR) -> None:
    """Download IEEE-CIS dataset via Kaggle API."""
    data_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json = Path.home() / ".kaggle/kaggle.json"
    if not kaggle_json.exists():
        print(
            "ERROR: Kaggle credentials not found.\n"
            "1. Go to kaggle.com → Settings → API → Create New Token\n"
            "2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/\n"
            "3. chmod 600 ~/.kaggle/kaggle.json\n"
            "4. Then re-run this script."
        )
        sys.exit(1)

    print(f"Downloading IEEE-CIS dataset to {data_dir}/...")
    result = subprocess.run(
        [
            sys.executable, "-m", "kaggle",
            "competitions", "download",
            "-c", "ieee-fraud-detection",
            "-p", str(data_dir),
        ],
        capture_output=False,
    )
    if result.returncode != 0:
        print(
            "\nNote: You may need to accept the competition rules at:\n"
            "https://www.kaggle.com/c/ieee-fraud-detection/rules"
        )
        sys.exit(1)

    # Unzip
    import zipfile
    zip_path = data_dir / "ieee-fraud-detection.zip"
    if zip_path.exists():
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
        zip_path.unlink()
    print(f"Download complete. Files in {data_dir}/")


# ── Load & clean ───────────────────────────────────────────────────────────

def load_and_merge(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load, merge, and clean the IEEE-CIS dataset."""
    txn_path = data_dir / "train_transaction.csv"
    idn_path = data_dir / "train_identity.csv"

    if not txn_path.exists():
        raise FileNotFoundError(
            f"train_transaction.csv not found at {txn_path}.\n"
            "Run: python data/load_ieee_cis.py --download"
        )

    print(f"Loading train_transaction.csv ({txn_path.stat().st_size / 1e6:.0f} MB)...")
    # Only read columns we need — avoids loading all 394 V-columns
    usecols = [c for c in TRANSACTION_FEATURES if c != "TransactionID" or True]
    available_txn_cols = pd.read_csv(txn_path, nrows=0).columns.tolist()
    read_txn_cols = [c for c in TRANSACTION_FEATURES if c in available_txn_cols]
    txn = pd.read_csv(txn_path, usecols=read_txn_cols)
    print(f"  Loaded {len(txn):,} transactions, {len(read_txn_cols)} columns")

    if idn_path.exists():
        print(f"Loading train_identity.csv ({idn_path.stat().st_size / 1e6:.0f} MB)...")
        available_idn_cols = pd.read_csv(idn_path, nrows=0).columns.tolist()
        read_idn_cols = [c for c in IDENTITY_FEATURES if c in available_idn_cols]
        idn = pd.read_csv(idn_path, usecols=read_idn_cols)
        print(f"  Loaded {len(idn):,} identity records, {len(read_idn_cols)} columns")
        df = txn.merge(idn, on="TransactionID", how="left")
    else:
        print("  train_identity.csv not found — skipping identity join")
        df = txn

    print(f"Merged shape: {df.shape}")
    print(f"Fraud rate: {df['isFraud'].mean():.2%}  ({df['isFraud'].sum():,} fraud)")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features on top of raw IEEE-CIS columns."""
    # Temporal: TransactionDT is seconds from a reference date
    df["hour_of_day"] = (df["TransactionDT"] // 3600) % 24
    df["day_of_week"] = (df["TransactionDT"] // 86400) % 7

    # Amount log-transform (right-skewed)
    df["log_amount"] = np.log1p(df["TransactionAmt"])

    # Email domain features
    df["email_match"] = (df["P_emaildomain"] == df["R_emaildomain"]).astype(int)
    df["purchaser_gmail"] = (df["P_emaildomain"] == "gmail.com").astype(int)

    # Card type binary
    df["is_credit"] = (df["card6"] == "credit").astype(int)
    df["is_debit"] = (df["card6"] == "debit").astype(int)

    # M-feature booleans → int
    for col in [f"M{i}" for i in range(1, 10)]:
        if col in df.columns:
            df[col] = df[col].map({"T": 1, "F": 0}).fillna(-1).astype(int)

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categoricals, fill NaN, drop low-variance columns."""
    # Label-encode object columns
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        if col == "TransactionID":
            continue
        df[col] = le.fit_transform(df[col].astype(str).fillna("__NAN__"))

    # Fill remaining NaN with -999 (XGBoost handles this natively)
    df = df.fillna(-999)

    # Drop near-constant columns (< 0.001 variance)
    num_cols = df.select_dtypes(include="number").columns.difference(
        ["TransactionID", "isFraud"]
    )
    variances = df[num_cols].var()
    drop_cols = variances[variances < 0.001].index.tolist()
    if drop_cols:
        print(f"Dropping {len(drop_cols)} near-zero-variance columns")
        df = df.drop(columns=drop_cols)

    return df


# ── Load to DB ─────────────────────────────────────────────────────────────

CREATE_IEEE_TABLE = """
CREATE TABLE IF NOT EXISTS raw.ieee_cis_transactions (
    transaction_id      BIGINT PRIMARY KEY,
    is_fraud            BOOLEAN NOT NULL,
    transaction_dt      INTEGER,
    transaction_amt     FLOAT,
    product_cd          TEXT,
    card4               TEXT,
    card6               TEXT,
    addr1               FLOAT,
    addr2               FLOAT,
    dist1               FLOAT,
    p_emaildomain       TEXT,
    r_emaildomain       TEXT,
    hour_of_day         INTEGER,
    day_of_week         INTEGER,
    log_amount          FLOAT,
    email_match         INTEGER,
    is_credit           INTEGER,
    features_json       JSONB,   -- all other features as JSON blob
    loaded_at           TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_ieee_is_fraud ON raw.ieee_cis_transactions(is_fraud);
CREATE INDEX IF NOT EXISTS idx_ieee_dt ON raw.ieee_cis_transactions(transaction_dt);
"""


def load_to_db(df: pd.DataFrame, batch_size: int = 5000) -> None:
    """Write IEEE-CIS data to PostgreSQL."""
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute(CREATE_IEEE_TABLE)
    conn.commit()

    # Determine which columns go in dedicated columns vs. JSON blob
    dedicated = {
        "TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
        "ProductCD", "card4", "card6", "addr1", "addr2", "dist1",
        "P_emaildomain", "R_emaildomain",
        "hour_of_day", "day_of_week", "log_amount", "email_match", "is_credit",
    }
    feature_cols = [c for c in df.columns if c not in dedicated and c != "TransactionID"]

    print(f"Loading {len(df):,} rows to raw.ieee_cis_transactions in batches of {batch_size}...")

    import json as _json
    rows = []
    for _, row in df.iterrows():
        feature_blob = {c: (None if row[c] == -999 else float(row[c])) for c in feature_cols}
        rows.append((
            int(row["TransactionID"]),
            bool(row["isFraud"]),
            int(row.get("TransactionDT", -999)),
            float(row.get("TransactionAmt", 0)),
            str(row.get("ProductCD", "")),
            str(row.get("card4", "")),
            str(row.get("card6", "")),
            float(row.get("addr1", -999)) if row.get("addr1", -999) != -999 else None,
            float(row.get("addr2", -999)) if row.get("addr2", -999) != -999 else None,
            float(row.get("dist1", -999)) if row.get("dist1", -999) != -999 else None,
            str(row.get("P_emaildomain", "")),
            str(row.get("R_emaildomain", "")),
            int(row.get("hour_of_day", 0)),
            int(row.get("day_of_week", 0)),
            float(row.get("log_amount", 0)),
            int(row.get("email_match", 0)),
            int(row.get("is_credit", 0)),
            _json.dumps(feature_blob),
        ))

        if len(rows) >= batch_size:
            _insert_batch(cur, rows)
            conn.commit()
            print(f"  Inserted {batch_size} rows...")
            rows = []

    if rows:
        _insert_batch(cur, rows)
        conn.commit()

    cur.close()
    conn.close()

    # Verify
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*), SUM(is_fraud::int) FROM raw.ieee_cis_transactions")
    total, fraud_count = cur.fetchone()
    conn.close()
    print(f"\nLoaded {total:,} transactions — {fraud_count:,} fraud ({fraud_count/total:.2%})")


def _insert_batch(cur, rows: list) -> None:
    execute_values(cur, """
        INSERT INTO raw.ieee_cis_transactions (
            transaction_id, is_fraud, transaction_dt, transaction_amt,
            product_cd, card4, card6, addr1, addr2, dist1,
            p_emaildomain, r_emaildomain, hour_of_day, day_of_week,
            log_amount, email_match, is_credit, features_json
        ) VALUES %s
        ON CONFLICT (transaction_id) DO NOTHING
    """, rows)


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IEEE-CIS Fraud Dataset ETL")
    parser.add_argument("--download", action="store_true", help="Download from Kaggle")
    parser.add_argument("--load-db", action="store_true", help="Load to PostgreSQL")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Path to CSV files")
    parser.add_argument("--sample", type=int, default=None,
                        help="Load only first N rows (for quick testing)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.download:
        download_dataset(data_dir)

    if args.load_db:
        df = load_and_merge(data_dir)
        if args.sample:
            df = df.head(args.sample)
            print(f"Using sample of {args.sample:,} rows")
        df = engineer_features(df)
        df = clean(df)
        load_to_db(df)
        print("\nDone. Now retrain the model:")
        print("  python models/fraud_detector.py --train --ieee-cis")
