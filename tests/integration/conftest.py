"""
CreditPulse — Integration Test Configuration

Fixtures for testing against the live stack:
  API  → http://127.0.0.1:8000 (uvicorn)
  DB   → postgresql://localhost:5435 (Docker)
  Kafka → localhost:9092 (Docker)

Run: pytest tests/integration/ -v
Prerequisites: API + Docker stack must be running.
"""

import os
import pytest
import httpx
import psycopg2
from psycopg2.extras import RealDictCursor

BASE_URL = os.environ.get("CREDITPULSE_API_URL", "http://127.0.0.1:8000")
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://creditpulse:creditpulse@localhost:5435/creditpulse",
)


@pytest.fixture(scope="session")
def client():
    """HTTP client pointed at the live API."""
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as c:
        yield c


@pytest.fixture(scope="session")
def db():
    """Live PostgreSQL connection (port 5435)."""
    try:
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
        yield conn
        conn.close()
    except psycopg2.OperationalError as e:
        pytest.skip(f"PostgreSQL not reachable: {e}")


@pytest.fixture
def high_risk_txn():
    """Wire transfer at 2 AM, 12x normal amount — should score FRAUD or REVIEW."""
    return {
        "account_id": "inttest-high-risk",
        "amount": 9500.0,
        "merchant_category": "wire_transfer",
        "is_foreign_merchant": True,
        "hour_of_day": 2,
        "day_of_week": 6,
        "txn_velocity_1h": 12,
        "amount_vs_avg_ratio": 15.0,
    }


@pytest.fixture
def low_risk_txn():
    """Grocery purchase at midday — should score CLEAR."""
    return {
        "account_id": "inttest-low-risk",
        "amount": 42.0,
        "merchant_category": "grocery",
        "is_foreign_merchant": False,
        "hour_of_day": 14,
        "day_of_week": 2,
        "txn_velocity_1h": 1,
        "amount_vs_avg_ratio": 0.8,
    }


@pytest.fixture(scope="session")
def scored_txn(client):
    """Score one transaction at session start; returns the full response dict."""
    payload = {
        "account_id": "inttest-explain-acct",
        "amount": 6000.0,
        "merchant_category": "wire_transfer",
        "is_foreign_merchant": True,
        "hour_of_day": 3,
        "day_of_week": 0,
        "txn_velocity_1h": 8,
        "amount_vs_avg_ratio": 10.0,
    }
    resp = client.post("/score/", json=payload)
    assert resp.status_code == 200, f"Pre-scoring failed: {resp.text}"
    return resp.json()
