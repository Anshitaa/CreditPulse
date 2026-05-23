"""
CreditPulse — WebSocket Load Test
Tests concurrent WebSocket clients against the /ws/scores endpoint.

Usage (headless, 50 concurrent users, 60s):
    locust -f tests/load/locustfile_ws.py \
           --headless -u 50 -r 5 -t 60s \
           --host http://127.0.0.1:8000 \
           --csv tests/load/results/ws_load_test

Usage (web UI — open http://localhost:8089):
    locust -f tests/load/locustfile_ws.py --host http://127.0.0.1:8000

Spec: CREDIT-001 NFR-001 (p99 WebSocket connection < 200ms)
      CREDIT-001 NFR-002 (support 100 concurrent WS clients without message loss)
"""

import json
import random
import time
import uuid

import websocket
from locust import HttpUser, TaskSet, between, events, task
from locust.contrib.fasthttp import FastHttpUser


# ── REST: score a transaction (also triggers a WS broadcast) ───────────────

class ScoreAndWatch(HttpUser):
    """
    Each virtual user:
      1. Opens a WebSocket connection to /ws/scores
      2. Concurrently POSTs transactions to /score (triggers broadcasts)
      3. Reads broadcast messages from the WebSocket
      4. Measures time-to-first-message (end-to-end latency)
    """
    wait_time = between(0.5, 2)
    host = "http://127.0.0.1:8000"

    def on_start(self):
        """Open a persistent WebSocket connection on user start."""
        ws_url = self.host.replace("http://", "ws://") + "/ws/scores"
        self.ws = websocket.create_connection(
            ws_url,
            timeout=5,
        )
        self.ws.settimeout(2)  # non-blocking reads

    def on_stop(self):
        """Close WebSocket on user teardown."""
        try:
            self.ws.close()
        except Exception:
            pass

    @task(3)
    def score_transaction_and_read_ws(self):
        """POST /score → read broadcast from WS → report latency."""
        txn = {
            "txn_id": f"T-LOAD-{uuid.uuid4().hex[:8]}",
            "account_id": f"ACC-{random.randint(1000, 9999)}",
            "merchant_id": f"MCH-{random.randint(100, 999)}",
            "amount": round(random.uniform(1.0, 5000.0), 2),
            "merchant_category": random.choice([
                "grocery", "online_retail", "wire_transfer",
                "atm_withdrawal", "gambling", "restaurant",
            ]),
            "is_foreign_merchant": random.random() < 0.15,
            "hour_of_day": random.randint(0, 23),
            "day_of_week": random.randint(0, 6),
            "txn_velocity_1h": random.randint(0, 20),
            "amount_vs_avg_ratio": round(random.uniform(0.1, 10.0), 2),
        }

        t0 = time.perf_counter()

        with self.client.post(
            "/score",
            json=txn,
            catch_response=True,
            name="POST /score",
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Score returned {resp.status_code}")
                return
            resp.success()

        # Try to read broadcast message — non-blocking
        try:
            msg = self.ws.recv()
            latency_ms = (time.perf_counter() - t0) * 1000
            data = json.loads(msg)
            events.request.fire(
                request_type="WS",
                name="ws/scores broadcast",
                response_time=latency_ms,
                response_length=len(msg),
                exception=None,
                context={},
            )
        except websocket.WebSocketTimeoutException:
            # Didn't receive a broadcast within timeout — log as miss
            events.request.fire(
                request_type="WS",
                name="ws/scores broadcast",
                response_time=2000,
                response_length=0,
                exception=Exception("No broadcast received within 2s"),
                context={},
            )

    @task(1)
    def keep_alive_ping(self):
        """Send a ping to keep the WS connection alive."""
        try:
            self.ws.ping()
        except Exception:
            pass

    @task(2)
    def score_only(self):
        """Pure REST throughput — no WS read."""
        txn = {
            "txn_id": f"T-LOAD-{uuid.uuid4().hex[:8]}",
            "account_id": f"ACC-{random.randint(1000, 9999)}",
            "merchant_id": f"MCH-{random.randint(100, 999)}",
            "amount": round(random.uniform(5.0, 2000.0), 2),
            "merchant_category": random.choice(["grocery", "retail", "online_retail"]),
            "is_foreign_merchant": False,
            "hour_of_day": random.randint(8, 20),
            "day_of_week": random.randint(0, 4),
            "txn_velocity_1h": random.randint(0, 5),
            "amount_vs_avg_ratio": round(random.uniform(0.5, 3.0), 2),
        }
        with self.client.post(
            "/score",
            json=txn,
            catch_response=True,
            name="POST /score (REST only)",
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Score returned {resp.status_code}")


# ── WebSocket-only stress test (no REST) ───────────────────────────────────

class WebSocketOnlyUser(HttpUser):
    """
    Simulates dashboard clients that only hold a WS connection open.
    Tests whether the server handles 100 idle WS connections without leak.
    """
    wait_time = between(10, 30)
    weight = 1  # lower weight than ScoreAndWatch

    def on_start(self):
        ws_url = self.host.replace("http://", "ws://") + "/ws/scores"
        self.ws = websocket.create_connection(ws_url, timeout=5)
        self.ws.settimeout(0.1)

    def on_stop(self):
        try:
            self.ws.close()
        except Exception:
            pass

    @task
    def idle_listen(self):
        """Just keep the connection open — measure server idle connection overhead."""
        try:
            self.ws.recv()
        except websocket.WebSocketTimeoutException:
            pass
