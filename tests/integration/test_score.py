"""
Integration tests: /score endpoint

Covers:
- Response schema (all required fields present)
- Latency SLA: p99 < 100ms (Spec: CREDIT-001 NFR-001)
- Decision thresholds fire correctly
- High-risk txn triggers FRAUD or REVIEW decision
- Audit trail: score written to audit.model_decisions (Spec: CREDIT-001 FR-003)
- Audit trail: score written to mart.risk_scores (dashboard source)
- SHAP top_features: exactly 5 features with direction
- X-Latency-MS header present on every response
"""

import time
import statistics
import pytest
import psycopg2
from psycopg2.extras import RealDictCursor


SCORE_URL = "/score/"


class TestScoreResponseSchema:
    def test_required_keys_present(self, client, high_risk_txn):
        resp = client.post(SCORE_URL, json=high_risk_txn)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        required = [
            "txn_id", "fraud_probability", "fraud_risk_score",
            "decision", "top_features", "inference_latency_ms",
            "scored_at", "spec_ref",
        ]
        for key in required:
            assert key in data, f"Missing key in /score response: {key}"

    def test_fraud_probability_in_range(self, client, high_risk_txn):
        data = client.post(SCORE_URL, json=high_risk_txn).json()
        assert 0.0 <= data["fraud_probability"] <= 1.0

    def test_risk_score_in_range(self, client, high_risk_txn):
        data = client.post(SCORE_URL, json=high_risk_txn).json()
        assert 0.0 <= data["fraud_risk_score"] <= 100.0

    def test_decision_is_valid_enum(self, client, high_risk_txn):
        data = client.post(SCORE_URL, json=high_risk_txn).json()
        assert data["decision"] in ("FRAUD", "REVIEW", "CLEAR")

    def test_spec_ref_is_credit_001(self, client, high_risk_txn):
        data = client.post(SCORE_URL, json=high_risk_txn).json()
        assert data["spec_ref"] == "CREDIT-001"

    def test_top_features_count(self, client, high_risk_txn):
        """SHAP must return exactly 5 top features."""
        data = client.post(SCORE_URL, json=high_risk_txn).json()
        assert len(data["top_features"]) == 5

    def test_top_features_have_direction(self, client, high_risk_txn):
        data = client.post(SCORE_URL, json=high_risk_txn).json()
        for feat in data["top_features"]:
            assert "feature" in feat
            assert "shap_value" in feat
            assert "direction" in feat
            assert feat["direction"] in ("increases_risk", "decreases_risk")

    def test_txn_id_auto_generated_when_omitted(self, client, high_risk_txn):
        """txn_id should be auto-assigned if not provided."""
        data = client.post(SCORE_URL, json=high_risk_txn).json()
        assert data["txn_id"] is not None
        assert len(data["txn_id"]) > 0

    def test_txn_id_passthrough(self, client, high_risk_txn):
        payload = {**high_risk_txn, "txn_id": "inttest-explicit-id"}
        data = client.post(SCORE_URL, json=payload).json()
        assert data["txn_id"] == "inttest-explicit-id"


class TestScoreDecisionThresholds:
    def test_high_risk_is_not_clear(self, client, high_risk_txn):
        """Wire transfer at 2 AM, 15x avg — should not be cleared."""
        data = client.post(SCORE_URL, json=high_risk_txn).json()
        assert data["decision"] in ("FRAUD", "REVIEW"), (
            f"High-risk txn scored {data['fraud_risk_score']:.1f} but got CLEAR"
        )

    def test_low_risk_score_is_lower(self, client, high_risk_txn, low_risk_txn):
        """Low-risk grocery txn should score lower than high-risk wire transfer."""
        high = client.post(SCORE_URL, json=high_risk_txn).json()["fraud_risk_score"]
        low = client.post(SCORE_URL, json=low_risk_txn).json()["fraud_risk_score"]
        assert high > low, (
            f"High-risk ({high:.1f}) should outscore low-risk ({low:.1f})"
        )

    def test_validation_rejects_negative_amount(self, client):
        payload = {
            "account_id": "x", "amount": -1.0, "merchant_category": "retail",
            "is_foreign_merchant": False, "hour_of_day": 12, "day_of_week": 0,
            "txn_velocity_1h": 0, "amount_vs_avg_ratio": 1.0,
        }
        resp = client.post(SCORE_URL, json=payload)
        assert resp.status_code == 422

    def test_validation_rejects_amount_over_limit(self, client):
        payload = {
            "account_id": "x", "amount": 200_000.0, "merchant_category": "retail",
            "is_foreign_merchant": False, "hour_of_day": 12, "day_of_week": 0,
            "txn_velocity_1h": 0, "amount_vs_avg_ratio": 1.0,
        }
        resp = client.post(SCORE_URL, json=payload)
        assert resp.status_code == 422


class TestScoreLatencySLA:
    """Spec: CREDIT-001 NFR-001 — p99 inference latency < 100ms."""

    def test_single_request_under_100ms(self, client, high_risk_txn):
        data = client.post(SCORE_URL, json=high_risk_txn).json()
        assert data["inference_latency_ms"] < 100, (
            f"Single inference took {data['inference_latency_ms']:.1f}ms — exceeds 100ms SLA"
        )

    def test_p99_latency_under_100ms(self, client, high_risk_txn):
        """Score 20 transactions and check p99 < 100ms."""
        latencies = []
        for _ in range(20):
            data = client.post(SCORE_URL, json=high_risk_txn).json()
            latencies.append(data["inference_latency_ms"])
        p99 = sorted(latencies)[int(0.99 * len(latencies))]
        avg = statistics.mean(latencies)
        assert p99 < 100, (
            f"p99 latency {p99:.1f}ms exceeds 100ms SLA. avg={avg:.1f}ms, all={[f'{l:.1f}' for l in latencies]}"
        )

    def test_latency_header_is_valid_number(self, client, high_risk_txn):
        # X-Latency-MS = total request time (model + DB audit write + Kafka publish)
        # inference_latency_ms in body = XGBoost inference only — they are intentionally different
        resp = client.post(SCORE_URL, json=high_risk_txn)
        assert "X-Latency-MS" in resp.headers
        header_ms = float(resp.headers["X-Latency-MS"])
        assert header_ms > 0, "X-Latency-MS should be positive"


class TestScoreAuditTrail:
    """Spec: CREDIT-001 FR-003 — scored transaction must appear in audit log."""

    def test_score_written_to_audit_decisions(self, client, db, high_risk_txn):
        txn_id = f"inttest-audit-{int(time.time())}"
        client.post(SCORE_URL, json={**high_risk_txn, "txn_id": txn_id})
        cur = db.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT * FROM audit.model_decisions WHERE txn_id = %s", (txn_id,)
        )
        row = cur.fetchone()
        cur.close()
        assert row is not None, f"txn_id {txn_id} not found in audit.model_decisions"
        assert row["decision"] in ("FRAUD", "REVIEW", "CLEAR")
        assert row["score"] is not None

    def test_score_written_to_mart_risk_scores(self, client, db, high_risk_txn):
        txn_id = f"inttest-mart-{int(time.time())}"
        client.post(SCORE_URL, json={**high_risk_txn, "txn_id": txn_id})
        cur = db.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT * FROM mart.risk_scores WHERE txn_id = %s", (txn_id,)
        )
        row = cur.fetchone()
        cur.close()
        assert row is not None, f"txn_id {txn_id} not found in mart.risk_scores (dashboard data missing)"
        assert row["account_id"] == high_risk_txn["account_id"]
        assert row["composite_risk_score"] is not None

    def test_audit_decision_matches_api_response(self, client, db, high_risk_txn):
        txn_id = f"inttest-match-{int(time.time())}"
        api_resp = client.post(SCORE_URL, json={**high_risk_txn, "txn_id": txn_id}).json()
        cur = db.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT decision, score FROM audit.model_decisions WHERE txn_id = %s", (txn_id,)
        )
        row = cur.fetchone()
        cur.close()
        assert row["decision"] == api_resp["decision"]
        assert abs(float(row["score"]) - api_resp["fraud_risk_score"]) < 0.01
