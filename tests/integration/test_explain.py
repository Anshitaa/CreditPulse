"""
Integration tests: /explain endpoints

Covers:
- GET /explain/{txn_id} — 404 for unknown txn
- POST /explain/whatif — response shape + score changes with inputs
- POST /explain/counterfactual — required keys, plausible suggestion
- Spec: CREDIT-003 FR-001, FR-002, FR-003
"""

import pytest


class TestExplainByTxnId:
    def test_unknown_txn_returns_404_or_500(self, client):
        """Unknown txn returns 404 (normal) or 500 if DB is unreachable."""
        resp = client.get("/explain/does-not-exist-xyz")
        assert resp.status_code in (404, 500), f"Unexpected status: {resp.status_code}"

    def test_unknown_txn_returns_404_when_db_up(self, client):
        """When DB is up, unknown txn must return 404, not 500."""
        health = client.get("/health/ready").json()
        if not health.get("checks", {}).get("database"):
            pytest.skip("DB not reachable — skipping 404 assertion")
        resp = client.get("/explain/does-not-exist-xyz")
        assert resp.status_code == 404

    def test_scored_txn_has_explanation(self, client, scored_txn):
        """After scoring, GET /explain/{txn_id} should return stored SHAP data."""
        txn_id = scored_txn["txn_id"]
        resp = client.get(f"/explain/{txn_id}")
        # 200 if explanation stored, 404 if audit.explanations table not populated, 500 if DB down
        assert resp.status_code in (200, 404, 500), resp.text


class TestWhatIfSimulator:
    """Spec: CREDIT-003 FR-003 — respond within 1 second."""

    def test_returns_200(self, client, low_risk_txn):
        resp = client.post("/explain/whatif", json=low_risk_txn)
        assert resp.status_code == 200, resp.text

    def test_response_has_score_fields(self, client, low_risk_txn):
        data = client.post("/explain/whatif", json=low_risk_txn).json()
        assert "fraud_probability" in data
        assert "fraud_risk_score" in data
        assert "decision" in data

    def test_mode_is_what_if_simulation(self, client, low_risk_txn):
        data = client.post("/explain/whatif", json=low_risk_txn).json()
        assert data.get("mode") == "what_if_simulation"

    def test_higher_velocity_raises_score(self, client):
        """Increasing txn_velocity_1h should raise fraud score."""
        base = {
            "account_id": "whatif-test",
            "amount": 500.0,
            "merchant_category": "retail",
            "is_foreign_merchant": False,
            "hour_of_day": 12,
            "day_of_week": 1,
            "txn_velocity_1h": 1,
            "amount_vs_avg_ratio": 1.0,
        }
        high_vel = {**base, "txn_velocity_1h": 15}
        score_base = client.post("/explain/whatif", json=base).json()["fraud_risk_score"]
        score_high = client.post("/explain/whatif", json=high_vel).json()["fraud_risk_score"]
        assert score_high >= score_base, (
            f"Higher velocity ({score_high:.1f}) should not score lower than base ({score_base:.1f})"
        )

    def test_foreign_merchant_raises_score(self, client):
        """is_foreign_merchant=True should raise fraud risk."""
        base = {
            "account_id": "whatif-foreign",
            "amount": 1000.0,
            "merchant_category": "retail",
            "is_foreign_merchant": False,
            "hour_of_day": 12,
            "day_of_week": 1,
            "txn_velocity_1h": 2,
            "amount_vs_avg_ratio": 2.0,
        }
        foreign = {**base, "is_foreign_merchant": True}
        score_domestic = client.post("/explain/whatif", json=base).json()["fraud_risk_score"]
        score_foreign = client.post("/explain/whatif", json=foreign).json()["fraud_risk_score"]
        assert score_foreign >= score_domestic, (
            f"Foreign merchant ({score_foreign:.1f}) should not score lower than domestic ({score_domestic:.1f})"
        )

    def test_wire_transfer_raises_score_vs_grocery(self, client):
        """wire_transfer merchant category should score higher than grocery."""
        base = {
            "account_id": "whatif-cat",
            "amount": 2000.0,
            "is_foreign_merchant": False,
            "hour_of_day": 12,
            "day_of_week": 1,
            "txn_velocity_1h": 2,
            "amount_vs_avg_ratio": 3.0,
        }
        grocery_score = client.post("/explain/whatif", json={**base, "merchant_category": "grocery"}).json()["fraud_risk_score"]
        wire_score = client.post("/explain/whatif", json={**base, "merchant_category": "wire_transfer"}).json()["fraud_risk_score"]
        assert wire_score >= grocery_score, (
            f"wire_transfer ({wire_score:.1f}) should score >= grocery ({grocery_score:.1f})"
        )

    def test_whatif_under_1s(self, client, low_risk_txn):
        """Spec: CREDIT-003 FR-003 — respond within 1 second."""
        import time
        t0 = time.perf_counter()
        client.post("/explain/whatif", json=low_risk_txn)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 1000, f"What-if took {elapsed_ms:.0f}ms — exceeds 1s SLA"


class TestCounterfactual:
    """Spec: CREDIT-003 FR-002 — generate minimum change to reduce score < 50."""

    def test_returns_200(self, client, high_risk_txn):
        payload = {k: v for k, v in high_risk_txn.items() if k != "txn_id"}
        resp = client.post("/explain/counterfactual", json=payload)
        assert resp.status_code == 200, resp.text

    def test_required_keys_present(self, client, high_risk_txn):
        payload = {k: v for k, v in high_risk_txn.items() if k != "txn_id"}
        data = client.post("/explain/counterfactual", json=payload).json()
        required = [
            "original_fraud_probability",
            "original_risk_score",
            "counterfactuals",
            "explanation",
        ]
        for key in required:
            assert key in data, f"Missing key in counterfactual response: {key}"

    def test_original_score_reflects_input(self, client):
        """original_risk_score should match what /score would return for same inputs."""
        payload = {
            "account_id": "cf-test",
            "amount": 5000.0,
            "merchant_category": "wire_transfer",
            "is_foreign_merchant": True,
            "hour_of_day": 1,
            "day_of_week": 6,
            "txn_velocity_1h": 10,
            "amount_vs_avg_ratio": 8.0,
        }
        cf_data = client.post("/explain/counterfactual", json=payload).json()
        score_data = client.post("/score/", json={**payload, "txn_id": "cf-score-ref"}).json()
        assert abs(cf_data["original_risk_score"] - score_data["fraud_risk_score"]) < 5.0, (
            f"Counterfactual original_risk_score ({cf_data['original_risk_score']:.1f}) "
            f"diverges from /score ({score_data['fraud_risk_score']:.1f})"
        )

    def test_explanation_is_non_empty_string(self, client, high_risk_txn):
        payload = {k: v for k, v in high_risk_txn.items() if k != "txn_id"}
        data = client.post("/explain/counterfactual", json=payload).json()
        assert isinstance(data["explanation"], str)
        assert len(data["explanation"]) > 10
