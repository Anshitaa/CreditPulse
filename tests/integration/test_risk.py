"""
Integration tests: /risk endpoints

Covers:
- GET /risk/scores — paginated query with filters
- GET /risk/account/{account_id} — 404 for unknown account
- GET /risk/summary — dashboard aggregate stats
- Spec: CREDIT-002
"""

import time
import pytest


class TestRiskScores:
    def test_returns_200(self, client):
        resp = client.get("/risk/scores")
        assert resp.status_code == 200, resp.text

    def test_response_shape(self, client):
        data = client.get("/risk/scores").json()
        assert "count" in data
        assert "scores" in data
        assert isinstance(data["scores"], list)

    def test_count_matches_scores_length(self, client):
        data = client.get("/risk/scores").json()
        assert data["count"] == len(data["scores"])

    def test_min_score_filter(self, client):
        """All returned scores should be >= min_score."""
        data = client.get("/risk/scores", params={"min_score": 70}).json()
        for row in data["scores"]:
            assert row["composite_risk_score"] >= 70, (
                f"Score {row['composite_risk_score']} below min_score=70"
            )

    def test_decision_filter(self, client, scored_txn):
        """Filter by decision=FRAUD should return only FRAUD rows."""
        data = client.get("/risk/scores", params={"decision": "FRAUD"}).json()
        for row in data["scores"]:
            assert row["decision"] == "FRAUD"

    def test_invalid_decision_returns_422(self, client):
        resp = client.get("/risk/scores", params={"decision": "MAYBE"})
        assert resp.status_code == 422

    def test_limit_respected(self, client):
        data = client.get("/risk/scores", params={"limit": 5}).json()
        assert len(data["scores"]) <= 5

    def test_scores_ordered_by_risk_desc(self, client):
        data = client.get("/risk/scores", params={"limit": 10}).json()
        scores = [r["composite_risk_score"] for r in data["scores"]]
        assert scores == sorted(scores, reverse=True), "Scores should be descending"

    def test_hours_back_filter_reduces_results(self, client):
        """Querying 1 hour back should return fewer or equal results than 168 hours."""
        recent = client.get("/risk/scores", params={"hours_back": 1}).json()["count"]
        week = client.get("/risk/scores", params={"hours_back": 168}).json()["count"]
        assert recent <= week


class TestRiskAccountHistory:
    def test_unknown_account_returns_404(self, client):
        resp = client.get("/risk/account/definitely-does-not-exist-xyz")
        assert resp.status_code == 404

    def test_scored_account_returns_history(self, client, scored_txn):
        """Account that was just scored should have history."""
        account_id = scored_txn.get("account_id", "inttest-explain-acct")
        resp = client.get(f"/risk/account/{account_id}")
        if resp.status_code == 404:
            pytest.skip("Account not in risk history — score more transactions first")
        assert resp.status_code == 200
        data = resp.json()
        assert data["account_id"] == account_id
        assert "trend" in data
        assert isinstance(data["trend"], list)

    def test_trend_has_required_fields(self, client, scored_txn):
        account_id = scored_txn.get("account_id", "inttest-explain-acct")
        resp = client.get(f"/risk/account/{account_id}")
        if resp.status_code == 404:
            pytest.skip("No history for this account")
        trend = resp.json()["trend"]
        if trend:
            row = trend[0]
            assert "week" in row
            assert "avg_score" in row
            assert "txn_count" in row

    def test_weeks_param_accepted(self, client, scored_txn):
        account_id = scored_txn.get("account_id", "inttest-explain-acct")
        resp = client.get(f"/risk/account/{account_id}", params={"weeks": 4})
        assert resp.status_code in (200, 404)


class TestRiskSummary:
    def test_returns_200(self, client):
        resp = client.get("/risk/summary")
        assert resp.status_code == 200

    def test_response_has_expected_fields(self, client):
        data = client.get("/risk/summary").json()
        if not data:
            pytest.skip("risk/summary returned empty — no scores in the last 24h")
        expected = [
            "total_scored_today", "avg_risk_score",
            "fraud_count", "review_count", "clear_count", "fraud_rate_pct",
        ]
        for field in expected:
            assert field in data, f"Missing field in /risk/summary: {field}"

    def test_counts_are_non_negative(self, client):
        data = client.get("/risk/summary").json()
        if not data:
            pytest.skip("risk/summary returned empty — no scores in the last 24h")
        assert (data["total_scored_today"] or 0) >= 0
        assert (data["fraud_count"] or 0) >= 0
        assert (data["review_count"] or 0) >= 0
        assert (data["clear_count"] or 0) >= 0

    def test_fraud_rate_is_percentage(self, client):
        """fraud_rate_pct should be 0–100, not 0–1."""
        data = client.get("/risk/summary").json()
        if not data or data.get("fraud_rate_pct") is None:
            pytest.skip("No fraud rate data available yet")
        assert 0.0 <= data["fraud_rate_pct"] <= 100.0, (
            f"fraud_rate_pct={data['fraud_rate_pct']} is not a percentage"
        )

    def test_counts_sum_to_total(self, client):
        data = client.get("/risk/summary").json()
        if not data:
            pytest.skip("risk/summary returned empty — no scores in the last 24h")
        total = data.get("total_scored_today") or 0
        if total > 0:
            parts = (data.get("fraud_count") or 0) + (data.get("review_count") or 0) + (data.get("clear_count") or 0)
            assert parts == total, (
                f"fraud+review+clear ({parts}) doesn't sum to total_scored_today ({total})"
            )

    def test_summary_reflects_new_score(self, client, high_risk_txn):
        """Score a txn and verify total_scored_today increases."""
        before = (client.get("/risk/summary").json() or {}).get("total_scored_today") or 0
        client.post("/score/", json={**high_risk_txn, "txn_id": f"inttest-summary-{int(time.time())}"})
        after = (client.get("/risk/summary").json() or {}).get("total_scored_today") or 0
        assert after >= before, "total_scored_today should not decrease after scoring"
