"""
Integration tests: /health endpoints

Covers:
- Liveness probe always returns 200
- Readiness probe reports model + DB status
- Response schema is correct for K8s probes
"""

import pytest


class TestLiveness:
    def test_returns_200(self, client):
        resp = client.get("/health/live")
        assert resp.status_code == 200

    def test_response_shape(self, client):
        data = client.get("/health/live").json()
        assert data["status"] == "alive"
        assert "timestamp" in data

    def test_has_latency_header(self, client):
        resp = client.get("/health/live")
        assert "X-Latency-MS" in resp.headers
        assert "X-Request-ID" in resp.headers


class TestReadiness:
    def test_returns_200_or_503(self, client):
        resp = client.get("/health/ready")
        assert resp.status_code in (200, 503)

    def test_response_has_checks(self, client):
        data = client.get("/health/ready").json()
        assert "status" in data
        assert "checks" in data
        assert "database" in data["checks"]
        assert "model" in data["checks"]

    def test_database_check_true(self, client):
        """DB must be reachable for integration tests to be meaningful."""
        data = client.get("/health/ready").json()
        assert data["checks"]["database"] is True, (
            "PostgreSQL not reachable — is Docker running?"
        )

    def test_model_loaded(self, client):
        data = client.get("/health/ready").json()
        assert data["checks"]["model"] is True, (
            "Fraud model not loaded — run: python models/fraud_detector.py --train"
        )
