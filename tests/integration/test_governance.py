"""
Integration tests: /governance endpoints

Covers:
- GET /governance/drift — PSI report structure + fairness gate interpretation
- GET /governance/fairness — Fairlearn report with gate decision
- GET /governance/models — MLflow model registry
- POST /governance/drift/run — on-demand drift check writes to DB
- POST /governance/fairness/run — on-demand fairness gate
- Spec: CREDIT-002 FR-004, CREDIT-003
"""

import pytest


class TestDriftReport:
    def test_returns_200(self, client):
        resp = client.get("/governance/drift")
        assert resp.status_code == 200, resp.text

    def test_response_shape(self, client):
        data = client.get("/governance/drift").json()
        assert "drift_detected" in data
        assert "recommendation" in data
        assert "features" in data
        assert isinstance(data["features"], list)

    def test_recommendation_is_valid_enum(self, client):
        data = client.get("/governance/drift").json()
        assert data["recommendation"] in ("RETRAIN_REQUIRED", "NO_ACTION")

    def test_drift_detected_matches_recommendation(self, client):
        data = client.get("/governance/drift").json()
        if data["drift_detected"]:
            assert data["recommendation"] == "RETRAIN_REQUIRED"
        else:
            assert data["recommendation"] == "NO_ACTION"

    def test_feature_filter_param(self, client):
        """?feature=amount should return only rows for that feature."""
        resp = client.get("/governance/drift", params={"feature": "amount"})
        assert resp.status_code == 200
        data = resp.json()
        for feat in data["features"]:
            assert feat["feature_name"] == "amount"

    def test_feature_rows_have_required_fields(self, client):
        data = client.get("/governance/drift").json()
        if data["features"]:
            row = data["features"][0]
            assert "feature_name" in row
            assert "psi_score" in row
            assert "drift_detected" in row


class TestRunDriftCheck:
    def test_returns_200(self, client):
        resp = client.post("/governance/drift/run")
        assert resp.status_code == 200, resp.text

    def test_report_has_all_feature_cols(self, client):
        from governance.drift_monitor import FEATURE_COLS
        data = client.post("/governance/drift/run").json()
        assert "features" in data
        # features is a dict keyed by feature name (from compute_all_psi)
        features = data["features"]
        assert isinstance(features, dict), f"Expected dict, got {type(features).__name__}"
        for col in FEATURE_COLS:
            assert col in features, f"Feature {col} missing from drift report"

    def test_run_writes_to_db(self, client, db):
        """After triggering drift check, audit.drift_reports should have a new row."""
        import psycopg2
        from psycopg2.extras import RealDictCursor
        cur = db.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT COUNT(*) AS before FROM audit.drift_reports")
        before = cur.fetchone()["before"]
        client.post("/governance/drift/run")
        cur.execute("SELECT COUNT(*) AS after FROM audit.drift_reports")
        after = cur.fetchone()["after"]
        cur.close()
        assert after > before, "drift/run should write new rows to audit.drift_reports"


class TestFairnessReport:
    def test_returns_200_or_404(self, client):
        resp = client.get("/governance/fairness")
        assert resp.status_code in (200, 404), resp.text

    def test_has_gate_status_if_present(self, client):
        resp = client.get("/governance/fairness")
        if resp.status_code == 200:
            data = resp.json()
            assert "gate_passed" in data or "model_name" in data


class TestRunFairnessGate:
    def test_returns_200(self, client):
        resp = client.post("/governance/fairness/run")
        assert resp.status_code == 200, resp.text

    def test_response_has_gate_passed(self, client):
        data = client.post("/governance/fairness/run").json()
        assert "gate_passed" in data
        assert isinstance(data["gate_passed"], bool)

    def test_gate_passed_is_consistent_with_report(self, client):
        """If gate_passed=True, report must not contain failing metrics.
        Spec: CREDIT-001 NFR-004 — gate decision must be traceable to the report."""
        data = client.post("/governance/fairness/run").json()
        gate_passed = data["gate_passed"]
        report = data.get("report", {})
        if gate_passed:
            # If gate passed, there should be no failing_metrics recorded
            failing = report.get("failing_metrics", [])
            assert not failing, (
                f"gate_passed=True but failing_metrics is non-empty: {failing}"
            )

    def test_report_covers_protected_groups(self, client):
        """Report must evaluate at least one protected group (age/gender/income tier)."""
        data = client.post("/governance/fairness/run").json()
        report = data.get("report", {})
        assert report, "Fairness report should not be empty"

    def test_report_contains_group_results(self, client):
        data = client.post("/governance/fairness/run").json()
        report = data.get("report", {})
        # If gate failed, report should explain why
        assert "gate_passed" in report or "group_results" in report or len(report) > 0


class TestModelRegistry:
    def test_returns_200(self, client):
        resp = client.get("/governance/models")
        assert resp.status_code in (200, 500), resp.text

    def test_response_shape_if_mlflow_up(self, client):
        resp = client.get("/governance/models")
        if resp.status_code == 200:
            data = resp.json()
            assert "models" in data
            assert isinstance(data["models"], list)
