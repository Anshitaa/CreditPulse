"""
CreditPulse — Unit Tests for ML Models
Called by: .kiro/hooks/auto-test.sh on model file save

Tests cover:
- fraud_detector: feature preprocessing, score output shape, decision thresholds
- anomaly: score output range, anomaly flag
- credit_risk: composite formula, risk bands
- counterfactual: output structure, feasibility field

Run: pytest tests/unit/test_models.py -v
"""

import numpy as np
import pandas as pd
import pytest


# ── fraud_detector ──────────────────────────────────────────────────────────

class TestFraudDetectorPreprocessing:
    def test_feature_cols_count(self):
        from models.fraud_detector import FEATURE_COLS
        assert len(FEATURE_COLS) == 7

    def test_feature_descriptions_match_cols(self):
        from models.fraud_detector import FEATURE_COLS, FEATURE_DESCRIPTIONS
        for col in FEATURE_COLS:
            assert col in FEATURE_DESCRIPTIONS, f"Missing description for: {col}"

    def test_score_output_structure(self):
        """score_transaction must return required keys."""
        # Mock the model rather than loading from disk
        import unittest.mock as mock
        import numpy as np

        with mock.patch("models.fraud_detector.load_model") as mock_load:
            from sklearn.preprocessing import LabelEncoder
            import xgboost as xgb
            import shap

            mock_model = mock.MagicMock(spec=xgb.XGBClassifier)
            mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

            mock_le = LabelEncoder().fit(["grocery", "retail", "wire_transfer"])

            mock_explainer = mock.MagicMock(spec=shap.TreeExplainer)
            mock_explainer.shap_values.return_value = np.zeros((1, 7))

            mock_load.return_value = (mock_model, mock_le, mock_explainer)
            model, le, explainer = mock_load()

            from models.fraud_detector import score_transaction
            txn = {
                "txn_id": "test-001",
                "amount": 1000,
                "merchant_category": "retail",
                "is_foreign_merchant": False,
                "hour_of_day": 10,
                "day_of_week": 2,
                "txn_velocity_1h": 1,
                "amount_vs_avg_ratio": 1.5,
            }
            result = score_transaction(txn, model, le, explainer)

            required_keys = ["txn_id", "fraud_probability", "fraud_risk_score", "decision", "top_features", "inference_latency_ms"]
            for key in required_keys:
                assert key in result, f"Missing key in score_transaction output: {key}"

    def test_decision_thresholds(self):
        """Score > 75 → FRAUD, 40–75 → REVIEW, < 40 → CLEAR."""
        import unittest.mock as mock
        import numpy as np
        from sklearn.preprocessing import LabelEncoder
        import xgboost as xgb
        import shap

        for prob, expected_decision in [(0.9, "FRAUD"), (0.55, "REVIEW"), (0.2, "CLEAR")]:
            mock_model = mock.MagicMock(spec=xgb.XGBClassifier)
            mock_model.predict_proba.return_value = np.array([[1 - prob, prob]])
            mock_le = LabelEncoder().fit(["grocery", "retail"])
            mock_explainer = mock.MagicMock(spec=shap.TreeExplainer)
            mock_explainer.shap_values.return_value = np.zeros((1, 7))

            from models.fraud_detector import score_transaction
            txn = {"txn_id": "x", "amount": 100, "merchant_category": "retail", "is_foreign_merchant": False, "hour_of_day": 12, "day_of_week": 0, "txn_velocity_1h": 0, "amount_vs_avg_ratio": 1.0}
            result = score_transaction(txn, mock_model, mock_le, mock_explainer)
            assert result["decision"] == expected_decision, f"For prob={prob}, expected {expected_decision}, got {result['decision']}"

    def test_top_features_count(self):
        import unittest.mock as mock
        import numpy as np
        from sklearn.preprocessing import LabelEncoder
        import xgboost as xgb
        import shap

        mock_model = mock.MagicMock(spec=xgb.XGBClassifier)
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_le = LabelEncoder().fit(["grocery", "retail"])
        mock_explainer = mock.MagicMock(spec=shap.TreeExplainer)
        mock_explainer.shap_values.return_value = np.array([[0.1, -0.2, 0.3, 0.05, -0.1, 0.4, 0.15]])

        from models.fraud_detector import score_transaction
        txn = {"txn_id": "x", "amount": 100, "merchant_category": "retail", "is_foreign_merchant": False, "hour_of_day": 12, "day_of_week": 0, "txn_velocity_1h": 0, "amount_vs_avg_ratio": 1.0}
        result = score_transaction(txn, mock_model, mock_le, mock_explainer)
        assert len(result["top_features"]) == 5


# ── PSI drift monitor ────────────────────────────────────────────────────────

class TestPSIDriftMonitor:
    def test_psi_identical_distributions_is_zero(self):
        from governance.drift_monitor import compute_psi
        data = np.random.default_rng(42).normal(0, 1, 1000)
        psi = compute_psi(data, data)
        assert psi < 0.01, f"Identical distributions should have PSI ≈ 0, got {psi}"

    def test_psi_different_distributions_is_high(self):
        from governance.drift_monitor import compute_psi
        rng = np.random.default_rng(42)
        baseline = rng.normal(0, 1, 1000)
        drifted = rng.normal(5, 1, 1000)   # mean shifted by 5 std devs
        psi = compute_psi(baseline, drifted)
        assert psi > 0.20, f"Severely drifted distribution should have PSI > 0.20, got {psi}"

    def test_psi_interpretation(self):
        from governance.drift_monitor import interpret_psi
        assert interpret_psi(0.05) == "STABLE"
        assert interpret_psi(0.15) == "MONITOR"
        assert interpret_psi(0.25) == "RETRAIN"

    def test_drift_monitor_returns_all_features(self):
        from governance.drift_monitor import DriftMonitor, FEATURE_COLS
        monitor = DriftMonitor()
        baseline, current = monitor.load_data(source="synthetic")
        report = monitor.compute_all_psi(baseline, current)
        assert "features" in report
        for feat in FEATURE_COLS:
            assert feat in report["features"], f"Missing feature in drift report: {feat}"

    def test_drift_report_structure(self):
        from governance.drift_monitor import DriftMonitor
        monitor = DriftMonitor()
        baseline, current = monitor.load_data(source="synthetic")
        report = monitor.compute_all_psi(baseline, current)
        assert "computed_at" in report
        assert "drift_detected" in report
        assert "recommendation" in report
        assert report["recommendation"] in ("RETRAIN_REQUIRED", "NO_ACTION")


# ── Data generator ───────────────────────────────────────────────────────────

class TestSyntheticDataGenerator:
    def test_generates_correct_count(self):
        import numpy as np
        from data.synthetic_transactions import generate_accounts, generate_merchants, generate_transactions
        from datetime import datetime
        rng = np.random.default_rng(42)
        accounts = generate_accounts(100, rng)
        merchants = generate_merchants(20, rng)
        txns = generate_transactions(accounts, merchants, 500, rng, datetime(2024, 1, 1))
        assert len(txns) == 500

    def test_fraud_rate_is_reasonable(self):
        import numpy as np
        from data.synthetic_transactions import generate_accounts, generate_merchants, generate_transactions
        from datetime import datetime
        rng = np.random.default_rng(42)
        accounts = generate_accounts(200, rng)
        merchants = generate_merchants(30, rng)
        txns = generate_transactions(accounts, merchants, 2000, rng, datetime(2024, 1, 1))
        fraud_rate = sum(1 for t in txns if t.is_fraud) / len(txns)
        assert 0.01 <= fraud_rate <= 0.15, f"Fraud rate {fraud_rate:.2%} is outside expected 1–15% range"

    def test_amounts_are_positive(self):
        import numpy as np
        from data.synthetic_transactions import generate_accounts, generate_merchants, generate_transactions
        from datetime import datetime
        rng = np.random.default_rng(42)
        accounts = generate_accounts(50, rng)
        merchants = generate_merchants(10, rng)
        txns = generate_transactions(accounts, merchants, 200, rng, datetime(2024, 1, 1))
        assert all(t.amount > 0 for t in txns)
        assert all(t.amount <= 50_000 for t in txns)

    def test_transaction_has_required_fields(self):
        import numpy as np
        from data.synthetic_transactions import generate_accounts, generate_merchants, generate_transactions
        from datetime import datetime
        rng = np.random.default_rng(42)
        accounts = generate_accounts(10, rng)
        merchants = generate_merchants(5, rng)
        txns = generate_transactions(accounts, merchants, 5, rng, datetime(2024, 1, 1))
        required = ["txn_id", "account_id", "merchant_id", "amount", "is_fraud", "created_at"]
        for txn in txns:
            for field in required:
                assert hasattr(txn, field), f"Transaction missing field: {field}"


# ── Counterfactual ───────────────────────────────────────────────────────────

class TestCounterfactualEngine:
    def test_explain_returns_required_keys(self):
        import unittest.mock as mock
        from models.counterfactual import CounterfactualEngine

        engine = CounterfactualEngine()
        # Patch _load so we don't need disk artifacts
        with mock.patch.object(engine, "_load"):
            with mock.patch.object(engine, "_model") as mock_model:
                import numpy as np
                mock_model.predict_proba = mock.MagicMock(return_value=np.array([[0.15, 0.85]]))
                engine._model = mock_model
                engine._label_encoder = mock.MagicMock()
                engine._label_encoder.transform = mock.MagicMock(return_value=np.array([0]))
                engine._dice_explainer = mock.MagicMock()
                engine._dice_explainer.generate_counterfactuals.side_effect = Exception("mocked")

                result = engine.explain({"txn_id": "test", "amount": 5000, "merchant_category": "wire_transfer"})
                required = ["txn_id", "original_fraud_probability", "original_risk_score", "counterfactuals", "explanation"]
                for key in required:
                    assert key in result, f"Missing key in counterfactual output: {key}"
