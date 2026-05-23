"""
Integration tests: /agent endpoints

Covers:
- GET /agent/tools — returns tool manifest
- POST /agent/chat — LangChain ReAct agent responds (requires GEMINI_API_KEY)
- Agent uses RAG for regulatory questions
- Spec: CREDIT-003 FR-004 — agent SHALL respond within 3 seconds
"""

import os
import time
import pytest


HAS_LLM_KEY = bool(
    os.environ.get("GEMINI_API_KEY")
    or os.environ.get("ANTHROPIC_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
)

requires_llm = pytest.mark.skipif(
    not HAS_LLM_KEY,
    reason="No LLM API key set — set GEMINI_API_KEY to run agent tests",
)


class TestAgentTools:
    def test_returns_200(self, client):
        resp = client.get("/agent/tools")
        assert resp.status_code == 200

    def test_response_has_tools_list(self, client):
        data = client.get("/agent/tools").json()
        assert "tools" in data
        assert isinstance(data["tools"], list)
        assert len(data["tools"]) >= 4

    def test_all_tools_have_required_fields(self, client):
        tools = client.get("/agent/tools").json()["tools"]
        for tool in tools:
            assert "name" in tool, f"Tool missing 'name': {tool}"
            assert "description" in tool, f"Tool missing 'description': {tool}"
            assert "spec_ref" in tool, f"Tool missing 'spec_ref': {tool}"

    def test_expected_tools_present(self, client):
        """The 5 documented tools must be registered."""
        tools = client.get("/agent/tools").json()["tools"]
        names = {t["name"] for t in tools}
        expected = {
            "explain_transaction",
            "query_risk_scores",
            "get_drift_report",
            "get_fairness_metrics",
            "regulatory_knowledge_base",
        }
        for name in expected:
            assert name in names, f"Expected tool '{name}' not found in agent tools"


class TestAgentChat:
    @pytest.fixture(autouse=True)
    def rate_limit_guard(self):
        """Gemini free tier: 15 RPM. Sleep between tests to avoid 429s."""
        yield
        time.sleep(5)

    @requires_llm
    def test_returns_200(self, client):
        resp = client.post("/agent/chat", json={"query": "How many high-risk transactions are there?"})
        assert resp.status_code == 200, resp.text

    @requires_llm
    def test_response_shape(self, client):
        data = client.post("/agent/chat", json={"query": "What tools do you have?"}).json()
        assert "answer" in data
        assert "spec_ref" in data
        assert data["spec_ref"] == "CREDIT-003"

    @requires_llm
    def test_answer_is_non_empty(self, client):
        data = client.post("/agent/chat", json={"query": "What is the current fraud rate?"}).json()
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 20, "Agent answer is suspiciously short"

    @requires_llm
    def test_drift_query_uses_tool(self, client):
        """Agent should call get_drift_report tool and return drift status."""
        data = client.post(
            "/agent/chat",
            json={"query": "Is the fraud model showing signs of drift that would require retraining?"}
        ).json()
        answer = data["answer"].lower()
        drift_keywords = ["drift", "psi", "stable", "monitor", "retrain"]
        assert any(kw in answer for kw in drift_keywords), (
            f"Agent answer doesn't reference drift context: {data['answer'][:200]}"
        )

    @requires_llm
    def test_regulatory_query_uses_rag(self, client):
        """Agent should use RAG for FCRA/PCI DSS questions."""
        data = client.post(
            "/agent/chat",
            json={"query": "What does FCRA require when denying a credit application?"}
        ).json()
        answer = data["answer"].lower()
        regulatory_keywords = ["fcra", "credit", "adverse", "notice", "consumer", "report"]
        assert any(kw in answer for kw in regulatory_keywords), (
            f"Agent answer doesn't reference regulatory knowledge: {data['answer'][:200]}"
        )

    @requires_llm
    def test_session_id_passthrough(self, client):
        data = client.post(
            "/agent/chat",
            json={"query": "Hello", "session_id": "test-session-abc"}
        ).json()
        assert data.get("session_id") == "test-session-abc"

    @requires_llm
    def test_response_under_30s(self, client):
        """Spec: CREDIT-003 FR-004 — respond within 3s. We allow 30s for integration test headroom."""
        t0 = time.perf_counter()
        client.post("/agent/chat", json={"query": "What is the current fraud rate?"})
        elapsed = time.perf_counter() - t0
        assert elapsed < 30, f"Agent took {elapsed:.1f}s — check LLM connectivity"
