"""
CreditPulse — /agent router
Spec: CREDIT-003 FR-004 (LangChain ReAct agent endpoint)
"""

from typing import Optional

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

logger = structlog.get_logger(__name__)
router = APIRouter()

_agent_executor = None


def _get_agent():
    global _agent_executor
    if _agent_executor is None:
        from agent.react_agent import build_agent
        _agent_executor = build_agent()
    return _agent_executor


class ChatRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {"query": "Why was transaction abc-123 flagged as high risk?"}
    })
    query: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: Optional[str] = None
    spec_ref: str = "CREDIT-003"


@router.post("/chat", response_model=ChatResponse, summary="Ask the fraud intelligence agent")
async def chat(request: ChatRequest):
    """
    Natural language interface to the CreditPulse ReAct agent.
    The agent uses 4 tools + RAG over financial regulations to answer questions.

    Example queries:
    - "Why was transaction abc-123 flagged?"
    - "Are there any high-risk transactions I should know about?"
    - "Is the model showing drift? Should we retrain?"
    - "What does FCRA require when denying a credit request?"

    Spec: CREDIT-003 FR-004 — agent SHALL retrieve and narrate explanations within 3 seconds.
    """
    try:
        agent_executor = _get_agent()
        result = agent_executor.invoke({"input": request.query})
        return ChatResponse(answer=result["output"], session_id=request.session_id)
    except Exception as e:
        logger.error("agent_error", error=str(e), query=request.query)
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")


@router.get("/tools", summary="List available agent tools")
async def list_tools():
    """List all tools available to the ReAct agent."""
    return {
        "tools": [
            {
                "name": "explain_transaction",
                "description": "Get SHAP + counterfactual explanation for a transaction ID",
                "spec_ref": "CREDIT-003 FR-001, FR-002",
            },
            {
                "name": "query_risk_scores",
                "description": "Search high-risk transactions by natural language criteria",
                "spec_ref": "CREDIT-001 FR-003",
            },
            {
                "name": "get_drift_report",
                "description": "PSI drift status and retrain recommendation",
                "spec_ref": "CREDIT-002 FR-004",
            },
            {
                "name": "get_fairness_metrics",
                "description": "Latest Fairlearn fairness gate results",
                "spec_ref": "CREDIT-001 NFR-004",
            },
            {
                "name": "regulatory_knowledge_base",
                "description": "RAG over PCI DSS, FCRA, Regulation E, CFPB guidance (8 documents, 28 chunks)",
                "spec_ref": "CREDIT-003 FR-004",
            },
        ]
    }
