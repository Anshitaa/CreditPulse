"""
CreditPulse — LangChain ReAct Agent
Spec: CREDIT-003 FR-004

A fraud intelligence agent that answers natural language questions about transactions,
risk scores, model health, and regulatory compliance.

Example queries:
  "Why was transaction abc-123 flagged as high risk?"
  "Show me the top 10 riskiest transactions in the last hour"
  "Is our fraud model showing signs of drift? Should we retrain?"
  "What fairness metrics does the current model have?"
  "What does FCRA require when we deny a transaction?"

Supports multiple LLM providers (Anthropic Claude, OpenAI GPT-4, Google Gemini).
Defaults to Gemini Flash (free tier) for portfolio demo.

Usage:
    python agent/react_agent.py  # interactive demo
    python agent/react_agent.py --query "Why was txn X flagged?"
"""

import os
import sys
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


def _get_llm():
    """Pick LLM provider based on available API keys. Gemini Flash is default (free)."""
    if ANTHROPIC_API_KEY:
        from langchain_anthropic import ChatAnthropic
        logger.info("using_llm", provider="anthropic", model="claude-sonnet-4-6")
        return ChatAnthropic(model="claude-sonnet-4-6", api_key=ANTHROPIC_API_KEY, temperature=0)
    elif OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        logger.info("using_llm", provider="openai", model="gpt-4o")
        return ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0)
    elif GEMINI_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        logger.info("using_llm", provider="google", model="gemini-2.0-flash")
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY, temperature=0)
    else:
        raise ValueError(
            "No LLM API key found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY."
        )


def build_agent():
    """Build the CreditPulse ReAct agent with 4 tools + RAG retriever."""
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools.retriever import create_retriever_tool
    from langchain_core.prompts import PromptTemplate

    from agent.tools import explain_transaction, get_drift_report, get_fairness_metrics, query_risk_scores
    from agent.rag.indexer import INDEX_DIR, load_index

    llm = _get_llm()

    # RAG retriever tool for regulatory / model knowledge
    tools = [explain_transaction, query_risk_scores, get_drift_report, get_fairness_metrics]

    if INDEX_DIR.exists():
        retriever = load_index().as_retriever(search_kwargs={"k": 3})
        rag_tool = create_retriever_tool(
            retriever,
            name="regulatory_knowledge_base",
            description=(
                "Search the CreditPulse knowledge base for: regulatory requirements (PCI DSS, FCRA, "
                "Regulation E, CFPB guidance), model documentation, feature importance explanations, "
                "and operational runbooks. Use this for compliance questions or model interpretation."
            ),
        )
        tools.append(rag_tool)
    else:
        logger.warning("rag_index_not_found", msg="Run 'python agent/rag/indexer.py --build' first")

    system_prompt = """You are the CreditPulse Fraud Intelligence Agent — an expert in real-time fraud
detection, credit risk, and financial AI governance.

You have access to the following tools:
{tools}

Use this format exactly:
Question: the input question you must answer
Thought: think about which tool to use and why
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the complete answer to the original question

Guidelines:
- For transaction explanations: always include the fraud probability, top SHAP features, AND a counterfactual
- For compliance questions: cite the specific regulation (FCRA, Reg E, PCI DSS) and section
- For drift/fairness: give a clear recommendation (retrain / monitor / no action)
- Keep Final Answer concise and actionable (< 200 words)
- Never fabricate transaction IDs or scores — only use data from the tools

Question: {input}
Thought: {agent_scratchpad}"""

    prompt = PromptTemplate.from_template(system_prompt)
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=6, handle_parsing_errors=True)


def run_demo_queries(agent_executor) -> None:
    demo_queries = [
        "Are there any high-risk transactions I should be aware of right now?",
        "Is the fraud model showing any signs of feature drift that would require retraining?",
        "What fairness guarantees does the current CreditPulse model have?",
        "What does the FCRA require when we flag a transaction and deny a credit request?",
    ]

    print("\n" + "=" * 70)
    print("CreditPulse ReAct Agent — Demo Queries")
    print("=" * 70)

    for i, query in enumerate(demo_queries, 1):
        print(f"\n[Query {i}] {query}")
        print("-" * 50)
        try:
            result = agent_executor.invoke({"input": query})
            print(f"\nAnswer: {result['output']}")
        except Exception as e:
            print(f"Error: {e}")
        print()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="CreditPulse ReAct Agent")
    parser.add_argument("--query", type=str, help="Single query to run")
    parser.add_argument("--demo", action="store_true", help="Run demo queries")
    parser.add_argument("--interactive", action="store_true", help="Interactive REPL")
    args = parser.parse_args()

    agent_executor = build_agent()

    if args.query:
        result = agent_executor.invoke({"input": args.query})
        print(f"\nAnswer: {result['output']}")
    elif args.demo:
        run_demo_queries(agent_executor)
    elif args.interactive:
        print("CreditPulse Agent — Interactive Mode (Ctrl+C to exit)")
        while True:
            try:
                query = input("\n> ").strip()
                if not query:
                    continue
                result = agent_executor.invoke({"input": query})
                print(f"\nAnswer: {result['output']}")
            except KeyboardInterrupt:
                print("\nExiting.")
                break
    else:
        run_demo_queries(agent_executor)


if __name__ == "__main__":
    main()
