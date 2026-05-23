"""
CreditPulse PostgreSQL MCP Server
Exposes transaction DB, risk scores, and audit logs to Kiro AI agents.

Usage in .kiro/mcp_config.json:
  {
    "name": "creditpulse-postgres",
    "command": "python .kiro/mcp/postgres-server.py",
    "env": {"DATABASE_URL": "${DATABASE_URL}"}
  }
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta

import asyncpg
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, TextContent, Tool

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://creditpulse:creditpulse@localhost:5435/creditpulse",
)

server = Server("creditpulse-postgres")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="query_risk_scores",
            description="Get fraud/credit risk scores for accounts. Filter by account_id, score range, or time window.",
            inputSchema={
                "type": "object",
                "properties": {
                    "account_id": {"type": "string", "description": "Filter by specific account ID"},
                    "min_score": {"type": "number", "description": "Minimum risk score (0-100)"},
                    "max_score": {"type": "number", "description": "Maximum risk score (0-100)"},
                    "hours_back": {"type": "integer", "description": "Only scores from last N hours", "default": 24},
                    "limit": {"type": "integer", "description": "Max rows to return", "default": 20},
                },
            },
        ),
        Tool(
            name="get_transaction_explanation",
            description="Get SHAP explanation + counterfactuals for a specific transaction.",
            inputSchema={
                "type": "object",
                "properties": {
                    "txn_id": {"type": "string", "description": "Transaction ID to explain"},
                },
                "required": ["txn_id"],
            },
        ),
        Tool(
            name="query_audit_decisions",
            description="Query the audit log of model decisions. Useful for compliance checks.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_version": {"type": "string", "description": "Filter by model version"},
                    "decision": {"type": "string", "enum": ["FRAUD", "REVIEW", "CLEAR"], "description": "Filter by decision"},
                    "hours_back": {"type": "integer", "default": 24},
                    "limit": {"type": "integer", "default": 50},
                },
            },
        ),
        Tool(
            name="get_drift_report",
            description="Get the latest PSI drift report for model features.",
            inputSchema={
                "type": "object",
                "properties": {
                    "feature_name": {"type": "string", "description": "Filter by specific feature (optional)"},
                },
            },
        ),
        Tool(
            name="get_fairness_metrics",
            description="Get latest Fairlearn fairness metrics (demographic parity, equal opportunity).",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "default": "fraud_detector"},
                },
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        if name == "query_risk_scores":
            hours_back = arguments.get("hours_back", 24)
            since = datetime.utcnow() - timedelta(hours=hours_back)
            conditions = ["scored_at > $1"]
            params: list = [since]
            idx = 2
            if "account_id" in arguments:
                conditions.append(f"account_id = ${idx}")
                params.append(arguments["account_id"])
                idx += 1
            if "min_score" in arguments:
                conditions.append(f"composite_risk_score >= ${idx}")
                params.append(arguments["min_score"])
                idx += 1
            if "max_score" in arguments:
                conditions.append(f"composite_risk_score <= ${idx}")
                params.append(arguments["max_score"])
                idx += 1
            limit = arguments.get("limit", 20)
            where = " AND ".join(conditions)
            rows = await conn.fetch(
                f"SELECT txn_id, account_id, fraud_probability, credit_risk_score, "
                f"composite_risk_score, decision, scored_at FROM mart.risk_scores "
                f"WHERE {where} ORDER BY scored_at DESC LIMIT {limit}",
                *params,
            )
            result = [dict(r) for r in rows]
            return [TextContent(type="text", text=json.dumps(result, default=str, indent=2))]

        elif name == "get_transaction_explanation":
            txn_id = arguments["txn_id"]
            row = await conn.fetchrow(
                "SELECT * FROM audit.explanations WHERE txn_id = $1 ORDER BY created_at DESC LIMIT 1",
                txn_id,
            )
            if not row:
                return [TextContent(type="text", text=f"No explanation found for transaction {txn_id}")]
            return [TextContent(type="text", text=json.dumps(dict(row), default=str, indent=2))]

        elif name == "query_audit_decisions":
            hours_back = arguments.get("hours_back", 24)
            since = datetime.utcnow() - timedelta(hours=hours_back)
            conditions = ["decided_at > $1"]
            params = [since]
            idx = 2
            if "model_version" in arguments:
                conditions.append(f"model_version = ${idx}")
                params.append(arguments["model_version"])
                idx += 1
            if "decision" in arguments:
                conditions.append(f"decision = ${idx}")
                params.append(arguments["decision"])
                idx += 1
            limit = arguments.get("limit", 50)
            where = " AND ".join(conditions)
            rows = await conn.fetch(
                f"SELECT txn_id, model_version, score, decision, top_features, decided_at "
                f"FROM audit.model_decisions WHERE {where} ORDER BY decided_at DESC LIMIT {limit}",
                *params,
            )
            return [TextContent(type="text", text=json.dumps([dict(r) for r in rows], default=str, indent=2))]

        elif name == "get_drift_report":
            feature_filter = arguments.get("feature_name")
            if feature_filter:
                rows = await conn.fetch(
                    "SELECT * FROM audit.drift_reports WHERE feature_name = $1 ORDER BY computed_at DESC LIMIT 10",
                    feature_filter,
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM audit.drift_reports ORDER BY computed_at DESC LIMIT 20"
                )
            return [TextContent(type="text", text=json.dumps([dict(r) for r in rows], default=str, indent=2))]

        elif name == "get_fairness_metrics":
            model_name = arguments.get("model_name", "fraud_detector")
            row = await conn.fetchrow(
                "SELECT * FROM audit.fairness_reports WHERE model_name = $1 ORDER BY computed_at DESC LIMIT 1",
                model_name,
            )
            if not row:
                return [TextContent(type="text", text=f"No fairness report found for model: {model_name}")]
            return [TextContent(type="text", text=json.dumps(dict(row), default=str, indent=2))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    finally:
        await conn.close()


@server.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(
            uri="creditpulse://schema",
            name="CreditPulse Database Schema",
            description="Full schema of the CreditPulse PostgreSQL database",
            mimeType="text/markdown",
        )
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    if uri == "creditpulse://schema":
        return """# CreditPulse Database Schema

## mart.risk_scores
- txn_id (TEXT PK), account_id (TEXT), merchant_id (TEXT)
- fraud_probability (FLOAT), credit_risk_score (FLOAT), composite_risk_score (FLOAT)
- decision (TEXT: FRAUD/REVIEW/CLEAR), model_version (TEXT)
- scored_at (TIMESTAMPTZ)

## audit.model_decisions
- id (SERIAL PK), txn_id (TEXT), model_version (TEXT)
- score (FLOAT), decision (TEXT), top_features (JSONB)
- decided_at (TIMESTAMPTZ)

## audit.explanations
- id (SERIAL PK), txn_id (TEXT)
- shap_values (JSONB), top_features (JSONB)
- counterfactuals (JSONB), anchor (JSONB)
- created_at (TIMESTAMPTZ)

## audit.drift_reports
- id (SERIAL PK), feature_name (TEXT), psi_score (FLOAT)
- baseline_distribution (JSONB), current_distribution (JSONB)
- drift_detected (BOOLEAN), computed_at (TIMESTAMPTZ)

## audit.fairness_reports
- id (SERIAL PK), model_name (TEXT), model_version (TEXT)
- demographic_parity_delta (FLOAT), equal_opportunity_delta (FLOAT)
- predictive_parity_delta (FLOAT), gate_passed (BOOLEAN)
- computed_at (TIMESTAMPTZ)
"""
    return "Resource not found"


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
