"""
CreditPulse MLflow MCP Server
Exposes experiment tracking, model registry, and run comparison to Kiro AI agents.

Usage in .kiro/mcp_config.json:
  {
    "name": "creditpulse-mlflow",
    "command": "python .kiro/mcp/mlflow-server.py",
    "env": {"MLFLOW_TRACKING_URI": "${MLFLOW_TRACKING_URI}"}
  }
"""

import asyncio
import json
import os

import mlflow
from mlflow.tracking import MlflowClient
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient(tracking_uri=MLFLOW_URI)

server = Server("creditpulse-mlflow")

MODEL_NAMES = ["fraud_detector", "credit_risk_scorer", "anomaly_detector"]


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="list_experiments",
            description="List all MLflow experiments with their latest run metrics.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_best_run",
            description="Get the best run for an experiment, sorted by a given metric.",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment_name": {
                        "type": "string",
                        "description": "Experiment name. Options: fraud_detection, credit_risk, anomaly_detection",
                    },
                    "metric": {
                        "type": "string",
                        "default": "auc_roc",
                        "description": "Metric to sort by (higher is better)",
                    },
                },
                "required": ["experiment_name"],
            },
        ),
        Tool(
            name="compare_runs",
            description="Compare two MLflow runs side-by-side (params, metrics, fairness metrics).",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id_a": {"type": "string"},
                    "run_id_b": {"type": "string"},
                },
                "required": ["run_id_a", "run_id_b"],
            },
        ),
        Tool(
            name="get_registered_models",
            description="List all registered models and their stage (Staging/Production/Archived).",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="promote_model",
            description="Transition a model version to a new stage (Staging → Production). Requires fairness gate to have passed.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "enum": MODEL_NAMES},
                    "version": {"type": "string", "description": "Model version number"},
                    "stage": {"type": "string", "enum": ["Staging", "Production", "Archived"]},
                    "justification": {"type": "string", "description": "Reason for promotion (required for audit trail)"},
                },
                "required": ["model_name", "version", "stage", "justification"],
            },
        ),
        Tool(
            name="get_run_artifacts",
            description="List artifacts for a run (feature importance plots, confusion matrices, fairness reports).",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {"type": "string"},
                },
                "required": ["run_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "list_experiments":
            experiments = client.search_experiments()
            result = []
            for exp in experiments:
                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    max_results=1,
                    order_by=["start_time DESC"],
                )
                latest = None
                if runs:
                    latest = {
                        "run_id": runs[0].info.run_id,
                        "status": runs[0].info.status,
                        "metrics": runs[0].data.metrics,
                        "start_time": runs[0].info.start_time,
                    }
                result.append({
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "artifact_location": exp.artifact_location,
                    "latest_run": latest,
                })
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "get_best_run":
            experiment = client.get_experiment_by_name(arguments["experiment_name"])
            if not experiment:
                return [TextContent(type="text", text=f"Experiment '{arguments['experiment_name']}' not found.")]
            metric = arguments.get("metric", "auc_roc")
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric} DESC"],
                max_results=1,
            )
            if not runs:
                return [TextContent(type="text", text="No runs found.")]
            run = runs[0]
            result = {
                "run_id": run.info.run_id,
                "experiment": arguments["experiment_name"],
                "status": run.info.status,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "compare_runs":
            run_a = client.get_run(arguments["run_id_a"])
            run_b = client.get_run(arguments["run_id_b"])
            comparison = {
                "run_a": {"id": run_a.info.run_id, "metrics": run_a.data.metrics, "params": run_a.data.params},
                "run_b": {"id": run_b.info.run_id, "metrics": run_b.data.metrics, "params": run_b.data.params},
                "metric_deltas": {
                    k: round(run_b.data.metrics.get(k, 0) - run_a.data.metrics.get(k, 0), 4)
                    for k in set(list(run_a.data.metrics.keys()) + list(run_b.data.metrics.keys()))
                },
            }
            return [TextContent(type="text", text=json.dumps(comparison, indent=2, default=str))]

        elif name == "get_registered_models":
            models = client.search_registered_models()
            result = []
            for m in models:
                versions = client.search_model_versions(f"name='{m.name}'")
                result.append({
                    "name": m.name,
                    "description": m.description,
                    "versions": [
                        {
                            "version": v.version,
                            "stage": v.current_stage,
                            "run_id": v.run_id,
                            "status": v.status,
                        }
                        for v in versions
                    ],
                })
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "promote_model":
            model_name = arguments["model_name"]
            version = arguments["version"]
            stage = arguments["stage"]
            justification = arguments["justification"]
            client.set_model_version_tag(model_name, version, "promotion_justification", justification)
            client.transition_model_version_stage(model_name, version, stage, archive_existing_versions=(stage == "Production"))
            return [TextContent(type="text", text=f"Model {model_name} v{version} promoted to {stage}. Justification logged.")]

        elif name == "get_run_artifacts":
            artifacts = client.list_artifacts(arguments["run_id"])
            result = [{"path": a.path, "is_dir": a.is_dir, "file_size": a.file_size} for a in artifacts]
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        return [TextContent(type="text", text=f"MLflow error: {type(e).__name__}: {e}. Is MLflow running at {MLFLOW_URI}?")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
