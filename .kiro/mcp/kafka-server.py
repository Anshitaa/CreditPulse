"""
CreditPulse Kafka MCP Server
Exposes Kafka consumer lag, topic health, and recent events to Kiro AI agents.

Usage in .kiro/mcp_config.json:
  {
    "name": "creditpulse-kafka",
    "command": "python .kiro/mcp/kafka-server.py",
    "env": {"KAFKA_BOOTSTRAP_SERVERS": "${KAFKA_BOOTSTRAP_SERVERS}"}
  }
"""

import asyncio
import json
import os
from datetime import datetime

from confluent_kafka import Consumer, KafkaException, TopicPartition
from confluent_kafka.admin import AdminClient
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

server = Server("creditpulse-kafka")

TOPICS = [
    "transactions.raw",
    "transactions.scored",
    "alerts.fraud",
    "alerts.credit",
    "features.updates",
]

CONSUMER_GROUPS = [
    "fraud-scorer-group",
    "credit-scorer-group",
    "alert-dispatcher-group",
    "feature-updater-group",
    "audit-logger-group",
]


def _get_admin() -> AdminClient:
    return AdminClient({"bootstrap.servers": KAFKA_BOOTSTRAP})


def _get_consumer_lag(group_id: str) -> dict:
    admin = _get_admin()
    consumer = Consumer(
        {
            "bootstrap.servers": KAFKA_BOOTSTRAP,
            "group.id": group_id,
            "enable.auto.commit": False,
        }
    )
    lag_info = {"group_id": group_id, "topics": {}, "total_lag": 0}
    try:
        for topic in TOPICS:
            meta = admin.list_topics(topic=topic, timeout=5)
            if topic not in meta.topics:
                continue
            partitions = list(meta.topics[topic].partitions.keys())
            tps = [TopicPartition(topic, p) for p in partitions]
            committed = consumer.committed(tps, timeout=5)
            ends = consumer.get_watermark_offsets
            topic_lag = 0
            partition_lags = {}
            for tp in committed:
                if tp.offset < 0:
                    continue
                low, high = consumer.get_watermark_offsets(tp, timeout=5)
                lag = max(0, high - tp.offset)
                topic_lag += lag
                partition_lags[tp.partition] = lag
            lag_info["topics"][topic] = {"total_lag": topic_lag, "partitions": partition_lags}
            lag_info["total_lag"] += topic_lag
    finally:
        consumer.close()
    return lag_info


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_consumer_lag",
            description="Check Kafka consumer lag for a specific consumer group or all groups. High lag indicates the scorer is falling behind.",
            inputSchema={
                "type": "object",
                "properties": {
                    "group_id": {
                        "type": "string",
                        "description": f"Consumer group ID. Options: {', '.join(CONSUMER_GROUPS)}. Omit for all groups.",
                    }
                },
            },
        ),
        Tool(
            name="get_topic_info",
            description="Get metadata for a Kafka topic: partition count, replication factor, message count.",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": f"Topic name. Options: {', '.join(TOPICS)}",
                    }
                },
                "required": ["topic"],
            },
        ),
        Tool(
            name="get_throughput_summary",
            description="Get a high-level throughput summary: messages/sec per topic, alert rate, fraud detection rate.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="peek_recent_alerts",
            description="Peek at the most recent messages from alerts.fraud or alerts.credit topic (non-destructive).",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "enum": ["alerts.fraud", "alerts.credit"],
                        "default": "alerts.fraud",
                    },
                    "count": {"type": "integer", "default": 5, "description": "Number of recent messages"},
                },
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "get_consumer_lag":
            group_id = arguments.get("group_id")
            if group_id:
                result = _get_consumer_lag(group_id)
            else:
                result = {g: _get_consumer_lag(g) for g in CONSUMER_GROUPS}
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_topic_info":
            topic = arguments["topic"]
            admin = _get_admin()
            meta = admin.list_topics(topic=topic, timeout=10)
            if topic not in meta.topics:
                return [TextContent(type="text", text=f"Topic '{topic}' not found.")]
            topic_meta = meta.topics[topic]
            info = {
                "topic": topic,
                "partitions": len(topic_meta.partitions),
                "error": str(topic_meta.error) if topic_meta.error else None,
                "partition_leaders": {
                    str(pid): p.leader for pid, p in topic_meta.partitions.items()
                },
            }
            return [TextContent(type="text", text=json.dumps(info, indent=2))]

        elif name == "get_throughput_summary":
            summary = {
                "timestamp": datetime.utcnow().isoformat(),
                "topics": TOPICS,
                "consumer_groups": CONSUMER_GROUPS,
                "note": "Connect to Kafka metrics endpoint for real-time throughput. Use get_consumer_lag for lag per group.",
                "lag_summary": {},
            }
            for group in CONSUMER_GROUPS:
                try:
                    lag = _get_consumer_lag(group)
                    summary["lag_summary"][group] = lag["total_lag"]
                except Exception as e:
                    summary["lag_summary"][group] = f"error: {e}"
            return [TextContent(type="text", text=json.dumps(summary, indent=2))]

        elif name == "peek_recent_alerts":
            topic = arguments.get("topic", "alerts.fraud")
            count = arguments.get("count", 5)
            consumer = Consumer(
                {
                    "bootstrap.servers": KAFKA_BOOTSTRAP,
                    "group.id": "kiro-mcp-peek-group",
                    "auto.offset.reset": "latest",
                    "enable.auto.commit": False,
                }
            )
            consumer.assign([TopicPartition(topic, 0)])
            messages = []
            for _ in range(count * 3):  # poll a few extra times to get count messages
                msg = consumer.poll(timeout=1.0)
                if msg and not msg.error():
                    try:
                        messages.append(json.loads(msg.value().decode("utf-8")))
                    except Exception:
                        messages.append({"raw": msg.value().decode("utf-8", errors="replace")})
                if len(messages) >= count:
                    break
            consumer.close()
            return [TextContent(type="text", text=json.dumps(messages[-count:], indent=2))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except KafkaException as e:
        return [TextContent(type="text", text=f"Kafka error: {e}. Is Kafka running at {KAFKA_BOOTSTRAP}?")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {type(e).__name__}: {e}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
