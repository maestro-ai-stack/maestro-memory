"""MCP server for maestro-memory — thin proxy to REST API daemon."""
from __future__ import annotations

import json

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

DAEMON_URL = "http://localhost:19830"

server = Server("maestro-memory")


def _client() -> httpx.Client:
    return httpx.Client(base_url=DAEMON_URL, timeout=30)


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="mem_search",
            description="Search memory for relevant facts. Returns ranked results with scores.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results", "default": 10},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="mem_add",
            description="Store a fact in memory. Optionally attach to an entity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Fact content"},
                    "entity_name": {"type": "string", "description": "Entity to attach to"},
                    "entity_type": {
                        "type": "string",
                        "description": "Entity type (prospect/dataset/project/person/method/tool)",
                        "default": "concept",
                    },
                    "fact_type": {
                        "type": "string",
                        "description": "Fact type (observation/decision/feedback)",
                        "default": "observation",
                    },
                    "importance": {"type": "number", "description": "Importance 0-1", "default": 0.5},
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="mem_feedback",
            description="Report which facts were actually used (implicit feedback for learning).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Original search query"},
                    "used_fact_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "IDs of facts that were used",
                    },
                },
                "required": ["query", "used_fact_ids"],
            },
        ),
        Tool(
            name="mem_status",
            description="Get memory database statistics.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        with _client() as client:
            if name == "mem_search":
                resp = client.post(
                    "/search",
                    json={
                        "query": arguments["query"],
                        "limit": arguments.get("limit", 10),
                        "rerank": True,
                    },
                )
                resp.raise_for_status()
                results = resp.json()
                if not results:
                    return [TextContent(type="text", text="No results found.")]
                lines = []
                for i, r in enumerate(results, 1):
                    entity = f" [{r.get('entity_name', '')}]" if r.get("entity_name") else ""
                    lines.append(f"{i}. [{r['score']:.4f}]{entity} {r['content']}")
                return [TextContent(type="text", text="\n".join(lines))]

            elif name == "mem_add":
                resp = client.post("/add", json=arguments)
                resp.raise_for_status()
                data = resp.json()
                return [
                    TextContent(
                        type="text",
                        text=f"Stored. episode={data['episode_id']} facts_added={data['facts_added']} entities={data['entities_created']}",
                    )
                ]

            elif name == "mem_feedback":
                resp = client.post("/feedback", json=arguments)
                resp.raise_for_status()
                data = resp.json()
                return [TextContent(type="text", text=f"Feedback recorded. {data}")]

            elif name == "mem_status":
                resp = client.get("/status")
                resp.raise_for_status()
                data = resp.json()
                return [TextContent(type="text", text=json.dumps(data, indent=2))]

            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except httpx.ConnectError:
        return [TextContent(type="text", text="ERROR: mmem daemon not running. Start with: mmem server-start")]
    except Exception as e:
        return [TextContent(type="text", text=f"ERROR: {e}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def run():
    import asyncio

    asyncio.run(main())


if __name__ == "__main__":
    run()
