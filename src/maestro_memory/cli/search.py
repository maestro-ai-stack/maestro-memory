from __future__ import annotations

import asyncio
from typing import Optional

import typer

from maestro_memory.core.memory import Memory


async def _try_daemon_search(query: str, limit: int) -> list[dict] | None:
    """Try daemon first, return None if not available."""
    try:
        from maestro_memory.client import MemoryClient

        client = MemoryClient()
        await client.health()
        results = await client.search(query, limit=limit, rerank=True)
        await client.close()
        return results
    except Exception:
        return None


def search_cmd(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum results"),
    current: bool = typer.Option(True, "--current/--all", help="Only current (non-invalidated) facts"),
    as_of: Optional[str] = typer.Option(None, "--as-of", help="Point-in-time query (ISO date)"),
    entity: Optional[str] = typer.Option(None, "--entity", help="Entity name for graph search"),
    hops: int = typer.Option(1, "--hops", help="Graph traversal depth"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project name"),
) -> None:
    """Search memory with hybrid retrieval."""
    asyncio.run(_search(query, limit, current, as_of, entity, hops, project))


async def _search(
    query: str, limit: int, current: bool, as_of: str | None,
    entity: str | None, hops: int, project: str | None,
) -> None:
    # Try daemon first for simple searches (no entity/as_of/all flags)
    if not entity and not as_of and current and not project:
        daemon_results = await _try_daemon_search(query, limit)
        if daemon_results is not None:
            if not daemon_results:
                typer.echo("No results found.")
                return
            for i, r in enumerate(daemon_results, 1):
                entity_str = f" [{r.get('entity_name', '')}]" if r.get("entity_name") else ""
                typer.echo(f"{i}. [{r['score']:.4f}]{entity_str} {r['content']}")
            return

    # Fallback to direct Memory access
    mem = Memory(project=project)
    await mem.init()
    try:
        if entity:
            graph_data = await mem.graph(entity, hops=hops)
            e = graph_data["entity"]
            if not e:
                typer.echo(f"Entity '{entity}' not found.")
                return
            typer.echo(f"Entity: {e.name} ({e.entity_type})")
            typer.echo(f"Summary: {e.summary or '(none)'}")
            for rel in graph_data["relations"]:
                typer.echo(f"  -> {rel.predicate} (confidence={rel.confidence:.2f})")
        else:
            results = await mem.search(query, limit=limit, current_only=current, as_of=as_of)
            if not results:
                typer.echo("No results found.")
                return
            for i, r in enumerate(results, 1):
                entity_str = f" [{r.entity.name}]" if r.entity else ""
                typer.echo(f"{i}. [{r.score:.4f}]{entity_str} {r.fact.content}")
                typer.echo(f"   type={r.fact.fact_type} valid_from={r.fact.valid_from}")
    finally:
        await mem.close()
