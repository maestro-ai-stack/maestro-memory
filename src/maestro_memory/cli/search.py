from __future__ import annotations

import asyncio
from typing import Optional

import typer


def _print_meta(meta: dict) -> None:
    """Print agent guidance from daemon search metadata."""
    conf = meta.get("confidence", "high")
    hint = meta.get("hint", "")
    suggestion = meta.get("suggestion")
    if conf == "none":
        typer.echo("[CONFIDENCE: NONE] No relevant data found. This topic may not be in memory.")
    elif conf == "low":
        typer.echo(f"[CONFIDENCE: LOW] Results may not be relevant to your query.{' ' + hint if hint else ''}")
    elif conf == "medium" and hint:
        typer.echo(f"[NOTE] {hint}")
    elif hint:
        typer.echo(f"[NOTE] {hint}")
    if suggestion:
        typer.echo(f"[SUGGESTION] {suggestion}")


def _print_meta_obj(meta) -> None:
    """Print agent guidance from SearchMeta object."""
    if meta.confidence == "none":
        typer.echo("[CONFIDENCE: NONE] No relevant data found. This topic may not be in memory.")
    elif meta.confidence == "low":
        typer.echo(f"[CONFIDENCE: LOW] Results may not be relevant.{' ' + meta.hint if meta.hint else ''}")
    elif meta.hint:
        typer.echo(f"[NOTE] {meta.hint}")
    if meta.suggestion:
        typer.echo(f"[SUGGESTION] {meta.suggestion}")


async def _try_daemon_search(query: str, limit: int) -> dict | None:
    """Ensure daemon is running, then search via HTTP. Returns None if unavailable."""
    try:
        from maestro_memory.cli.daemon import ensure_daemon
        if not ensure_daemon():
            return None
        from maestro_memory.client import MemoryClient
        client = MemoryClient()
        result = await client.search(query, limit=limit, rerank=True)
        await client.close()
        return result  # dict with "results" and "meta"
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
            meta = daemon_results.get("meta") if isinstance(daemon_results, dict) else None
            items = daemon_results.get("results", daemon_results) if isinstance(daemon_results, dict) else daemon_results
            # Print agent guidance header
            if meta:
                _print_meta(meta)
            if not items:
                typer.echo("No results found. This topic may not be in memory.")
                return
            for i, r in enumerate(items, 1):
                entity_str = f" [{r.get('entity_name', '')}]" if r.get("entity_name") else ""
                typer.echo(f"{i}. [{r['score']:.4f}]{entity_str} {r['content']}")
            return

    # Fallback to direct Memory access (lazy import to avoid model loading)
    from maestro_memory.core.memory import Memory
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
            # Print agent guidance from search metadata
            if mem.last_search_meta:
                _print_meta_obj(mem.last_search_meta)
            if not results:
                typer.echo("No results found. This topic may not be in memory.")
                return
            for i, r in enumerate(results, 1):
                entity_str = f" [{r.entity.name}]" if r.entity else ""
                typer.echo(f"{i}. [{r.score:.4f}]{entity_str} {r.fact.content}")
                typer.echo(f"   type={r.fact.fact_type} valid_from={r.fact.valid_from}")
    finally:
        await mem.close()
