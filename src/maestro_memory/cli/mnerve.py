"""Backward-compatible ``mnerve`` commands backed by local maestro-memory."""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import typer

from maestro_memory.cli.daemon import ensure_daemon
from maestro_memory.server.config import DAEMON_URL


app = typer.Typer(
    name="mnerve",
    help="Local compatibility CLI for maestro-memory.",
)


def _request(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    if not ensure_daemon():
        raise RuntimeError("local maestro-memory daemon did not become ready")
    try:
        with httpx.Client(base_url=DAEMON_URL, timeout=30, trust_env=False) as client:
            response = client.request(method, path, json=payload)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"local maestro-memory request failed: {exc}") from exc
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("local maestro-memory returned an invalid response")
    return data


def _query_id(query: str) -> str:
    encoded = base64.urlsafe_b64encode(query.encode("utf-8")).decode("ascii").rstrip("=")
    return f"local:{encoded}"


def _decode_query_id(query_id: str) -> str:
    prefix, separator, encoded = query_id.partition(":")
    if prefix != "local" or not separator or not encoded:
        raise ValueError("query_id must be copied from local mnerve understand output")
    try:
        padding = "=" * (-len(encoded) % 4)
        return base64.urlsafe_b64decode(f"{encoded}{padding}").decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("query_id must be copied from local mnerve understand output") from exc


def _clip(value: Any, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= max_chars else f"{text[: max_chars - 1]}…"


@app.command()
def understand(
    question: str = typer.Argument(..., help="Question or topic to recall."),
    limit: int = typer.Option(3, "--limit", "--per-type", min=1, max=10),
    max_chars: int = typer.Option(160, "--max-chars", min=60, max=400),
    json_output: bool = typer.Option(False, "--json", help="Print the complete local response."),
) -> None:
    """Search the local consolidated memory."""
    try:
        result = _request("POST", "/search", {"query": question, "limit": limit})
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}. Local daemon only; no fallback was attempted.", err=True)
        raise typer.Exit(1) from exc
    if json_output:
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
        return

    query_id = _query_id(question)
    meta = result.get("meta") or {}
    typer.echo(f"Recall [{query_id}] confidence={meta.get('confidence', 'unknown')}")
    items = [item for item in result.get("results") or [] if int(item.get("fact_id", -1)) >= 0]
    for item in items[:limit]:
        entity = f" [{item['entity_name']}]" if item.get("entity_name") else ""
        typer.echo(f"  fact:{item.get('fact_id')}{entity} | {_clip(item.get('content'), max_chars)}")
    if not items:
        typer.echo("  No relevant local memory found.")
    typer.echo(f"Feedback: mnerve feedback '{query_id}' <fact:id>  (or --none)")


@app.command()
def remember(
    content: str = typer.Argument(..., help="Knowledge to store locally."),
    fact_type: str = typer.Option("observation", "--type", help="observation, fact, conclusion, or decision"),
    entity: str | None = typer.Option(None, "--entity", help="Optional entity name."),
    entity_type: str = typer.Option("concept", "--entity-type", help="Optional entity type."),
    importance: float = typer.Option(0.5, "--importance", min=0.0, max=1.0),
    title: str | None = typer.Option(None, "--title", help="Accepted for compatibility; stored as source reference."),
    idempotency_key: str | None = typer.Option(None, "--idempotency-key", help="Stable source reference."),
) -> None:
    """Store knowledge in the local consolidated memory."""
    type_map = {"fact": "observation", "claim": "observation", "conclusion": "observation", "note": "observation"}
    payload: dict[str, Any] = {
        "content": content,
        "source_type": "agent",
        "source_ref": idempotency_key or title,
        "fact_type": type_map.get(fact_type, fact_type),
        "importance": importance,
        "entity_name": entity,
        "entity_type": entity_type,
    }
    try:
        result = _request("POST", "/add", payload)
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}. Local daemon only; no fallback was attempted.", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"Remembered episode:{result.get('episode_id')} "
        f"added={result.get('facts_added', 0)} updated={result.get('facts_updated', 0)}"
    )


@app.command()
def feedback(
    query_id: str = typer.Argument(..., help="query_id from understand output."),
    selected: list[str] | None = typer.Argument(None, help="Copyable fact:<id> targets."),
    none: bool = typer.Option(False, "--none", help="Record that no result was useful."),
) -> None:
    """Record which local facts were useful."""
    selected = selected or []
    if none and selected:
        typer.echo("Error: use selected facts or --none, not both", err=True)
        raise typer.Exit(1)
    if not none and not selected:
        typer.echo("Error: provide fact:<id> targets or use --none", err=True)
        raise typer.Exit(1)
    try:
        query = _decode_query_id(query_id)
        fact_ids = [] if none else [_parse_fact_target(token) for token in selected]
        result = _request("POST", "/feedback", {"query": query, "used_fact_ids": fact_ids})
    except (RuntimeError, ValueError) as exc:
        typer.echo(f"Error: {exc}. Local daemon only; no fallback was attempted.", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Feedback recorded: {result.get('facts_updated', len(fact_ids))} fact(s)")


def _parse_fact_target(token: str) -> int:
    prefix, separator, value = token.partition(":")
    if prefix != "fact" or not separator:
        raise ValueError("feedback targets must use fact:<id>")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError("feedback targets must use fact:<id>") from exc


@app.command()
def status() -> None:
    """Show local daemon and memory status."""
    try:
        result = _request("GET", "/status")
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}. Local daemon only; no fallback was attempted.", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"Local memory: entities={result.get('entities', 0)} facts={result.get('facts', 0)} "
        f"relations={result.get('relations', 0)} episodes={result.get('episodes', 0)}"
    )


if __name__ == "__main__":
    app()
