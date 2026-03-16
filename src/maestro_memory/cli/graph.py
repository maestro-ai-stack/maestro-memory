from __future__ import annotations

import asyncio
from typing import Optional

import typer

from maestro_memory.core.memory import Memory


def graph_cmd(
    entity: Optional[str] = typer.Option(None, "--entity", "-e", help="Entity name"),
    hops: int = typer.Option(1, "--hops", help="Traversal depth"),
    list_entities: bool = typer.Option(False, "--list-entities", help="List all entities"),
    list_relations: bool = typer.Option(False, "--list-relations", help="List all relations"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project name"),
) -> None:
    """Explore the knowledge graph."""
    asyncio.run(_graph(entity, hops, list_entities, list_relations, project))


async def _graph(
    entity: str | None, hops: int,
    list_entities: bool, list_relations: bool, project: str | None,
) -> None:
    mem = Memory(project=project)
    await mem.init()
    try:
        if list_entities:
            entities = await mem.store.list_entities()
            if not entities:
                typer.echo("No entities found.")
                return
            for e in entities:
                typer.echo(f"  {e.name} ({e.entity_type}) — {e.summary or '(no summary)'}")
            return

        if list_relations:
            entities = await mem.store.list_entities()
            for e in entities:
                rels = await mem.store.get_relations_for_entity(e.id)
                for r in rels:
                    sub = await mem.store.get_entity(r.subject_id)
                    obj = await mem.store.get_entity(r.object_id)
                    sub_name = sub.name if sub else f"#{r.subject_id}"
                    obj_name = obj.name if obj else f"#{r.object_id}"
                    valid = "" if not r.valid_until else f" (expired {r.valid_until})"
                    typer.echo(f"  {sub_name} --[{r.predicate}]--> {obj_name}{valid}")
            return

        if entity:
            data = await mem.graph(entity, hops=hops)
            e = data["entity"]
            if not e:
                typer.echo(f"Entity '{entity}' not found.")
                return
            typer.echo(f"Entity: {e.name} ({e.entity_type})")
            typer.echo(f"Summary: {e.summary or '(none)'}")
            typer.echo(f"Relations ({len(data['relations'])}):")
            for r in data["relations"]:
                sub = await mem.store.get_entity(r.subject_id)
                obj = await mem.store.get_entity(r.object_id)
                sub_name = sub.name if sub else f"#{r.subject_id}"
                obj_name = obj.name if obj else f"#{r.object_id}"
                typer.echo(f"  {sub_name} --[{r.predicate}]--> {obj_name}")
            typer.echo(f"Neighbors ({len(data['neighbors'])}):")
            for n in data["neighbors"]:
                typer.echo(f"  {n.name} ({n.entity_type})")
        else:
            typer.echo("Use --entity NAME, --list-entities, or --list-relations")
    finally:
        await mem.close()
