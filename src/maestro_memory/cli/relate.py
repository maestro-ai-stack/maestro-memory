from __future__ import annotations

import asyncio
from typing import Optional

import typer

from maestro_memory.core.memory import Memory


def relate_cmd(
    subject: str = typer.Argument(..., help="Subject entity name"),
    predicate: str = typer.Argument(..., help="Relationship type (e.g. works_at, leads, depends_on)"),
    obj: str = typer.Argument(..., help="Object entity name"),
    confidence: float = typer.Option(1.0, "--confidence", "-c", help="Confidence 0.0-1.0"),
    subject_type: str = typer.Option("concept", "--subject-type", help="Subject entity type"),
    object_type: str = typer.Option("concept", "--object-type", help="Object entity type"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project name"),
) -> None:
    """Create a relation between two entities."""
    asyncio.run(_relate(subject, predicate, obj, confidence, subject_type, object_type, project))


async def _relate(
    subject: str, predicate: str, obj: str,
    confidence: float, subject_type: str, object_type: str,
    project: str | None,
) -> None:
    mem = Memory(project=project)
    await mem.init()
    try:
        # Auto-create entities if they don't exist
        sub_entity, sub_created = await mem.store.get_or_create_entity(subject, subject_type)
        obj_entity, obj_created = await mem.store.get_or_create_entity(obj, object_type)

        rid = await mem.store.add_relation(sub_entity.id, predicate, obj_entity.id, confidence)

        created_str = []
        if sub_created:
            created_str.append(subject)
        if obj_created:
            created_str.append(obj)

        typer.echo(f"Relation #{rid}: {subject} --[{predicate}]--> {obj}")
        if created_str:
            typer.echo(f"  new entities: {', '.join(created_str)}")
    finally:
        await mem.close()
