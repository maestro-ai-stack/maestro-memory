"""FastAPI app factory for mmem daemon."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from maestro_memory.server.lifecycle import lifespan


def create_app(db_path: str | Path | None = None, project: str | None = None) -> FastAPI:
    """Create a FastAPI app with the given db_path or project."""
    if db_path is None:
        from maestro_memory.core.config import get_db_path
        db_path = get_db_path(project)

    app = FastAPI(title="maestro-memory", lifespan=lifespan)
    app.state.db_path = str(db_path)

    from maestro_memory.server.routes import router
    app.include_router(router)

    return app
