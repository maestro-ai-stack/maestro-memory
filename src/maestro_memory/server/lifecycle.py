"""Server lifecycle: pre-load models on startup, close on shutdown."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from maestro_memory.core.memory import Memory

_memory: Memory | None = None


def get_memory() -> Memory:
    """Return the server-wide Memory instance (set during lifespan)."""
    if _memory is None:
        raise RuntimeError("Server not initialized — call lifespan first")
    return _memory


@asynccontextmanager
async def lifespan(app):
    """Pre-load models on startup, close on shutdown."""
    global _memory
    db_path = Path(app.state.db_path)
    _memory = Memory(path=db_path)
    await _memory.init()
    yield
    await _memory.close()
    _memory = None
