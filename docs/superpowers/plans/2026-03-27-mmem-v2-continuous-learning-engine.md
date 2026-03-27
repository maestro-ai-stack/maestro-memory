# maestro-memory v2: Continuous Learning Memory Engine

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform mmem from a CLI tool with 11s cold-start into a persistent daemon with <100ms search, 6-channel recall, learned ranking, online learning, and user profiling.

**Architecture:** Daemon server (FastAPI/uvicorn) holds models warm + SQLite open. CLI and MCP are thin clients over HTTP/Unix socket. 6-channel multi-recall → LightGBM pre-rank → cross-encoder rerank → Thompson Sampling blend. River streaming ML for online learning. User profile (3-layer: session/mid-term/long-term) with entity affinity and behavior sequence features.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, SQLite + hnswlib, sentence-transformers (all-MiniLM-L6-v2 + cross-encoder/ms-marco-MiniLM-L-6-v2), LightGBM, River, numpy

---

## Phase 0: Daemon Architecture (P0 — fixes 11s → <100ms)

### Task 0.1: Server core (FastAPI + uvicorn daemon)

**Files:**
- Create: `src/maestro_memory/server/__init__.py`
- Create: `src/maestro_memory/server/app.py`
- Create: `src/maestro_memory/server/routes.py`
- Create: `src/maestro_memory/server/lifecycle.py`
- Modify: `pyproject.toml` (add fastapi, uvicorn, httpx deps)
- Test: `tests/test_server.py`

- [ ] **Step 1: Add server dependencies to pyproject.toml**

```toml
# In [project.optional-dependencies]
server = ["fastapi>=0.115", "uvicorn>=0.32", "httpx>=0.28"]
```

- [ ] **Step 2: Write failing test for server health endpoint**

```python
# tests/test_server.py
import pytest
from httpx import AsyncClient, ASGITransport
from maestro_memory.server.app import create_app

@pytest.mark.asyncio
async def test_health(tmp_path):
    app = create_app(db_path=tmp_path / "test.db")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_server.py::test_health -v`
Expected: FAIL (no module maestro_memory.server)

- [ ] **Step 4: Implement server lifecycle (model pre-loading)**

```python
# src/maestro_memory/server/lifecycle.py
from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
from maestro_memory import Memory

_memory: Memory | None = None

def get_memory() -> Memory:
    assert _memory is not None, "Server not initialized"
    return _memory

@asynccontextmanager
async def lifespan(app):
    """Pre-load models on startup, close on shutdown."""
    global _memory
    db_path = Path(app.state.db_path)
    _memory = Memory(path=db_path)
    await _memory.init()
    # Warm up models by running a dummy search
    await _memory.search("warmup", limit=1, rerank=True)
    yield
    await _memory.close()
    _memory = None
```

- [ ] **Step 5: Implement FastAPI app + routes**

```python
# src/maestro_memory/server/app.py
from fastapi import FastAPI
from pathlib import Path
from maestro_memory.server.lifecycle import lifespan

def create_app(db_path: str | Path | None = None, project: str | None = None) -> FastAPI:
    from maestro_memory.core.config import get_db_path
    resolved = str(db_path) if db_path else str(get_db_path(project))
    app = FastAPI(title="maestro-memory", lifespan=lifespan)
    app.state.db_path = resolved
    from maestro_memory.server.routes import router
    app.include_router(router)
    return app
```

```python
# src/maestro_memory/server/routes.py
from __future__ import annotations
from fastapi import APIRouter
from pydantic import BaseModel
from maestro_memory.server.lifecycle import get_memory

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    current_only: bool = True
    as_of: str | None = None
    rerank: bool = True

class SearchResultItem(BaseModel):
    fact_id: int
    content: str
    score: float
    entity_name: str | None = None
    fact_type: str = ""
    importance: float = 0.5

class AddRequest(BaseModel):
    content: str
    source_type: str = "manual"
    source_ref: str | None = None
    fact_type: str = "observation"
    importance: float = 0.5
    entity_name: str | None = None
    entity_type: str = "concept"

class AddResponse(BaseModel):
    episode_id: int
    facts_added: int = 0
    facts_updated: int = 0
    facts_invalidated: int = 0
    entities_created: int = 0

@router.get("/health")
async def health():
    mem = get_memory()
    stats = await mem.status()
    return {"status": "ok", **stats}

@router.post("/search")
async def search(req: SearchRequest) -> list[SearchResultItem]:
    mem = get_memory()
    results = await mem.search(
        req.query, limit=req.limit, current_only=req.current_only,
        as_of=req.as_of, rerank=req.rerank,
    )
    return [
        SearchResultItem(
            fact_id=r.fact.id, content=r.fact.content, score=r.score,
            entity_name=r.entity.name if r.entity else None,
            fact_type=r.fact.fact_type, importance=r.fact.importance,
        )
        for r in results
    ]

@router.post("/add")
async def add(req: AddRequest) -> AddResponse:
    mem = get_memory()
    result = await mem.add(
        req.content, source_type=req.source_type, source_ref=req.source_ref,
        fact_type=req.fact_type, importance=req.importance,
        entity_name=req.entity_name, entity_type=req.entity_type,
    )
    return AddResponse(
        episode_id=result.episode_id, facts_added=result.facts_added,
        facts_updated=result.facts_updated, facts_invalidated=result.facts_invalidated,
        entities_created=result.entities_created,
    )

@router.get("/status")
async def status():
    mem = get_memory()
    return await mem.status()
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_server.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/maestro_memory/server/ tests/test_server.py pyproject.toml
git commit -m "feat(server): add FastAPI daemon with model pre-loading"
```

### Task 0.2: CLI thin client (mmem → HTTP → daemon)

**Files:**
- Create: `src/maestro_memory/client.py`
- Modify: `src/maestro_memory/cli/search.py`
- Modify: `src/maestro_memory/cli/add.py`
- Modify: `src/maestro_memory/cli/__init__.py` (add `server start/stop` commands)
- Test: `tests/test_client.py`

- [ ] **Step 1: Write failing test for thin client**

```python
# tests/test_client.py
import pytest
from httpx import AsyncClient, ASGITransport
from maestro_memory.server.app import create_app
from maestro_memory.client import MemoryClient

@pytest.mark.asyncio
async def test_client_search(tmp_path):
    app = create_app(db_path=tmp_path / "test.db")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        # Seed data via direct HTTP
        await http.post("/add", json={"content": "User likes dark mode"})
        # Test client
        client = MemoryClient(base_url="http://test", http_client=http)
        results = await client.search("dark mode")
        assert len(results) >= 1
```

- [ ] **Step 2: Implement MemoryClient**

```python
# src/maestro_memory/client.py
from __future__ import annotations
import httpx

class MemoryClient:
    """Thin HTTP client for mmem daemon."""

    def __init__(self, base_url: str = "http://localhost:19830", http_client: httpx.AsyncClient | None = None):
        self._base_url = base_url
        self._http = http_client

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(base_url=self._base_url, timeout=30)
        return self._http

    async def search(self, query: str, limit: int = 10, rerank: bool = True) -> list[dict]:
        http = await self._get_http()
        resp = await http.post("/search", json={"query": query, "limit": limit, "rerank": rerank})
        resp.raise_for_status()
        return resp.json()

    async def add(self, content: str, **kwargs) -> dict:
        http = await self._get_http()
        resp = await http.post("/add", json={"content": content, **kwargs})
        resp.raise_for_status()
        return resp.json()

    async def status(self) -> dict:
        http = await self._get_http()
        resp = await http.get("/status")
        resp.raise_for_status()
        return resp.json()

    async def health(self) -> dict:
        http = await self._get_http()
        resp = await http.get("/health")
        resp.raise_for_status()
        return resp.json()
```

- [ ] **Step 3: Add `mmem server start/stop` CLI commands**

```python
# In src/maestro_memory/cli/__init__.py, add:

@app.command()
def server_start(
    port: int = typer.Option(19830, help="Server port"),
    project: str = typer.Option(None, help="Project name"),
    daemon: bool = typer.Option(True, help="Run as background daemon"),
):
    """Start mmem server (pre-loads models, holds DB open)."""
    import subprocess, sys, os
    if daemon:
        pid_file = Path.home() / ".maestro" / "memory" / "server.pid"
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            [sys.executable, "-m", "maestro_memory.server", "--port", str(port)],
            stdout=open(pid_file.with_suffix(".log"), "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pid_file.write_text(str(proc.pid))
        typer.echo(f"Server started (pid={proc.pid}, port={port})")
    else:
        import uvicorn
        from maestro_memory.server.app import create_app
        app = create_app(project=project)
        uvicorn.run(app, host="127.0.0.1", port=port)

@app.command()
def server_stop():
    """Stop mmem server daemon."""
    import signal
    pid_file = Path.home() / ".maestro" / "memory" / "server.pid"
    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        try:
            os.kill(pid, signal.SIGTERM)
            typer.echo(f"Server stopped (pid={pid})")
        except ProcessLookupError:
            typer.echo("Server not running")
        pid_file.unlink(missing_ok=True)
    else:
        typer.echo("No server pid file found")
```

- [ ] **Step 4: Modify CLI search/add to prefer daemon, fallback to direct**

```python
# In cli/search.py, modify to try daemon first:
async def _search(query, limit, rerank):
    try:
        from maestro_memory.client import MemoryClient
        client = MemoryClient()
        await client.health()  # Check if daemon running
        return await client.search(query, limit=limit, rerank=rerank)
    except Exception:
        # Fallback to direct mode (cold start)
        mem = Memory()
        await mem.init()
        results = await mem.search(query, limit=limit, rerank=rerank)
        await mem.close()
        return [{"content": r.fact.content, "score": r.score} for r in results]
```

- [ ] **Step 5: Run tests, commit**

Run: `pytest tests/test_client.py tests/test_server.py -v`

```bash
git add src/maestro_memory/client.py src/maestro_memory/server/ src/maestro_memory/cli/
git commit -m "feat(cli): thin client mode — CLI talks to daemon over HTTP, fallback to direct"
```

### Task 0.3: Benchmark daemon vs direct

- [ ] **Step 1: Write benchmark script**

```python
# evals/bench_latency.py
"""Compare daemon vs direct mode latency."""
import asyncio, time, httpx
from maestro_memory import Memory
from maestro_memory.client import MemoryClient

async def bench_direct(db_path, queries):
    """Cold start every time."""
    times = []
    for q in queries:
        t0 = time.perf_counter()
        mem = Memory(path=db_path)
        await mem.init()
        await mem.search(q, limit=10)
        await mem.close()
        times.append(time.perf_counter() - t0)
    return times

async def bench_daemon(queries):
    """Hit running daemon."""
    client = MemoryClient()
    times = []
    for q in queries:
        t0 = time.perf_counter()
        await client.search(q, limit=10)
        times.append(time.perf_counter() - t0)
    return times
```

- [ ] **Step 2: Run and record in experiment log**

Expected: direct ~11s/query → daemon ~0.1s/query (100x speedup)

---

## Phase 1: Storage — hnswlib ANN Index (P0)

### Task 1.1: Replace numpy linear scan with hnswlib

**Files:**
- Create: `src/maestro_memory/retrieval/ann_index.py`
- Modify: `src/maestro_memory/retrieval/fusion.py` (replace `cosine_top_k` call)
- Modify: `src/maestro_memory/core/memory.py` (build index on init)
- Modify: `pyproject.toml` (add hnswlib dep)
- Test: `tests/test_retrieval.py` (add ANN test)

- [ ] **Step 1: Write failing test**

```python
# In tests/test_retrieval.py
import numpy as np
from maestro_memory.retrieval.ann_index import ANNIndex

def test_ann_index_search():
    idx = ANNIndex(dim=4)
    idx.add(1, np.array([1.0, 0, 0, 0], dtype=np.float32))
    idx.add(2, np.array([0, 1.0, 0, 0], dtype=np.float32))
    idx.add(3, np.array([0.9, 0.1, 0, 0], dtype=np.float32))
    results = idx.search(np.array([1.0, 0, 0, 0], dtype=np.float32), k=2)
    assert results[0][0] == 1  # closest
    assert results[1][0] == 3  # second closest

def test_ann_index_empty():
    idx = ANNIndex(dim=4)
    results = idx.search(np.array([1.0, 0, 0, 0], dtype=np.float32), k=5)
    assert results == []
```

- [ ] **Step 2: Implement ANNIndex**

```python
# src/maestro_memory/retrieval/ann_index.py
from __future__ import annotations
import numpy as np

class ANNIndex:
    """HNSW approximate nearest neighbor index, lazy-loaded."""

    def __init__(self, dim: int = 384, max_elements: int = 100_000):
        self._dim = dim
        self._max = max_elements
        self._index = None
        self._id_map: dict[int, int] = {}  # fact_id → internal_id
        self._reverse: dict[int, int] = {}  # internal_id → fact_id
        self._count = 0

    def _ensure_index(self):
        if self._index is None:
            try:
                import hnswlib
                self._index = hnswlib.Index(space="cosine", dim=self._dim)
                self._index.init_index(max_elements=self._max, ef_construction=200, M=16)
                self._index.set_ef(50)
            except ImportError:
                return False
        return True

    def add(self, fact_id: int, embedding: np.ndarray) -> None:
        if not self._ensure_index():
            return
        if fact_id in self._id_map:
            return  # already indexed
        internal_id = self._count
        self._index.add_items(embedding.reshape(1, -1), np.array([internal_id]))
        self._id_map[fact_id] = internal_id
        self._reverse[internal_id] = fact_id
        self._count += 1

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        if self._index is None or self._count == 0:
            return []
        k = min(k, self._count)
        labels, distances = self._index.knn_query(query.reshape(1, -1), k=k)
        results = []
        for internal_id, dist in zip(labels[0], distances[0]):
            fact_id = self._reverse.get(int(internal_id))
            if fact_id is not None:
                similarity = 1.0 - dist  # cosine distance → similarity
                results.append((fact_id, float(similarity)))
        return results

    @property
    def size(self) -> int:
        return self._count
```

- [ ] **Step 3: Wire into fusion.py — replace linear scan**

In `hybrid_search()`, replace the embedding search block:

```python
# Replace lines 44-57 in fusion.py with:
emb_results: list[tuple[int, float]] = []
if embedding_provider:
    query_emb = await embedding_provider.embed(query)
    if query_emb is not None:
        from maestro_memory.retrieval.ann_index import get_ann_index
        ann = get_ann_index()
        if ann is not None and ann.size > 0:
            emb_results = ann.search(query_emb, k=fetch_limit)
        else:
            # Fallback to brute-force
            cur = await store.db.execute("SELECT id, embedding FROM facts WHERE embedding IS NOT NULL")
            rows = await cur.fetchall()
            fact_embeddings = [(row[0], np.frombuffer(row[1], dtype=np.float32)) for row in rows]
            if fact_embeddings:
                emb_results = cosine_top_k(query_emb, fact_embeddings, k=fetch_limit)
```

- [ ] **Step 4: Build index on Memory.init() and Memory.add()**

- [ ] **Step 5: Run benchmark, verify speedup at 10K facts**

Expected: embedding search 50ms → <5ms at 10K facts

- [ ] **Step 6: Run full test suite + retrieval benchmark, commit**

```bash
pytest tests/ -v
python evals/run_retrieval_eval.py
git commit -m "feat(retrieval): add hnswlib ANN index — O(log n) embedding search"
```

---

## Phase 2: Multi-Channel Recall (P1)

### Task 2.1: User profile (3-layer)

**Files:**
- Create: `src/maestro_memory/core/profile.py`
- Modify: `src/maestro_memory/core/schema.py` (add user_profile table)
- Modify: `src/maestro_memory/core/store.py` (profile CRUD)
- Modify: `src/maestro_memory/core/memory.py` (load/save profile)
- Test: `tests/test_profile.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_profile.py
import pytest
import numpy as np
from maestro_memory.core.profile import UserProfile

def test_profile_entity_affinity():
    p = UserProfile()
    p.update_entity(entity_id=1, boost=1.0)
    p.update_entity(entity_id=1, boost=1.0)
    p.update_entity(entity_id=2, boost=1.0)
    assert p.get_affinity(1) > p.get_affinity(2)  # accessed more
    assert p.get_affinity(99) == 0.0  # never accessed

def test_profile_decay():
    p = UserProfile()
    p.update_entity(entity_id=1, boost=1.0)
    initial = p.get_affinity(1)
    p.decay_all(factor=0.5)
    assert p.get_affinity(1) == pytest.approx(initial * 0.5)
```

- [ ] **Step 2: Implement UserProfile**

```python
# src/maestro_memory/core/profile.py
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class UserProfile:
    """Persistent user profile (stored in SQLite, loaded on init)."""

    entity_affinity: dict[int, float] = field(default_factory=dict)
    topic_embedding: np.ndarray | None = field(default=None, repr=False)
    total_searches: int = 0
    total_adds: int = 0

    def update_entity(self, entity_id: int, boost: float = 1.0) -> None:
        old = self.entity_affinity.get(entity_id, 0.0)
        self.entity_affinity[entity_id] = old * 0.95 + boost

    def get_affinity(self, entity_id: int) -> float:
        return self.entity_affinity.get(entity_id, 0.0)

    def decay_all(self, factor: float = 0.85) -> None:
        for eid in list(self.entity_affinity):
            self.entity_affinity[eid] *= factor
            if self.entity_affinity[eid] < 0.01:
                del self.entity_affinity[eid]

    def update_topic(self, query_embedding: np.ndarray, alpha: float = 0.05) -> None:
        if self.topic_embedding is None:
            self.topic_embedding = query_embedding.copy()
        else:
            self.topic_embedding = (1 - alpha) * self.topic_embedding + alpha * query_embedding

    def top_entities(self, n: int = 10) -> list[tuple[int, float]]:
        return sorted(self.entity_affinity.items(), key=lambda x: -x[1])[:n]
```

- [ ] **Step 3: Add profile persistence to schema + store, commit**

### Task 2.2: Add recall channels 4-6

**Files:**
- Create: `src/maestro_memory/retrieval/channels.py`
- Modify: `src/maestro_memory/retrieval/fusion.py` (add channels to hybrid_search)
- Test: `tests/test_channels.py`

- [ ] **Step 1: Implement 3 new recall channels**

```python
# src/maestro_memory/retrieval/channels.py
from __future__ import annotations
from maestro_memory.core.store import Store
from maestro_memory.core.profile import UserProfile
from maestro_memory.core.session import SessionState

async def recall_user_interest(
    store: Store, profile: UserProfile, limit: int = 20,
) -> list[tuple[int, float]]:
    """Channel 4: user behavior inverted — facts from high-affinity entities."""
    candidates = []
    for eid, affinity in profile.top_entities(10):
        facts = await store.get_facts_by_entity(eid, limit=5)
        for f in facts:
            candidates.append((f.id, affinity * f.importance))
    candidates.sort(key=lambda x: -x[1])
    return candidates[:limit]

async def recall_time_window(
    store: Store, days: int = 7, limit: int = 20,
) -> list[tuple[int, float]]:
    """Channel 5: recent facts by creation time."""
    from datetime import datetime, timedelta
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    cur = await store.db.execute(
        "SELECT id, importance FROM facts WHERE created_at >= ? AND valid_until IS NULL ORDER BY created_at DESC LIMIT ?",
        (cutoff, limit),
    )
    return [(row[0], row[1]) for row in await cur.fetchall()]

async def recall_session_context(
    store: Store, session: SessionState, limit: int = 20,
) -> list[tuple[int, float]]:
    """Channel 6: facts from entities active in current session."""
    candidates = []
    for eid, activation in sorted(session.entity_activation.items(), key=lambda x: -x[1])[:10]:
        facts = await store.get_facts_by_entity(eid, limit=5)
        for f in facts:
            if f.id not in session.recent_fact_ids:
                candidates.append((f.id, activation * f.importance))
    candidates.sort(key=lambda x: -x[1])
    return candidates[:limit]
```

- [ ] **Step 2: Wire into fusion.py — 6-channel RRF**

- [ ] **Step 3: Run LongMemEval retrieval eval, compare, commit**

---

## Phase 3: Learned Ranking (P1)

### Task 3.1: Feature extraction (12 features per candidate)

**Files:**
- Create: `src/maestro_memory/ranking/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Implement feature extractor**

```python
# src/maestro_memory/ranking/features.py
from __future__ import annotations
import numpy as np
from maestro_memory.core.models import Fact, SearchResult
from maestro_memory.core.session import SessionState
from maestro_memory.core.profile import UserProfile

def extract_features(
    query: str,
    result: SearchResult,
    bm25_score: float,
    embed_score: float,
    graph_distance: float,
    session: SessionState,
    profile: UserProfile,
) -> np.ndarray:
    """Extract 12-dim feature vector for a candidate fact."""
    fact = result.fact
    entity_id = fact.entity_id

    from datetime import datetime
    now = datetime.now()
    try:
        created = datetime.fromisoformat(fact.created_at)
        days_since_created = max((now - created).total_seconds() / 86400, 0)
    except (ValueError, TypeError):
        days_since_created = 0

    try:
        if fact.last_accessed:
            accessed = datetime.fromisoformat(fact.last_accessed)
            days_since_accessed = max((now - accessed).total_seconds() / 86400, 0)
        else:
            days_since_accessed = days_since_created
    except (ValueError, TypeError):
        days_since_accessed = days_since_created

    # Query-fact matching
    query_words = set(query.lower().split())
    fact_words = set(fact.content.lower().split())
    entity_overlap = len(query_words & fact_words) / max(len(query_words), 1)

    import math
    base_level = math.log(fact.access_count + 1) - 0.5 * math.log(days_since_created + 1)

    return np.array([
        bm25_score,             # 0: BM25 relevance
        embed_score,            # 1: embedding cosine similarity
        graph_distance,         # 2: graph hop distance (0=direct)
        entity_overlap,         # 3: word overlap ratio
        fact.importance,        # 4: fact importance (0-1)
        fact.access_count,      # 5: historical access frequency
        len(fact.content),      # 6: fact text length
        days_since_created,     # 7: age of fact
        days_since_accessed,    # 8: recency of last access
        base_level,             # 9: ACT-R base-level activation
        profile.get_affinity(entity_id) if entity_id else 0.0,  # 10: user entity affinity
        session.get_entity_boost(entity_id),  # 11: session entity activation
    ], dtype=np.float32)
```

### Task 3.2: LightGBM pre-ranker (trained via cross-encoder distillation)

**Files:**
- Create: `src/maestro_memory/ranking/prerank.py`
- Create: `src/maestro_memory/ranking/train.py`
- Test: `tests/test_prerank.py`

- [ ] **Step 1: Write training data generator (cross-encoder distillation, NOT eval set)**

```python
# src/maestro_memory/ranking/train.py
"""Generate training data by distilling cross-encoder scores.
Uses ShareGPT/UltraChat data — NOT evaluation benchmarks."""

async def generate_training_data(
    memory: Memory, conversations: list[dict], output_path: Path,
) -> int:
    """Ingest conversations, generate queries, score with cross-encoder.
    Returns number of (features, label) pairs generated.
    """
    from maestro_memory.retrieval.fusion import _get_reranker
    reranker = _get_reranker()
    # ... for each conversation: ingest → create query → retrieve candidates
    # ... for each candidate: extract 12 features + cross-encoder score as label
    # ... save as numpy arrays
```

- [ ] **Step 2: Implement LightGBM pre-ranker**

```python
# src/maestro_memory/ranking/prerank.py
from __future__ import annotations
from pathlib import Path
import numpy as np

class PreRanker:
    """LightGBM pre-ranker using 12-dim features."""

    def __init__(self, model_path: Path | None = None):
        self._model = None
        self._model_path = model_path

    def load(self) -> bool:
        if self._model_path and self._model_path.exists():
            import lightgbm as lgb
            self._model = lgb.Booster(model_file=str(self._model_path))
            return True
        return False

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Score candidates. features shape: (n_candidates, 12)."""
        if self._model is None:
            return features[:, 0]  # fallback to BM25 score
        return self._model.predict(features)

    def rank(self, features: np.ndarray, candidate_ids: list[int], limit: int) -> list[int]:
        """Return top-limit candidate IDs sorted by predicted score."""
        scores = self.predict(features)
        ranked = sorted(zip(candidate_ids, scores), key=lambda x: -x[1])
        return [cid for cid, _ in ranked[:limit]]
```

- [ ] **Step 3: Train on synthetic data, evaluate, commit**

### Task 3.3: River online pre-ranker (replaces LightGBM over time)

**Files:**
- Create: `src/maestro_memory/ranking/online.py`

```python
# src/maestro_memory/ranking/online.py
from __future__ import annotations
from river import tree

class OnlineRanker:
    """Streaming pre-ranker using River Hoeffding tree."""

    def __init__(self):
        self._model = tree.HoeffdingTreeClassifier()
        self._n_updates = 0

    def predict(self, features: dict) -> float:
        return self._model.predict_proba_one(features).get(True, 0.5)

    def update(self, features: dict, used: bool) -> None:
        self._model.learn_one(features, used)
        self._n_updates += 1

    @property
    def n_updates(self) -> int:
        return self._n_updates
```

---

## Phase 4: Online Learning + Feedback (P2)

### Task 4.1: Serving logger

**Files:**
- Create: `src/maestro_memory/logging/serving_log.py`
- Modify: `src/maestro_memory/core/schema.py` (add serving_logs table)

```sql
CREATE TABLE serving_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    query TEXT NOT NULL,
    intent TEXT,
    candidate_fact_ids TEXT,  -- JSON array
    returned_fact_ids TEXT,   -- JSON array
    used_fact_ids TEXT,       -- JSON array (filled later by feedback)
    features_json TEXT,       -- JSON: {fact_id: [12 features]}
    latency_ms REAL
);
```

### Task 4.2: Implicit feedback endpoint

```python
# In server/routes.py
@router.post("/feedback")
async def feedback(req: FeedbackRequest):
    """Agent reports which facts it actually used."""
    mem = get_memory()
    # Update serving log with used_fact_ids
    # Update River online ranker
    # Update entity affinity in profile
    # Increment access_count for used facts (stronger boost)
```

### Task 4.3: Thompson Sampling blender (replaces RRF)

**Files:**
- Create: `src/maestro_memory/ranking/blender.py`

```python
# src/maestro_memory/ranking/blender.py
import numpy as np

class ThompsonBlender:
    """Thompson Sampling for channel blending weights."""

    def __init__(self, n_channels: int = 6):
        self.alpha = np.ones(n_channels)  # Beta prior: successes
        self.beta = np.ones(n_channels)   # Beta prior: failures

    def sample_weights(self) -> np.ndarray:
        return np.random.beta(self.alpha, self.beta)

    def update(self, channel_idx: int, reward: float) -> None:
        if reward > 0.5:
            self.alpha[channel_idx] += 1
        else:
            self.beta[channel_idx] += 1
```

---

## Phase 5: MCP Interface (P2)

### Task 5.1: MCP server wrapping HTTP daemon

**Files:**
- Create: `src/maestro_memory/mcp/__init__.py`
- Create: `src/maestro_memory/mcp/server.py`

7 MCP tools: `mem_search`, `mem_add`, `mem_profile`, `mem_recent`, `mem_graph`, `mem_timeline`, `mem_status`

Each tool is a thin wrapper calling the HTTP daemon endpoints.

---

## Phase 6: Offline Training Pipeline (P3)

### Task 6.1: `mmem train` CLI

Commands:
- `mmem train prerank` — export serving_logs → train LightGBM → save model
- `mmem train rerank` — export (query, fact, score) → finetune cross-encoder on 4090
- `mmem train intent` — export (query, intent) → finetune DistilBERT
- `mmem train memory-rl` — export episodes → GRPO on Qwen2.5-3B on 4090

### Task 6.2: Model registry + hot-reload

Daemon watches `~/.maestro/memory/models/` for new weights. When a new model file appears, hot-swap without restart.

---

## Experiment Checkpoints

After each phase, run the autoresearch eval loop:

| Phase | What to measure | Target |
|-------|----------------|--------|
| 0 (daemon) | Latency: 11s → ? | <200ms search |
| 1 (hnswlib) | 10K facts latency | <50ms embedding search |
| 2 (6-channel) | LongMemEval session recall | >90% (was 83%) |
| 3 (LightGBM) | LongMemEval QA (subagent-judged, N=50) | Maintain 94% with 2x less reranker calls |
| 4 (online learning) | QA after 100 feedback cycles | Self-improving trend |
| 5 (MCP) | Integration test with Claude Code | Working multi-tool |
| 6 (training) | QA after first offline train cycle | +2-5% from domain adaptation |

---

## File Map Summary

```
src/maestro_memory/
  server/                    ← Phase 0 (NEW)
    __init__.py
    app.py                   # FastAPI app factory
    routes.py                # HTTP endpoints
    lifecycle.py             # Model pre-loading, graceful shutdown
  client.py                  ← Phase 0 (NEW)
  core/
    memory.py                ← Modified (Phases 0-4)
    session.py               ← Existing
    profile.py               ← Phase 2 (NEW)
    store.py                 ← Modified (Phase 2: profile CRUD)
    schema.py                ← Modified (Phase 2: profile table, Phase 4: serving_logs)
    config.py                ← Existing
    models.py                ← Existing
  retrieval/
    fusion.py                ← Modified (Phase 1: hnswlib, Phase 2: 6-channel)
    ann_index.py             ← Phase 1 (NEW)
    channels.py              ← Phase 2 (NEW)
    bm25.py                  ← Existing
    embedding.py             ← Existing
    graph.py                 ← Existing
    temporal.py              ← Existing
  ranking/                   ← Phase 3 (NEW directory)
    features.py              # 12-dim feature extraction
    prerank.py               # LightGBM pre-ranker
    online.py                # River streaming ranker
    blender.py               # Thompson Sampling blend
    train.py                 # Training data generation (cross-encoder distillation)
  logging/                   ← Phase 4 (NEW directory)
    serving_log.py           # Serving log collection
  mcp/                       ← Phase 5 (NEW directory)
    server.py                # MCP tool definitions
  ingestion/                 ← Existing (no changes)
  cli/                       ← Modified (Phase 0: server commands)
```
