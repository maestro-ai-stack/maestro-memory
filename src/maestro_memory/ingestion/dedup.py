from __future__ import annotations

import hashlib
import re

import numpy as np

from maestro_memory.core.store import Store
from maestro_memory.retrieval.embedding import EmbeddingProvider, cosine_similarity

# ── Content hash ────────────────────────────────────────────────────


def content_hash(text: str) -> str:
    """Normalize whitespace + lowercase, then compute SHA256."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode()).hexdigest()


# ── Three-tier dedup ────────────────────────────────────────────────────


async def is_duplicate(
    content: str,
    store: Store,
    embedding_provider: EmbeddingProvider | None = None,
    threshold: float = 0.9,
) -> tuple[bool, int | None]:
    """
    Three-tier duplicate detection:
    1. Exact hash match
    2. Embedding cosine similarity
    3. LLM (skipped for now)
    Returns (is_dup, existing_fact_id).
    """
    target_hash = content_hash(content)

    # Tier 1: exact hash
    facts = await store.list_facts(limit=5000, current_only=True)
    for fact in facts:
        if content_hash(fact.content) == target_hash:
            return True, fact.id

    # Tier 2: vector similarity
    if embedding_provider is not None:
        query_emb = await embedding_provider.embed(content)
        if query_emb is not None:
            for fact in facts:
                if not hasattr(fact, "id"):
                    continue
                # Read fact embedding (stored as bytes)
                raw = await store.db.execute(
                    "SELECT embedding FROM facts WHERE id = ?", (fact.id,),
                )
                row = await raw.fetchone()
                if row and row["embedding"]:
                    fact_emb = np.frombuffer(row["embedding"], dtype=np.float32)
                    if cosine_similarity(query_emb, fact_emb) > threshold:
                        return True, fact.id

    # Tier 3: LLM — skipped for now
    return False, None
