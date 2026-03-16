from __future__ import annotations

import hashlib
import re

import numpy as np

from maestro_memory.core.store import Store
from maestro_memory.retrieval.embedding import EmbeddingProvider, cosine_similarity

# ── 内容哈希 ────────────────────────────────────────────────────


def content_hash(text: str) -> str:
    """归一化空白 + 小写后取 SHA256"""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode()).hexdigest()


# ── 三级去重 ────────────────────────────────────────────────────


async def is_duplicate(
    content: str,
    store: Store,
    embedding_provider: EmbeddingProvider | None = None,
    threshold: float = 0.9,
) -> tuple[bool, int | None]:
    """
    三级去重检测：
    1. 精确哈希匹配
    2. Embedding 余弦相似度
    3. LLM（暂跳过）
    返回 (is_dup, existing_fact_id)
    """
    target_hash = content_hash(content)

    # Tier 1: 精确哈希
    facts = await store.list_facts(limit=5000, current_only=True)
    for fact in facts:
        if content_hash(fact.content) == target_hash:
            return True, fact.id

    # Tier 2: 向量相似度
    if embedding_provider is not None:
        query_emb = await embedding_provider.embed(content)
        if query_emb is not None:
            for fact in facts:
                if not hasattr(fact, "id"):
                    continue
                # 读取 fact embedding（存储为 bytes）
                raw = await store.db.execute(
                    "SELECT embedding FROM facts WHERE id = ?", (fact.id,),
                )
                row = await raw.fetchone()
                if row and row["embedding"]:
                    fact_emb = np.frombuffer(row["embedding"], dtype=np.float32)
                    if cosine_similarity(query_emb, fact_emb) > threshold:
                        return True, fact.id

    # Tier 3: LLM — 暂跳过
    return False, None
