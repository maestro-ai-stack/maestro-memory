"""Maximum Marginal Relevance for diverse retrieval."""
from __future__ import annotations

import numpy as np


def mmr_rerank(
    results: list,
    embeddings: dict[int, np.ndarray],
    query_embedding: np.ndarray | None = None,
    lambda_param: float = 0.7,
    limit: int = 10,
) -> list:
    """Rerank results using MMR for diversity.

    lambda_param: 1.0 = pure relevance, 0.0 = pure diversity.
    """
    if len(results) <= limit or query_embedding is None:
        return results[:limit]

    selected = [results[0]]
    remaining = list(results[1:])

    while len(selected) < limit and remaining:
        best_score = -float("inf")
        best_idx = 0

        selected_embs = []
        for s in selected:
            emb = embeddings.get(s.fact.id)
            if emb is not None:
                selected_embs.append(emb)

        for i, candidate in enumerate(remaining):
            # Relevance score (already computed)
            relevance = candidate.score

            # Diversity: max similarity to already-selected items
            cand_emb = embeddings.get(candidate.fact.id)
            if cand_emb is not None and selected_embs:
                similarities = [
                    float(
                        np.dot(cand_emb, s_emb)
                        / (np.linalg.norm(cand_emb) * np.linalg.norm(s_emb) + 1e-8)
                    )
                    for s_emb in selected_embs
                ]
                max_similarity = max(similarities)
            else:
                max_similarity = 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected
