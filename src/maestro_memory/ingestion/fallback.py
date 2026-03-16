from __future__ import annotations


async def fallback_extract(content: str) -> list[dict]:
    """No-LLM fallback: store the full content as a single fact.

    No entity extraction or relation extraction is performed.
    BM25 + embedding search still works on the raw text.
    """
    return [
        {
            "op": "ADD",
            "fact": content,
            "type": "observation",
            "importance": 0.5,
        }
    ]
