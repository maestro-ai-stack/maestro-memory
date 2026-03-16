"""Chinese word segmentation preprocessing for FTS5 indexing."""

from __future__ import annotations

import re

# Match CJK Unified Ideographs blocks
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df]")

try:
    import jieba

    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False


def segment(text: str) -> str:
    """Segment Chinese text with jieba, return space-joined tokens.

    Pure English text is returned as-is (already space-delimited).
    Falls back to raw text when jieba is not installed.
    """
    if not _HAS_JIEBA or not _CJK_RE.search(text):
        return text
    # search mode: finer granularity, broader recall
    tokens = jieba.cut_for_search(text)
    return " ".join(t.strip() for t in tokens if t.strip())
