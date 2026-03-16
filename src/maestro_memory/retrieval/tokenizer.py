"""中文分词预处理，配合 FTS5 索引使用。"""

from __future__ import annotations

import re

# 匹配 CJK 统一汉字区块
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df]")

try:
    import jieba

    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False


def segment(text: str) -> str:
    """对含中文的文本做 jieba 分词，空格拼接后返回。

    纯英文文本原样返回（本身已空格分隔）。
    jieba 未安装时退化为原文。
    """
    if not _HAS_JIEBA or not _CJK_RE.search(text):
        return text
    # search 模式：粒度更细，召回更广
    tokens = jieba.cut_for_search(text)
    return " ".join(t.strip() for t in tokens if t.strip())
