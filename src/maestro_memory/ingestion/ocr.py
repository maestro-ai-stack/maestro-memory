from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── 支持的图片格式 ────────────────────────────────────────────────
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
OCR_EXTS = IMAGE_EXTS | {".pdf"}
_TIMEOUT = 30


# ── GLM-OCR 提取 ─────────────────────────────────────────────────
async def ocr_extract(image_path: Path) -> str:
    """通过 ollama glm-ocr 提取图片/PDF 文字。
    ollama 或模型不可用时静默返回空字符串。"""
    if image_path.suffix.lower() not in OCR_EXTS:
        return ""

    prompt = f"Text Recognition: {image_path}"
    try:
        proc = await asyncio.create_subprocess_exec(
            "ollama", "run", "glm-ocr", prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(
            proc.communicate(), timeout=_TIMEOUT,
        )
        if proc.returncode != 0:
            return ""
        return stdout.decode("utf-8", errors="replace").strip()
    except (FileNotFoundError, asyncio.TimeoutError, OSError) as exc:
        logger.debug("OCR unavailable for %s: %s", image_path, exc)
        return ""


def is_ocr_target(path: Path) -> bool:
    """判断是否需要 OCR 处理"""
    return path.suffix.lower() in OCR_EXTS


def is_image(path: Path) -> bool:
    """判断是否为图片文件"""
    return path.suffix.lower() in IMAGE_EXTS
