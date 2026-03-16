from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Supported image formats ────────────────────────────────────────────────
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
OCR_EXTS = IMAGE_EXTS | {".pdf"}
_TIMEOUT = 30


# ── GLM-OCR extraction ─────────────────────────────────────────────────
async def ocr_extract(image_path: Path) -> str:
    """Extract text from images/PDFs via ollama glm-ocr.
    Silently returns empty string when ollama or model is unavailable."""
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
    """Check if the file needs OCR processing."""
    return path.suffix.lower() in OCR_EXTS


def is_image(path: Path) -> bool:
    """Check if the file is an image."""
    return path.suffix.lower() in IMAGE_EXTS
