from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maestro_memory.core.models import Entity

_EXTRACTION_PROMPT = """You are a memory extraction system. Given the following text, extract structured memory operations.

Existing entities: {entities}

Text to process:
{content}

Return a JSON array of operations. Each operation is one of:
- {{"op": "ADD", "fact": "...", "entity": "entity_name", "entity_type": "concept|person|project|tool", "type": "observation|preference|feedback|decision", "importance": 0.0-1.0}}
- {{"op": "UPDATE", "fact_id": <int>, "new_content": "..."}}
- {{"op": "INVALIDATE", "fact_id": <int>, "reason": "..."}}

Return ONLY the JSON array, no other text."""


async def llm_extract(content: str, existing_entities: list[Entity]) -> list[dict]:
    """Extract facts/entities/relations via LLM. Returns empty list if no LLM available."""
    # Try OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        return await _extract_openai(content, existing_entities, openai_key)

    # Try Anthropic
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        return await _extract_anthropic(content, existing_entities, anthropic_key)

    return []


async def _extract_openai(content: str, entities: list[Entity], api_key: str) -> list[dict]:
    try:
        import openai

        entity_names = [e.name for e in entities]
        prompt = _EXTRACTION_PROMPT.format(entities=", ".join(entity_names) or "none", content=content)
        client = openai.AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = response.choices[0].message.content or "[]"
        return _parse_response(text)
    except Exception:
        return []


async def _extract_anthropic(content: str, entities: list[Entity], api_key: str) -> list[dict]:
    try:
        import anthropic

        entity_names = [e.name for e in entities]
        prompt = _EXTRACTION_PROMPT.format(entities=", ".join(entity_names) or "none", content=content)
        client = anthropic.AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text if response.content else "[]"
        return _parse_response(text)
    except Exception:
        return []


def _parse_response(text: str) -> list[dict]:
    """Parse LLM JSON response, handling markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines)
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, TypeError):
        pass
    return []
