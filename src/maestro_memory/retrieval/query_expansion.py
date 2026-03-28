"""Template-based multi-query expansion. No LLM required.

Generates 2-3 variant queries to improve recall for complex questions:
- Extracts key entities/nouns
- Generates time-specific variants
- Creates semantic variants by removing question words
"""
from __future__ import annotations

import re

# Common date references → expand to date strings
DATE_PATTERNS = {
    r"\bvalentine'?s?\s*day\b": "February 14 Valentine",
    r"\bchristmas\b": "December 25 Christmas",
    r"\bnew\s*year'?s?\b": "January 1 New Year",
    r"\bthanksgiving\b": "November Thanksgiving",
    r"\bhalloween\b": "October 31 Halloween",
    r"\beaster\b": "April Easter",
    r"\bmother'?s?\s*day\b": "May Mother's Day",
    r"\bfather'?s?\s*day\b": "June Father's Day",
    r"\blast\s+saturday\b": "Saturday last week",
    r"\blast\s+sunday\b": "Sunday last week",
    r"\blast\s+weekend\b": "Saturday Sunday last weekend",
}

# Question words to strip for entity-focused search
QUESTION_WORDS = re.compile(
    r"^(what|which|who|when|where|how\s+many|how\s+much|how\s+long|how|did|do|does|is|are|was|were|can|could|tell\s+me)\b\s*",
    re.IGNORECASE,
)


def expand_query(query: str) -> list[str]:
    """Generate 1-3 expanded queries from the original.

    Returns list starting with original query, followed by variants.
    """
    variants = [query]

    # 1. Date expansion: "Valentine's Day" → "February 14 Valentine"
    for pattern, expansion in DATE_PATTERNS.items():
        if re.search(pattern, query, re.IGNORECASE):
            expanded = re.sub(pattern, expansion, query, flags=re.IGNORECASE)
            if expanded != query:
                variants.append(expanded)
            break  # only one date expansion

    # 2. Entity-focused: strip question words to get the core noun phrases
    stripped = QUESTION_WORDS.sub("", query).strip()
    stripped = re.sub(r"\?$", "", stripped).strip()
    if stripped and stripped.lower() != query.lower() and len(stripped) > 10:
        variants.append(stripped)

    # 3. Key noun extraction: pull out capitalized/quoted terms
    quoted = re.findall(r'"([^"]+)"', query)
    caps = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query)
    key_terms = quoted + [c for c in caps if len(c) > 2]
    if key_terms:
        noun_query = " ".join(key_terms)
        if noun_query not in variants and len(noun_query) > 8:
            variants.append(noun_query)

    # 4. For "X from Y" patterns, search for Y specifically
    from_match = re.search(r"(\w+)\s+from\s+(\w+(?:\s+\w+)?)", query, re.IGNORECASE)
    if from_match:
        item, source = from_match.group(1), from_match.group(2)
        variants.append(f"{source} {item}")

    # 5. Negation-aware expansion: "why don't we use X" → also search "rejected X", "instead of X"
    negation_match = re.search(
        r"(?:why\s+)?(?:don'?t|doesn'?t|can'?t|shouldn'?t|not|never)\s+(?:we\s+)?(?:use|do|apply|choose|pick)\s+(.+?)(?:\?|$)",
        query, re.IGNORECASE,
    )
    if negation_match:
        rejected_thing = negation_match.group(1).strip().rstrip("?")
        variants.append(f"rejected {rejected_thing}")
        variants.append(f"instead of {rejected_thing}")
        variants.append(f"{rejected_thing} not appropriate")
        variants.append(f"switched from {rejected_thing}")

    # 6. Sufficiency/should queries
    sufficiency_match = re.search(
        r"(?:is|are)\s+(.+?)\s+(?:sufficient|enough|adequate|the right|correct)\s*\??$",
        query, re.IGNORECASE,
    )
    if sufficiency_match:
        thing = sufficiency_match.group(1).strip()
        variants.append(f"{thing} not sufficient")
        variants.append(f"{thing} validation")

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for v in variants:
        v_lower = v.lower().strip()
        if v_lower not in seen:
            seen.add(v_lower)
            unique.append(v)

    return unique[:6]  # max 6 variants
