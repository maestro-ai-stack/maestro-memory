"""Benchmark test scenarios covering 9 use cases for maestro-memory."""
from __future__ import annotations

from pathlib import Path

import pytest

from maestro_memory.core.memory import Memory


# ── Helpers ──────────────────────────────────────────────────────


async def fresh_memory(tmp_path: Path, name: str = "bench.db") -> Memory:
    """Create and initialize a temp Memory instance"""
    m = Memory(path=tmp_path / name)
    await m.init()
    return m


def contents(results) -> list[str]:
    """Extract text list from search results"""
    return [r.fact.content for r in results]


def any_match(results, keyword: str) -> bool:
    """Check if any result contains the keyword"""
    return any(keyword.lower() in c.lower() for c in contents(results))


# ── 1. Multi-turn extraction ─────────────────────────────────────


@pytest.mark.asyncio
async def test_multiturn_extraction(tmp_path: Path) -> None:
    """Facts from multiple add() calls should all be retrievable"""
    mem = await fresh_memory(tmp_path)
    try:
        facts = [
            "Python is the primary language for data science",
            "TypeScript powers the frontend dashboard",
            "PostgreSQL stores transactional data",
            "Redis handles session caching",
            "Kubernetes orchestrates all microservices",
        ]
        for f in facts:
            await mem.add(f)

        # Each fact should be retrievable by keyword
        for keyword in ["Python", "TypeScript", "PostgreSQL", "Redis", "Kubernetes"]:
            results = await mem.search(keyword, limit=5)
            assert len(results) >= 1, f"failed to retrieve fact about {keyword}"
            assert any_match(results, keyword)

        # Total facts count >= 5
        stats = await mem.status()
        assert stats["facts"] >= 5
    finally:
        await mem.close()


# ── 2. CRM extraction ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_crm_extraction(tmp_path: Path) -> None:
    """CRM scenario: search deals by company/status"""
    mem = await fresh_memory(tmp_path)
    try:
        await mem.add("Alice from Acme Corp, deal worth $50K, stage: negotiation, next: send proposal by Friday")
        await mem.add("Bob from Beta Inc, deal $20K, stage: closed-won")

        # Search by company name
        results = await mem.search("Acme Corp deal", limit=5)
        assert len(results) >= 1
        assert any_match(results, "Acme")

        # Search by deal status
        results = await mem.search("closed deals", limit=5)
        assert len(results) >= 1
        assert any_match(results, "closed-won") or any_match(results, "Bob")
    finally:
        await mem.close()


# ── 3. Calendar scheduling ────────────────────────────────────────


@pytest.mark.asyncio
async def test_calendar_scheduling(tmp_path: Path) -> None:
    """Calendar scenario: conflict detection + event update"""
    mem = await fresh_memory(tmp_path)
    try:
        await mem.add("Meeting with team on March 20 at 2pm, room 301")
        await mem.add("Dentist appointment March 20 at 3pm")

        # Searching March 20 should return two results
        results = await mem.search("March 20", limit=10)
        assert len(results) >= 2
        assert any_match(results, "team") or any_match(results, "Meeting")
        assert any_match(results, "Dentist")

        # Invalidate old meeting, add new date
        all_facts = await mem.store.list_facts(current_only=True)
        for f in all_facts:
            if "team" in f.content.lower() and "March 20" in f.content:
                await mem.store.invalidate_fact(f.id)
        await mem.add("Team meeting moved to March 21 at 2pm, room 301")

        # March 20 should now only have the dentist
        results = await mem.search("March 20", current_only=True, limit=10)
        for r in results:
            assert "team" not in r.fact.content.lower() or "March 21" in r.fact.content
    finally:
        await mem.close()


# ── 4. Travel planning ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_travel_planning(tmp_path: Path) -> None:
    """Travel scenario: preference + itinerary retrieval"""
    mem = await fresh_memory(tmp_path)
    try:
        await mem.add("Travel preference: prefer window seat")
        await mem.add("Travel budget: $3000 total")
        await mem.add("Food allergy: allergic to shellfish")
        await mem.add("Itinerary: Tokyo March 15-20, Osaka March 20-23")

        # Food restrictions
        results = await mem.search("food restrictions", limit=5)
        assert len(results) >= 1
        assert any_match(results, "shellfish")

        # Itinerary retrieval (BM25 needs lexical overlap, use itinerary keyword)
        results = await mem.search("Tokyo Osaka itinerary", limit=5)
        assert len(results) >= 1
        assert any_match(results, "Tokyo") or any_match(results, "Osaka")
    finally:
        await mem.close()


# ── 5. Patient records ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_patient_records(tmp_path: Path) -> None:
    """Medical scenario: symptom updates + current_only filtering"""
    mem = await fresh_memory(tmp_path)
    try:
        await mem.add("Patient reports headache for 3 days, no fever")
        await mem.add("Prescribed ibuprofen 400mg twice daily")

        # Invalidate old symptom
        all_facts = await mem.store.list_facts(current_only=True)
        for f in all_facts:
            if "headache" in f.content.lower():
                await mem.store.invalidate_fact(f.id)

        await mem.add("Follow-up: headache resolved, new symptom: dizziness")

        # current_only=True should find dizziness, not old headache
        # Use follow-up/dizziness keywords to ensure BM25 hit
        results = await mem.search("headache dizziness follow-up", current_only=True, limit=10)
        active_contents = " ".join(contents(results)).lower()
        assert "dizziness" in active_contents
    finally:
        await mem.close()


# ── 6. HR candidate ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hr_candidate(tmp_path: Path) -> None:
    """HR scenario: entity resolution + interview feedback retrieval"""
    mem = await fresh_memory(tmp_path)
    try:
        await mem.add("John Smith, 5 years Python, applied for Senior Engineer")
        await mem.add("J. Smith phone screen: strong system design, weak on concurrency")

        # Search by last name should return both
        results = await mem.search("Smith", limit=10)
        assert len(results) >= 2
        assert any_match(results, "John Smith") or any_match(results, "J. Smith")

        # Interview feedback (BM25 needs lexical overlap, use phone screen keyword)
        results = await mem.search("phone screen design", limit=5)
        assert len(results) >= 1
        assert any_match(results, "phone screen") or any_match(results, "system design")
    finally:
        await mem.close()


# ── 7. Paper management ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_paper_management(tmp_path: Path) -> None:
    """Paper scenario: semantic linking retrieval"""
    mem = await fresh_memory(tmp_path)
    try:
        await mem.add("Attention Is All You Need (Vaswani et al. 2017) - introduces transformer architecture")
        await mem.add("BERT (Devlin et al. 2019) - bidirectional pre-training, builds on transformers")

        # Search transformer should return both papers
        # BM25 FTS5 uses OR: transformer hits first, transformers hits second
        results = await mem.search("transformer transformers architecture", limit=5)
        assert len(results) >= 2
        assert any_match(results, "Attention")
        assert any_match(results, "BERT")
    finally:
        await mem.close()


# ── 8. Research tracking ────────────────────────────────────────


@pytest.mark.asyncio
async def test_research_tracking(tmp_path: Path) -> None:
    """Research scenario: questions/findings/dead-ends all traceable"""
    mem = await fresh_memory(tmp_path)
    try:
        await mem.add("Research Q: Does spaced repetition improve LLM memory?")
        await mem.add("Found: Ebbinghaus curve applies to embedding decay")
        await mem.add("Dead end: simple rehearsal doesn't help, need structured extraction")

        results = await mem.search("spaced repetition findings", limit=10)
        assert len(results) >= 1
        # Should retrieve at least the research question or finding
        combined = " ".join(contents(results)).lower()
        assert "spaced repetition" in combined or "ebbinghaus" in combined
    finally:
        await mem.close()


# ── 9. Work project from chat ───────────────────────────────────


@pytest.mark.asyncio
async def test_work_project_from_chat(tmp_path: Path) -> None:
    """Chat-style input: deploy status tracking"""
    mem = await fresh_memory(tmp_path)
    try:
        await mem.add("hey so the deploy is blocked bc jenkins is down, @mike is looking into it, eta 2hrs")
        await mem.add("update: jenkins back up, deploy succeeded, monitoring for 30min")

        # Deploy status should return the latest update
        results = await mem.search("deploy status", limit=5)
        assert len(results) >= 1
        assert any_match(results, "deploy")

        # Jenkins search should return both messages
        results = await mem.search("jenkins", limit=10)
        assert len(results) >= 2
    finally:
        await mem.close()


# ── Dedup effectiveness ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_dedup_effectiveness(tmp_path: Path) -> None:
    """Adding the same fact twice via consolidate should result in only 1 in DB"""
    mem = await fresh_memory(tmp_path)
    try:
        from maestro_memory.ingestion.consolidate import consolidate

        # Write two temp files with identical content
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        content = "The company was founded in 2020 in San Francisco."
        f1.write_text(content, encoding="utf-8")
        f2.write_text(content, encoding="utf-8")

        await consolidate(mem, [f1, f2], source_type="file")

        facts = await mem.store.list_facts(current_only=False)
        # Identical content should be deduped, keeping only 1
        matching = [f for f in facts if "founded" in f.content.lower()]
        assert len(matching) == 1, f"expected 1 fact, got {len(matching)}: {[f.content for f in matching]}"
    finally:
        await mem.close()


# ── Fact supersession ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_fact_supersession(tmp_path: Path) -> None:
    """After invalidating old fact, current_only search returns only the new one"""
    mem = await fresh_memory(tmp_path)
    try:
        await mem.add("Company HQ is in NYC")

        # Invalidate old fact
        facts = await mem.store.list_facts(current_only=True)
        for f in facts:
            if "NYC" in f.content:
                await mem.store.invalidate_fact(f.id)

        await mem.add("Company HQ moved to Austin")

        # current_only search should return only Austin
        results = await mem.search("company HQ", current_only=True, limit=10)
        assert len(results) >= 1
        active = contents(results)
        assert any("Austin" in c for c in active), f"Austin not found in {active}"
        assert not any("NYC" in c and "Austin" not in c for c in active), f"stale NYC fact leaked: {active}"
    finally:
        await mem.close()
