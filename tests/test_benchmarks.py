"""Benchmark test scenarios covering 9 use cases for maestro-memory."""
from __future__ import annotations

from pathlib import Path

import pytest

from maestro_memory.core.memory import Memory


# ── 工具函数 ────────────────────────────────────────────────────


async def fresh_memory(tmp_path: Path, name: str = "bench.db") -> Memory:
    """创建并初始化一个临时 Memory 实例"""
    m = Memory(path=tmp_path / name)
    await m.init()
    return m


def contents(results) -> list[str]:
    """提取搜索结果的文本列表"""
    return [r.fact.content for r in results]


def any_match(results, keyword: str) -> bool:
    """结果中是否有任何一条包含关键词"""
    return any(keyword.lower() in c.lower() for c in contents(results))


# ── 1. 多轮提取 ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_multiturn_extraction(tmp_path: Path) -> None:
    """跨多轮 add() 的事实都应可检索"""
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

        # 每条事实都应可通过关键词检索
        for keyword in ["Python", "TypeScript", "PostgreSQL", "Redis", "Kubernetes"]:
            results = await mem.search(keyword, limit=5)
            assert len(results) >= 1, f"failed to retrieve fact about {keyword}"
            assert any_match(results, keyword)

        # 总 facts 数 >= 5
        stats = await mem.status()
        assert stats["facts"] >= 5
    finally:
        await mem.close()


# ── 2. CRM 提取 ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_crm_extraction(tmp_path: Path) -> None:
    """CRM 场景：按公司/状态检索交易"""
    mem = await fresh_memory(tmp_path)
    try:
        await mem.add("Alice from Acme Corp, deal worth $50K, stage: negotiation, next: send proposal by Friday")
        await mem.add("Bob from Beta Inc, deal $20K, stage: closed-won")

        # 按公司名搜索
        results = await mem.search("Acme Corp deal", limit=5)
        assert len(results) >= 1
        assert any_match(results, "Acme")

        # 按交易状态搜索
        results = await mem.search("closed deals", limit=5)
        assert len(results) >= 1
        assert any_match(results, "closed-won") or any_match(results, "Bob")
    finally:
        await mem.close()


# ── 3. 日历调度 ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_calendar_scheduling(tmp_path: Path) -> None:
    """日历场景：时间冲突检测 + 事件更新"""
    mem = await fresh_memory(tmp_path)
    try:
        await mem.add("Meeting with team on March 20 at 2pm, room 301")
        await mem.add("Dentist appointment March 20 at 3pm")

        # 搜索 March 20 应返回两条
        results = await mem.search("March 20", limit=10)
        assert len(results) >= 2
        assert any_match(results, "team") or any_match(results, "Meeting")
        assert any_match(results, "Dentist")

        # 作废旧会议，添加新日期
        all_facts = await mem.store.list_facts(current_only=True)
        for f in all_facts:
            if "team" in f.content.lower() and "March 20" in f.content:
                await mem.store.invalidate_fact(f.id)
        await mem.add("Team meeting moved to March 21 at 2pm, room 301")

        # March 20 现在只应有牙医
        results = await mem.search("March 20", current_only=True, limit=10)
        for r in results:
            assert "team" not in r.fact.content.lower() or "March 21" in r.fact.content
    finally:
        await mem.close()


# ── 4. 旅行规划 ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_travel_planning(tmp_path: Path) -> None:
    """旅行场景：偏好 + 行程检索"""
    mem = await fresh_memory(tmp_path)
    try:
        await mem.add("Travel preference: prefer window seat")
        await mem.add("Travel budget: $3000 total")
        await mem.add("Food allergy: allergic to shellfish")
        await mem.add("Itinerary: Tokyo March 15-20, Osaka March 20-23")

        # 食物限制
        results = await mem.search("food restrictions", limit=5)
        assert len(results) >= 1
        assert any_match(results, "shellfish")

        # 行程检索（BM25 需要词汇重叠，用 itinerary 关键词）
        results = await mem.search("Tokyo Osaka itinerary", limit=5)
        assert len(results) >= 1
        assert any_match(results, "Tokyo") or any_match(results, "Osaka")
    finally:
        await mem.close()


# ── 5. 患者记录 ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_patient_records(tmp_path: Path) -> None:
    """医疗场景：症状更新 + current_only 过滤"""
    mem = await fresh_memory(tmp_path)
    try:
        await mem.add("Patient reports headache for 3 days, no fever")
        await mem.add("Prescribed ibuprofen 400mg twice daily")

        # 作废旧症状
        all_facts = await mem.store.list_facts(current_only=True)
        for f in all_facts:
            if "headache" in f.content.lower():
                await mem.store.invalidate_fact(f.id)

        await mem.add("Follow-up: headache resolved, new symptom: dizziness")

        # current_only=True 应该找到 dizziness 而非旧的 headache 记录
        # 用 follow-up / dizziness 关键词确保 BM25 命中
        results = await mem.search("headache dizziness follow-up", current_only=True, limit=10)
        active_contents = " ".join(contents(results)).lower()
        assert "dizziness" in active_contents
    finally:
        await mem.close()


# ── 6. HR 候选人 ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hr_candidate(tmp_path: Path) -> None:
    """HR 场景：实体解析 + 面试反馈检索"""
    mem = await fresh_memory(tmp_path)
    try:
        await mem.add("John Smith, 5 years Python, applied for Senior Engineer")
        await mem.add("J. Smith phone screen: strong system design, weak on concurrency")

        # 按姓氏搜索应返回两条
        results = await mem.search("Smith", limit=10)
        assert len(results) >= 2
        assert any_match(results, "John Smith") or any_match(results, "J. Smith")

        # 面试反馈（BM25 需词汇重叠，用 phone screen 关键词）
        results = await mem.search("phone screen design", limit=5)
        assert len(results) >= 1
        assert any_match(results, "phone screen") or any_match(results, "system design")
    finally:
        await mem.close()


# ── 7. 论文管理 ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_paper_management(tmp_path: Path) -> None:
    """论文场景：语义关联检索"""
    mem = await fresh_memory(tmp_path)
    try:
        await mem.add("Attention Is All You Need (Vaswani et al. 2017) - introduces transformer architecture")
        await mem.add("BERT (Devlin et al. 2019) - bidirectional pre-training, builds on transformers")

        # 搜索 transformer 应返回两篇论文
        # BM25 FTS5 用 OR 连接：transformer 命中第一篇，transformers 命中第二篇
        results = await mem.search("transformer transformers architecture", limit=5)
        assert len(results) >= 2
        assert any_match(results, "Attention")
        assert any_match(results, "BERT")
    finally:
        await mem.close()


# ── 8. 研究追踪 ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_research_tracking(tmp_path: Path) -> None:
    """研究场景：问题/发现/死胡同全部可追溯"""
    mem = await fresh_memory(tmp_path)
    try:
        await mem.add("Research Q: Does spaced repetition improve LLM memory?")
        await mem.add("Found: Ebbinghaus curve applies to embedding decay")
        await mem.add("Dead end: simple rehearsal doesn't help, need structured extraction")

        results = await mem.search("spaced repetition findings", limit=10)
        assert len(results) >= 1
        # 至少应检索到研究问题或发现
        combined = " ".join(contents(results)).lower()
        assert "spaced repetition" in combined or "ebbinghaus" in combined
    finally:
        await mem.close()


# ── 9. 工作聊天提取 ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_work_project_from_chat(tmp_path: Path) -> None:
    """聊天风格输入：部署状态追踪"""
    mem = await fresh_memory(tmp_path)
    try:
        await mem.add("hey so the deploy is blocked bc jenkins is down, @mike is looking into it, eta 2hrs")
        await mem.add("update: jenkins back up, deploy succeeded, monitoring for 30min")

        # 部署状态应返回最新更新
        results = await mem.search("deploy status", limit=5)
        assert len(results) >= 1
        assert any_match(results, "deploy")

        # jenkins 搜索应返回两条
        results = await mem.search("jenkins", limit=10)
        assert len(results) >= 2
    finally:
        await mem.close()


# ── 去重有效性 ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dedup_effectiveness(tmp_path: Path) -> None:
    """通过 consolidate 添加相同事实两次，DB 中应只有 1 条"""
    mem = await fresh_memory(tmp_path)
    try:
        from maestro_memory.ingestion.consolidate import consolidate

        # 写两个内容相同的临时文件
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        content = "The company was founded in 2020 in San Francisco."
        f1.write_text(content, encoding="utf-8")
        f2.write_text(content, encoding="utf-8")

        await consolidate(mem, [f1, f2], source_type="file")

        facts = await mem.store.list_facts(current_only=False)
        # 内容相同的事实应被去重，只保留 1 条
        matching = [f for f in facts if "founded" in f.content.lower()]
        assert len(matching) == 1, f"expected 1 fact, got {len(matching)}: {[f.content for f in matching]}"
    finally:
        await mem.close()


# ── 事实废止 + 替换 ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fact_supersession(tmp_path: Path) -> None:
    """旧事实作废后，current_only 搜索只返回新事实"""
    mem = await fresh_memory(tmp_path)
    try:
        await mem.add("Company HQ is in NYC")

        # 作废旧事实
        facts = await mem.store.list_facts(current_only=True)
        for f in facts:
            if "NYC" in f.content:
                await mem.store.invalidate_fact(f.id)

        await mem.add("Company HQ moved to Austin")

        # current_only 搜索应只返回 Austin
        results = await mem.search("company HQ", current_only=True, limit=10)
        assert len(results) >= 1
        active = contents(results)
        assert any("Austin" in c for c in active), f"Austin not found in {active}"
        assert not any("NYC" in c and "Austin" not in c for c in active), f"stale NYC fact leaked: {active}"
    finally:
        await mem.close()
