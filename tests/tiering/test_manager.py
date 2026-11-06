"""分层存储管理器第一增量的单测：正常 / 边界 / 异常三类，全部主机可跑。

重点钉的不变量（登记处 = docs/design/分层存储概念与设计基础.md 第五节，命名空间 =
分层存储子系统内部，不是池的 I-n，也不是总组任务号 T-0xx）：
- T-1 双指标：单侧水位不触发换出 / 预取；设备侧三键不可用 ⇒ 报错不启
- T-2 抖动抑制：滞回带校验不夹取；冷却窗不挡真实召回；四件禁整项关闭
- T-3 永不下放清单 R1..R5
- T-4 三种静默退化全被拒：换入失败当未命中 / 部分换出报成功 / 空操作报成功
- T-5 管理器本体不 import 框架、不 import offload 实现（源码扫描）
- T-6 搬运流未登记就跨流搬运 ⇒ 报错
- T-7 账本先落级二后 free_block；完成确认后才交块；拷贝中途失败 ⇒ 级二查不到、块留在用
- T-8 级二容量 ≤ 级一池容量 ⇒ 报错不启
- T-11 无已注册适配层 ⇒ 显式不可用
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import (
    BLOCK_BYTES,
    DEFAULT_POLICY,
    DEVICE_TOTAL,
    LEVEL2_CAP,
    alloc_kv_blocks,
)
from tier_fakes import GRANULARITY_BYTES, FakeTierBackend

from flagmemory.caps.fake import FakeStream
from flagmemory.pool import (
    BlockState,
    PoolForm,
    Purpose,
    PressureLevel,
    TierSpec,
    WorkloadType,
    create_pool,
)
from flagmemory.pool.types import Watermark
from flagmemory.tiering import manager as manager_module
from flagmemory.tiering.manager import (
    CooldownRejected,
    CrossStreamUndeclared,
    PrefetchNotTriggered,
    PromoteCapacityExhausted,
    PromoteResult,
    TierConfigError,
    TierCopyFailed,
    TierError,
    TierLevel,
    TierPolicy,
    TierUnavailable,
    create_tier_manager,
    destroy_tier_manager,
)

#: 用例几何：设备 4 MiB = 8 个 512 KiB 基档块；级二 8 MiB = 16 个基档块
N_BLOCKS = DEVICE_TOTAL // BLOCK_BYTES


def _full_pool_blocks(pool, tier_backend):
    """铺满设备侧 ⇒ 设备占用与引擎占用双高（E1 触发路可用）。"""
    return alloc_kv_blocks(pool, tier_backend, N_BLOCKS)


def _broken_watermark() -> Watermark:
    """设备侧三键不可用的水位（总量为 0 = 后端查询不可用）。"""
    return Watermark(
        device_total=0,
        device_used=0,
        device_free=0,
        pool_in_use=1,
        pool_reusable=0,
        pool_pending=0,
        pool_quarantined=0,
    )


# ======================================================================
# 正常
# ======================================================================


class TestNormal:
    def test_创建成功_能力视图恒可用(self, manager):
        cap = manager.query_tier_capability()
        assert cap.available
        assert cap.backend_id == "kunlunxin-like"
        assert cap.transfer_granularity_bytes == GRANULARITY_BYTES
        assert cap.effective_policy is not None

    def test_默认阈值由能力层推导且可查_T9(self, pool, tier_backend):
        m = create_tier_manager(
            pool, tier_backend, level2_capacity_bytes=LEVEL2_CAP, policy=None
        )
        cap = m.query_tier_capability()
        policy = cap.effective_policy
        # 推导式：带宽 = 粒度×N÷基数；H = 1−带宽；L = H−带宽（滞回带在默认值上天然成立）
        band = cap.transfer_granularity_bytes / DEVICE_TOTAL
        assert policy.h_dev == pytest.approx(1.0 - band)
        assert policy.l_dev == pytest.approx(policy.h_dev - band)
        assert policy.l_dev <= policy.h_dev - band

    def test_换出成功_账本有条目_块归还池(self, pool, tier_backend, manager):
        blocks = _full_pool_blocks(pool, tier_backend)
        result = manager.evict([b.block_id for b in blocks], target_bytes=BLOCK_BYTES)
        assert result.evicted_bytes == BLOCK_BYTES
        assert result.evicted_entries == (f"b{blocks[0].block_id}",)
        assert result.met_target and result.reason
        assert result.ordering_basis
        wm = manager.query_tier_watermark()
        assert wm.level2.entry_count == 1 and wm.level2.used_bytes == BLOCK_BYTES
        # 块已归还池：在用量下降、进了待回收/可复用（block_view 只认活租约，不适用于归还后）
        assert wm.level1.pool_pending + wm.level1.pool_reusable > 0
        assert wm.level1.pool_in_use < wm.level1.pool_total
        audit = manager.query_tier_audit()
        assert audit.evicted_entries == 1 and audit.evicted_bytes == BLOCK_BYTES

    def test_换入成功_内容一致_默认不保留副本(self, pool, tier_backend, manager):
        blocks = _full_pool_blocks(pool, tier_backend)
        manager.evict([b.block_id for b in blocks], target_bytes=BLOCK_BYTES)
        entry_id = f"b{blocks[0].block_id}"
        outcome = manager.promote(entry_id)
        assert isinstance(outcome, PromoteResult)
        assert outcome.nbytes == BLOCK_BYTES and outcome.kept_tier2_copy is False
        # 内容一致（fake 把级二字节写回级一模拟）
        payload = tier_backend.device_blocks[outcome.block_id]
        assert payload == b"kv-payload-0" * (BLOCK_BYTES // 16)
        # 条目出级二账本，主机侧副本已释放
        assert manager.query_tier_watermark().level2.entry_count == 0
        assert entry_id in tier_backend.dropped
        audit = manager.query_tier_audit()
        assert audit.promoted_entries == 1 and audit.promoted_bytes == BLOCK_BYTES

    def test_换入保留级二副本_可配策略(self, pool, tier_backend, manager):
        blocks = _full_pool_blocks(pool, tier_backend)
        manager.evict([b.block_id for b in blocks], target_bytes=BLOCK_BYTES)
        outcome = manager.promote(f"b{blocks[0].block_id}", keep_tier2_copy=True)
        assert isinstance(outcome, PromoteResult) and outcome.kept_tier2_copy
        assert manager.query_tier_watermark().level2.entry_count == 1

    @pytest.mark.parametrize(
        ("purpose", "rule"),
        [
            (Purpose.WEIGHT, "R1"),
            (Purpose.COMM_BUFFER, "R2"),
            (Purpose.UNDECLARED, "R3"),
            (Purpose.SCRATCH, "R4"),
            (Purpose.ACTIVATION, "R5"),
        ],
    )
    def test_放置判定_永不下放五类_T3(self, manager, purpose, rule):
        block = manager._pool.alloc_on_stream(BLOCK_BYTES, purpose=purpose)
        verdict = manager.classify_placement(block_id=block.block_id)
        assert verdict.target is TierLevel.TIER1 and verdict.rule == rule

    def test_放置判定_KV_双高水位走换出_R7_降级可查(
        self, pool, tier_backend, manager
    ):
        blocks = _full_pool_blocks(pool, tier_backend)
        verdict = manager.classify_placement(block_id=blocks[0].block_id)
        # 判据 4/5 没给 stats ⇒ 退用其余判据并标降级（不把频度当 0）
        assert verdict.rule == "R7" and verdict.target is TierLevel.TIER2
        assert verdict.degraded and not verdict.reading.stats_available

    def test_放置判定_水位未到留级一_R8(self, pool, tier_backend, manager):
        alloc_kv_blocks(pool, tier_backend, 2)  # 设备占用 0.125 < 0.5
        block = manager._pool.alloc_on_stream(
            BLOCK_BYTES, purpose=Purpose.KV_CACHE
        )
        verdict = manager.classify_placement(block_id=block.block_id)
        assert verdict.rule == "R8" and verdict.target is TierLevel.TIER1

    def test_放置判定_冷前热后_R9_R10_R12(self, pool, tier_backend, manager):
        blocks = _full_pool_blocks(pool, tier_backend)
        cold, hot = blocks[1].block_id, blocks[2].block_id
        tier_backend.stats[cold] = manager_module.TierEntryStats(
            hits=0, misses=3, last_access_seq=1, current_seq=100
        )
        tier_backend.stats[hot] = manager_module.TierEntryStats(
            hits=5, misses=0, last_access_seq=100, current_seq=100
        )
        assert manager.classify_placement(block_id=cold).rule == "R9"
        # 生成式负载 + 热条目 ⇒ R12（换出序里排在 R9 之后，即 R10 的语义位）
        assert manager.classify_placement(block_id=hot).rule == "R12"
        result = manager.evict([hot, cold], target_bytes=2 * BLOCK_BYTES)
        assert result.ordering_basis[0] == "R9"
        assert result.ordering_basis[1] == "R10"

    def test_放置判定_向量化负载热条目留级一_R11(self, device_backend, tier_backend):
        pool = create_pool(
            device_backend,
            form=PoolForm.HANDLE,
            workload=WorkloadType.EMBEDDING,
            tier_spec=TierSpec(base_bytes=BLOCK_BYTES, step_bytes=4096, tier_count=6),
        )
        manager = create_tier_manager(
            pool,
            tier_backend,
            level2_capacity_bytes=LEVEL2_CAP,
            policy=DEFAULT_POLICY,
        )
        blocks = alloc_kv_blocks(pool, tier_backend, N_BLOCKS)
        tier_backend.stats[blocks[0].block_id] = manager_module.TierEntryStats(
            hits=5, misses=0, last_access_seq=1, current_seq=1
        )
        verdict = manager.classify_placement(block_id=blocks[0].block_id)
        assert verdict.rule == "R11" and verdict.target is TierLevel.TIER1

    def test_放置判定_级二条目_引擎命中换入_R14(self, pool, tier_backend, manager):
        blocks = _full_pool_blocks(pool, tier_backend)
        manager.evict([b.block_id for b in blocks], target_bytes=BLOCK_BYTES)
        verdict = manager.classify_placement(entry_id=f"b{blocks[0].block_id}")
        assert verdict.rule == "R14" and verdict.target is TierLevel.TIER1

    def test_放置判定_级二到高位_R13_级三未实现(self, pool, tier_backend, manager):
        manager.set_tier_policy(
            TierPolicy(**{**DEFAULT_POLICY.__dict__, "level2_high_watermark": 0.05})
        )
        blocks = _full_pool_blocks(pool, tier_backend)
        manager.evict([b.block_id for b in blocks], target_bytes=BLOCK_BYTES)
        verdict = manager.classify_placement(entry_id=f"b{blocks[0].block_id}")
        assert verdict.rule == "R13" and verdict.target is TierLevel.TIER3
        assert "级三未实现" in verdict.note

    def test_水位查询_级一两侧_级二账本_级三未实现(self, manager):
        wm = manager.query_tier_watermark()
        assert wm.level1.device_total == DEVICE_TOTAL
        assert wm.level1.pool_in_use >= 0  # 引擎侧四项同给（T-1 两侧都给）
        assert wm.level2.capacity_bytes == LEVEL2_CAP
        assert "未实现" in wm.level3

    def test_改策略_生效值立即可查(self, manager):
        new_policy = TierPolicy(h_dev=0.6, h_eng=0.6, l_dev=0.1, l_eng=0.1)
        effective = manager.set_tier_policy(new_policy)
        assert effective is new_policy
        assert manager.query_tier_capability().effective_policy.h_dev == 0.6

    def test_登记后端_声明探测不到_按降级上报不静默_T4(self, pool):
        degrading = FakeTierBackend(
            declared_pinned=True,
            probe_pinned=False,
            declared_batch=True,
            probe_batch=False,
        )
        manager = create_tier_manager(
            pool, degrading, level2_capacity_bytes=LEVEL2_CAP, policy=DEFAULT_POLICY
        )
        registration = manager.register_tier_backend("kunlunxin-like", degrading)
        keys = {notice.key for notice in registration.degradations}
        assert keys == {"tier.pinned-unavailable", "tier.batch-copy-unavailable"}
        cap = manager.query_tier_capability()
        assert {notice.key for notice in cap.degradations} == keys
        assert cap.probed is not None and not cap.probed.supports_pinned

    def test_销毁_返回级二残留条目清单(self, pool, tier_backend, manager):
        blocks = _full_pool_blocks(pool, tier_backend)
        manager.evict([b.block_id for b in blocks], target_bytes=2 * BLOCK_BYTES)
        report = destroy_tier_manager(manager)
        assert len(report.residual_entries) == 2 and report.in_flight == ()
        with pytest.raises(TierError):
            manager.evict([], target_bytes=1, pressure=PressureLevel.HIGH)


# ======================================================================
# 边界
# ======================================================================


class TestBoundary:
    @pytest.mark.parametrize(
        "level2", [DEVICE_TOTAL, DEVICE_TOTAL - 1], ids=["等于池容量", "小于池容量"]
    )
    def test_级二容量不大于级一池容量_报错不启_T8(self, pool, tier_backend, level2):
        with pytest.raises(TierConfigError, match="T-8"):
            create_tier_manager(
                pool,
                tier_backend,
                level2_capacity_bytes=level2,
                policy=DEFAULT_POLICY,
            )

    def test_级二容量刚好大一字节_可启_T8(self, pool, tier_backend):
        manager = create_tier_manager(
            pool,
            tier_backend,
            level2_capacity_bytes=DEVICE_TOTAL + 1,
            policy=DEFAULT_POLICY,
        )
        assert manager.query_tier_capability().available

    def test_设备侧三键不可用_报错不启_T1(self, pool, tier_backend, monkeypatch):
        monkeypatch.setattr(pool, "query_watermark", lambda: _broken_watermark())
        with pytest.raises(TierConfigError, match="T-1"):
            create_tier_manager(
                pool,
                tier_backend,
                level2_capacity_bytes=LEVEL2_CAP,
                policy=DEFAULT_POLICY,
            )

    def test_水位查询_设备侧三键不可用_报错_T1(self, manager, monkeypatch):
        monkeypatch.setattr(manager._pool, "query_watermark", lambda: _broken_watermark())
        with pytest.raises(TierConfigError, match="T-1"):
            manager.query_tier_watermark()

    def test_单侧高位不触发换出_T1(self, pool, tier_backend):
        # 引擎侧占用 100%（全在用）但设备侧只占 1/4 ⇒ 双指标不同时到高位 ⇒ 不换出
        alloc_kv_blocks(pool, tier_backend, 2)
        manager = create_tier_manager(
            pool, tier_backend, level2_capacity_bytes=LEVEL2_CAP, policy=DEFAULT_POLICY
        )
        block = pool.alloc_on_stream(BLOCK_BYTES, purpose=Purpose.KV_CACHE)
        result = manager.evict([block.block_id], target_bytes=BLOCK_BYTES)
        assert result.evicted_bytes == 0 and not result.met_target
        assert "双指标" in result.reason

    def test_级二容量不足_部分换出不报成功_T4(
        self, pool, tier_backend, manager, monkeypatch
    ):
        # T-8 要求级二 > 级一，而候选总量 ≤ 级一，真实几何下天然装得下；
        # 这里把级二账本上限拧小到一块，专钉「到量即停 + 如实标注未达目标」的行为
        monkeypatch.setattr(manager, "_level2_capacity", BLOCK_BYTES)
        blocks = _full_pool_blocks(pool, tier_backend)
        result = manager.evict(
            [b.block_id for b in blocks], target_bytes=3 * BLOCK_BYTES
        )
        assert result.evicted_bytes == BLOCK_BYTES and not result.met_target
        assert "级二容量不足" in result.reason
        assert manager.query_tier_audit().unmet_target_evicts == 1

    def test_全部候选属永不下放标签_空操作不报成功_T4_T3(
        self, pool, tier_backend, manager
    ):
        blocks = [
            pool.alloc_on_stream(BLOCK_BYTES, purpose=Purpose.WEIGHT) for _ in range(2)
        ]
        # 走 E2 压力路（不查水位），专钉「R1..R5 标签不进候选 ⇒ 空操作不报成功」
        result = manager.evict(
            [b.block_id for b in blocks],
            target_bytes=BLOCK_BYTES,
            pressure=PressureLevel.HIGH,
        )
        assert result.evicted_bytes == 0 and not result.met_target
        assert "R1..R5" in result.reason

    def test_搬运预算闸_窗口到上限即停_可审计(self, pool, tier_backend):
        policy = TierPolicy(**{**DEFAULT_POLICY.__dict__, "budget_bytes": BLOCK_BYTES})
        manager = create_tier_manager(
            pool, tier_backend, level2_capacity_bytes=LEVEL2_CAP, policy=policy
        )
        blocks = _full_pool_blocks(pool, tier_backend)
        first = manager.evict([b.block_id for b in blocks], target_bytes=BLOCK_BYTES)
        assert first.met_target
        second = manager.evict(
            [b.block_id for b in blocks[1:]], target_bytes=BLOCK_BYTES
        )
        assert second.evicted_bytes == 0 and not second.met_target
        assert "预算闸" in second.reason
        assert manager.query_tier_audit().budget_gate_deferred >= 1

    def test_冷却窗拒绝预取_但不挡真实召回_T2(self, pool, tier_backend):
        # 先铺 2 块（池总量 1 MiB，滞回带按这个基数算），再放宽引擎侧低水位到 0.5
        blocks = alloc_kv_blocks(pool, tier_backend, 2)
        policy = TierPolicy(
            **{**DEFAULT_POLICY.__dict__, "l_dev": 0.3, "l_eng": 0.5, "cooldown_s": 3600.0}
        )
        manager = create_tier_manager(
            pool, tier_backend, level2_capacity_bytes=LEVEL2_CAP, policy=policy
        )
        manager.evict([blocks[0].block_id], pressure=PressureLevel.HIGH)
        entry_id = f"b{blocks[0].block_id}"
        # I2 预取：刚换出 ⇒ 冷却窗内被拒
        rejected = manager.promote(entry_id, prefetch=True)
        assert isinstance(rejected, CooldownRejected) and rejected.remaining_s > 0
        assert manager.query_tier_audit().cooldown_rejected_prefetch == 1
        # I1 真实请求召回：不受冷却限制（冷却挡正确性路径 = 静默退化）
        assert isinstance(manager.promote(entry_id), PromoteResult)

    def test_预取要求双指标低水位_T1(self, pool, tier_backend, manager):
        blocks = alloc_kv_blocks(pool, tier_backend, 2)
        manager.evict([blocks[0].block_id], pressure=PressureLevel.HIGH)
        # 引擎侧占用 0.5 > l_eng=0.2 ⇒ 两侧不同时低 ⇒ 不预取
        outcome = manager.promote(f"b{blocks[0].block_id}", prefetch=True)
        assert isinstance(outcome, PrefetchNotTriggered)

    def test_滞回带边界_刚好压线可过_T2(self, pool, tier_backend, manager):
        blocks = alloc_kv_blocks(pool, tier_backend, 2)  # pool_total = 1 MiB
        band_dev = GRANULARITY_BYTES / DEVICE_TOTAL
        band_eng = GRANULARITY_BYTES / (2 * BLOCK_BYTES)
        ok = TierPolicy(
            h_dev=0.6,
            h_eng=0.6,
            l_dev=0.6 - band_dev,
            l_eng=0.6 - band_eng,
        )
        assert manager.set_tier_policy(ok).l_dev == pytest.approx(0.6 - band_dev)
        assert blocks  # 铺块只为把池总量抬到有意义的基数

    def test_最小驻留_刚换入的块不作候选_E2可越窗_计审计(
        self, pool, tier_backend
    ):
        policy = TierPolicy(
            **{**DEFAULT_POLICY.__dict__, "min_residency_s": 3600.0}
        )
        manager = create_tier_manager(
            pool, tier_backend, level2_capacity_bytes=LEVEL2_CAP, policy=policy
        )
        blocks = _full_pool_blocks(pool, tier_backend)
        manager.evict([b.block_id for b in blocks], target_bytes=BLOCK_BYTES)
        outcome = manager.promote(f"b{blocks[0].block_id}")
        assert isinstance(outcome, PromoteResult)
        # 普通换出（E1）：被最小驻留挡下（E1 条件⑤）
        deferred = manager.evict([outcome.block_id], target_bytes=BLOCK_BYTES)
        assert deferred.evicted_bytes == 0
        assert manager.query_tier_audit().min_residency_deferred == 1
        # E2 压力回收：可越窗但计审计（3.3 第 3 条）
        bypassed = manager.evict(
            [outcome.block_id],
            pressure=PressureLevel.HIGH,
            allow_bypass_min_residency=True,
        )
        assert bypassed.bypassed_residency == 1
        assert manager.query_tier_audit().bypass_window_evicts == 1

    def test_登记后端_标识与池后端不一致_报错(self, manager, tier_backend):
        with pytest.raises(TierError, match="不一致"):
            manager.register_tier_backend("some-other-backend", tier_backend)

    def test_创建时后端标识与池后端不一致_报错不启(self, pool):
        odd = FakeTierBackend(backend_id="mismatched")
        with pytest.raises(TierError, match="不一致"):
            create_tier_manager(
                pool, odd, level2_capacity_bytes=LEVEL2_CAP, policy=DEFAULT_POLICY
            )


# ======================================================================
# 异常
# ======================================================================


class TestError:
    def test_无已注册适配层_报错不启_T11(self, pool):
        with pytest.raises(TierUnavailable, match="T-11"):
            create_tier_manager(
                pool, None, level2_capacity_bytes=LEVEL2_CAP, policy=DEFAULT_POLICY
            )

    def test_换入池容量不足_报错并附双指标水位_不许当未命中_T4(
        self, pool, device_backend, tier_backend, manager
    ):
        blocks = alloc_kv_blocks(pool, tier_backend, N_BLOCKS - 1)  # 3.5 MiB
        # 池外占掉最后 512 KiB：设备彻底满，池没有任何可复用块可给换入
        device_backend.alloc_handle(BLOCK_BYTES, 0, "device", "external", 0)
        manager.evict([blocks[0].block_id], pressure=PressureLevel.HIGH)
        # 把换出腾出的可复用块用掉（模拟运行支撑的正常分配）
        pool.alloc_on_stream(BLOCK_BYTES, purpose=Purpose.KV_CACHE)
        with pytest.raises(PromoteCapacityExhausted) as excinfo:
            manager.promote(f"b{blocks[0].block_id}")
        assert excinfo.value.watermark.device_total == DEVICE_TOTAL  # 双指标带在错误里
        assert excinfo.value.watermark.pool_in_use > 0

    def test_换出拷贝中途失败_条目不落账本_块留在用_T7(
        self, pool, tier_backend, manager
    ):
        blocks = _full_pool_blocks(pool, tier_backend)
        victim = blocks[3].block_id
        tier_backend.fail_copy_out = True
        with pytest.raises(TierCopyFailed):
            manager.evict([victim], target_bytes=BLOCK_BYTES)
        # 条目没落级二账本 ⇒ 级二查不到（fake 主机内存里那半截字节不算数，账本是权威）
        assert manager.query_tier_watermark().level2.entry_count == 0
        assert f"b{victim}" in tier_backend.storage
        # 块没归还池：仍在用
        assert pool.block_view(victim).state is BlockState.IN_USE
        assert manager.query_tier_audit().copy_failures == 1

    def test_换出完成确认不通过_同拷贝失败处置_T7(
        self, pool, tier_backend, manager
    ):
        blocks = _full_pool_blocks(pool, tier_backend)
        victim = blocks[3].block_id
        tier_backend.never_confirm = True
        with pytest.raises(TierCopyFailed):
            manager.evict([victim], target_bytes=BLOCK_BYTES)
        assert manager.query_tier_watermark().level2.entry_count == 0
        assert pool.block_view(victim).state is BlockState.IN_USE

    def test_账本先落级二_后free_block_顺序由池侧旁证_T7(
        self, pool, tier_backend, manager, monkeypatch
    ):
        blocks = _full_pool_blocks(pool, tier_backend)
        victim = blocks[2].block_id
        original_free = pool.free_block

        def free_block_must_see_ledger(block_id: int):
            entry_id = f"b{block_id}"
            assert entry_id in manager._entries, (
                f"T-7 顺序违反：free_block({block_id}) 时条目 {entry_id} 还没落级二账本"
            )
            return original_free(block_id)

        monkeypatch.setattr(pool, "free_block", free_block_must_see_ledger)
        assert manager.evict([victim], target_bytes=BLOCK_BYTES).evicted_bytes > 0

    def test_换入拷贝失败_归还已分的块_报错_35(self, pool, tier_backend, manager):
        blocks = _full_pool_blocks(pool, tier_backend)
        manager.evict([b.block_id for b in blocks], target_bytes=BLOCK_BYTES)
        tier_backend.fail_copy_in = True
        with pytest.raises(TierCopyFailed):
            manager.promote(f"b{blocks[0].block_id}")
        # 已分的块归还了池（不再在用），级二条目还在（可重试）
        wm = manager.query_tier_watermark()
        assert wm.level2.entry_count == 1
        assert wm.level1.pool_in_use < wm.level1.pool_total

    def test_搬运流未登记_跨流搬运报错_T6(self, pool, tier_backend, manager):
        blocks = _full_pool_blocks(pool, tier_backend)
        with pytest.raises(CrossStreamUndeclared, match="T-6"):
            manager.evict(
                [b.block_id for b in blocks], target_bytes=BLOCK_BYTES, stream_id=99
            )

    def test_登记过的流可用_跨流声明计入审计_T6(self, pool, tier_backend, manager):
        stream_id = manager.attach_stream(FakeStream(stream_id=7))  # 与默认流不同
        blocks = _full_pool_blocks(pool, tier_backend)
        result = manager.evict(
            [b.block_id for b in blocks], target_bytes=BLOCK_BYTES, stream_id=stream_id
        )
        assert result.evicted_bytes == BLOCK_BYTES
        assert manager.query_tier_audit().cross_stream_declarations >= 1

    def test_滞回带违反_报错不夹取_T2_T4(self, manager):
        bad = TierPolicy(h_dev=0.5, h_eng=0.5, l_dev=0.49, l_eng=0.49)
        with pytest.raises(TierConfigError, match="滞回带"):
            manager.set_tier_policy(bad)
        # 不夹取：生效值仍是原策略
        assert manager.query_tier_capability().effective_policy.l_dev == 0.2

    @pytest.mark.parametrize(
        "override",
        [
            {"cooldown_s": 0.0},
            {"min_residency_s": 0.0},
            {"budget_bytes": 0},
            {"backoff_trigger_max": 0},
            {"hysteresis_n": 0},
        ],
        ids=["关冷却窗", "关最小驻留", "关预算闸", "关空转退避", "N<1"],
    )
    def test_整项关闭抖动抑制_报错_T2(self, manager, override):
        bad = TierPolicy(**{**DEFAULT_POLICY.__dict__, **override})
        with pytest.raises(TierConfigError, match="T-2"):
            manager.set_tier_policy(bad)

    def test_换入不存在的条目_报错_不当未命中_T4(self, manager):
        with pytest.raises(TierError, match="不在级二"):
            manager.promote("b-nope")

    def test_放置判定_块与条目都不给_报错(self, manager):
        with pytest.raises(TierError):
            manager.classify_placement()

    def test_放置判定_级二没有该条目_报错(self, manager):
        with pytest.raises(TierError):
            manager.classify_placement(entry_id="b-nope")

    def test_能力口销毁后仍可用_降级可观测基础(self, manager):
        destroy_tier_manager(manager)
        cap = manager.query_tier_capability()
        assert not cap.available and "销毁" in cap.unavailable_reason


# ======================================================================
# 结构纪律（T-5）
# ======================================================================


class TestLayering:
    """T-5：管理器本体不 import 推理框架、不 import offload 的任何实现。"""

    SOURCE = Path(manager_module.__file__).read_text(encoding="utf-8")

    def test_不import推理框架(self):
        for banned in ("import torch", "import vllm", "from torch", "from vllm"):
            assert banned not in self.SOURCE

    def test_不import_offload实现(self):
        for banned in (
            "import flagmemory.tiering.offload",
            "from flagmemory.tiering.offload",
            "from .offload",
            "from ..offload",
        ):
            assert banned not in self.SOURCE, f"管理器本体出现 {banned!r}"

    def test_不出现芯片分支(self):
        for chip in ("910", "ascend", "Ascend", "kunlun", "Kunlun", "mlu", "MLU590", "ppu"):
            assert chip not in self.SOURCE, f"管理器本体出现了芯片字样 {chip!r}"
