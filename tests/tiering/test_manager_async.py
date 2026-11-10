"""分层存储管理器异步在途轨 + 级三留位增量的单测：正常 / 边界 / 异常三类，全部主机可跑。

重点钉的语义（登记处 = docs/design/分层存储概念与设计基础.md，命名空间 = 分层存储
子系统内部，不是池的 I-n，也不是总组任务号 T-0xx）：
- 在途状态生命周期：PENDING →（确认）CONFIRMED 出表落账/交块；→（confirm 抛错）
  FAILED 出表回收；→（问询预算耗尽）TIMED_OUT 留表可恢复（T-7「超时不等于失败」）
- 在途块不得被二次调度 / 重复归还：重复换出 / 重复换入 / 换出在途条目换入，全被显式拒绝并计审计
- 异步完成确认才变更归属：换出「条目先落级二账本后 free_block」、换入「确认后交块」（T-7）
- 失败时序语义：发起期失败与确认期失败都按 3.5 回收（换出不落账不还块 / 换入归还预留块）
- 级三留位：R13 处置口按 LRU 丢弃并计 tier3_missing_drops；能力面 tier3_backend_registered
  恒 False（留位不是实现）；换入在途的条目不丢（H2D 数据源不得悬空）
"""

from __future__ import annotations

import pytest

from conftest import (
    BLOCK_BYTES,
    DEFAULT_POLICY,
    DEVICE_TOTAL,
    LEVEL2_CAP,
    alloc_kv_blocks,
)

from flagmemory.pool import (
    BlockState,
    PoolForm,
    PressureLevel,
    TierSpec,
    WorkloadType,
    create_pool,
)
from flagmemory.tiering import manager as manager_module
from flagmemory.tiering.manager import (
    AsyncEvictReceipt,
    AsyncPromoteReceipt,
    PromoteResult,
    Tier3DispositionReport,
    TierCopyFailed,
    TierError,
    TierLevel,
    TierPolicy,
    TransferDirection,
    TransferState,
    create_tier_manager,
    destroy_tier_manager,
)

#: 用例几何：设备 4 MiB = 8 个 512 KiB 基档块；级二 8 MiB = 16 个基档块
N_BLOCKS = DEVICE_TOTAL // BLOCK_BYTES


def _full_pool_blocks(pool, tier_backend):
    """铺满设备侧 ⇒ 双指标到高位；E1 / E2 两条换出路都可用。"""
    return alloc_kv_blocks(pool, tier_backend, N_BLOCKS)


def _one_in_flight_out(manager, pool, tier_backend, *, index=0):
    """铺满 → 异步换出第 index 块（E2 压力路）⇒ 返回（receipt, block_id, entry_id）。"""
    blocks = _full_pool_blocks(pool, tier_backend)
    block_id = blocks[index].block_id
    receipt = manager.evict_async([block_id], pressure=PressureLevel.HIGH)
    assert isinstance(receipt, AsyncEvictReceipt) and receipt.scheduled_entries
    return receipt, block_id, f"b{block_id}"


# ======================================================================
# 正常
# ======================================================================


class TestAsyncNormal:
    def test_异步换出_发起即在途_不落账本不还块(self, pool, tier_backend, manager):
        receipt, block_id, entry_id = _one_in_flight_out(manager, pool, tier_backend)
        assert receipt.scheduled_bytes == BLOCK_BYTES
        assert receipt.met_target
        # 在途账本有明确状态与跟踪
        tickets = manager.query_transfers()
        assert len(tickets) == 1
        assert tickets[0].state is TransferState.PENDING
        assert tickets[0].direction is TransferDirection.EVICT
        assert tickets[0].entry_id == entry_id and tickets[0].block_id == block_id
        # 归属未变：条目不在级二账本，块仍在用，审计未计入换出（T-7 的异步形态）
        assert manager.query_tier_watermark().level2.entry_count == 0
        assert pool.block_view(block_id).state is BlockState.IN_USE
        assert manager.query_tier_audit().evicted_entries == 0

    def test_异步换出_轮询确认后_账本先行_块归还_T7(
        self, pool, tier_backend, manager, monkeypatch
    ):
        _, block_id, entry_id = _one_in_flight_out(manager, pool, tier_backend)
        original_free = pool.free_block

        def free_block_must_see_ledger(bid: int):
            assert f"b{bid}" in manager._entries, (
                f"T-7 顺序违反：异步确认的 free_block({bid}) 时条目还没落级二账本"
            )
            return original_free(bid)

        monkeypatch.setattr(pool, "free_block", free_block_must_see_ledger)
        report = manager.poll_transfers()
        assert len(report.confirmed) == 1 and not report.still_in_flight
        outcome = report.confirmed[0]
        assert outcome.state is TransferState.CONFIRMED and outcome.entry_id == entry_id
        # 归属此刻才变更：账本有条目、块归还池、审计计入（T-7）
        wm = manager.query_tier_watermark()
        assert wm.level2.entry_count == 1 and wm.level2.used_bytes == BLOCK_BYTES
        assert wm.level1.pool_pending + wm.level1.pool_reusable > 0
        audit = manager.query_tier_audit()
        assert audit.evicted_entries == 1 and audit.evicted_bytes == BLOCK_BYTES

    def test_异步换入_完成确认后才交块_内容一致(
        self, pool, tier_backend, manager
    ):
        _, block_id, entry_id = _one_in_flight_out(manager, pool, tier_backend)
        manager.poll_transfers()  # 换出确认，条目落级二
        receipt = manager.promote_async(entry_id)
        assert isinstance(receipt, AsyncPromoteReceipt)
        assert receipt.entry_id == entry_id and receipt.nbytes == BLOCK_BYTES
        # 回执不暴露块标识与底座引用：确认完成前不得交块（T-7 第 5 步）
        assert not hasattr(receipt, "block_id") and not hasattr(receipt, "base_ref")
        # 在途账本：换入方向
        tickets = manager.query_transfers(entry_id=entry_id)
        assert len(tickets) == 1
        assert tickets[0].direction is TransferDirection.PROMOTE
        report = manager.poll_transfers()
        assert len(report.confirmed) == 1
        promoted = report.confirmed[0].promoted
        assert promoted is not None and promoted.entry_id == entry_id
        assert promoted.nbytes == BLOCK_BYTES and promoted.kept_tier2_copy is False
        # 内容一致（fake 把级二字节写回级一模拟），条目出账本、主机侧副本释放
        payload = tier_backend.device_blocks[promoted.block_id]
        assert payload == b"kv-payload-0" * (BLOCK_BYTES // 16)
        assert manager.query_tier_watermark().level2.entry_count == 0
        assert entry_id in tier_backend.dropped
        assert manager.query_tier_audit().promoted_entries == 1

    def test_异步换入_保留级二副本(self, pool, tier_backend, manager):
        _, _, entry_id = _one_in_flight_out(manager, pool, tier_backend)
        manager.poll_transfers()
        assert isinstance(
            manager.promote_async(entry_id, keep_tier2_copy=True), AsyncPromoteReceipt
        )
        report = manager.poll_transfers()
        assert report.confirmed[0].promoted is not None
        assert report.confirmed[0].promoted.kept_tier2_copy is True
        assert manager.query_tier_watermark().level2.entry_count == 1

    def test_在途生命周期_多次轮询才确认_状态迁移(
        self, pool, tier_backend, manager
    ):
        tier_backend.confirm_after_polls = 2  # 前两次问询未完成，第三次确认
        _, block_id, _ = _one_in_flight_out(manager, pool, tier_backend)
        first = manager.poll_transfers()
        assert first.outcomes[0].state is TransferState.PENDING
        assert first.outcomes[0].polls == 1 and len(first.still_in_flight) == 1
        second = manager.poll_transfers()
        assert second.outcomes[0].state is TransferState.PENDING
        assert second.outcomes[0].polls == 2 and len(second.still_in_flight) == 1
        third = manager.poll_transfers()
        assert third.outcomes[0].state is TransferState.CONFIRMED
        assert manager.query_transfers() == ()  # 确认即出表
        assert manager.query_tier_audit().evicted_entries == 1

    def test_在途清单查询_按条目过滤(self, pool, tier_backend, manager):
        blocks = _full_pool_blocks(pool, tier_backend)
        receipt = manager.evict_async(
            [b.block_id for b in blocks[:3]], pressure=PressureLevel.HIGH
        )
        assert len(receipt.transfers) == 3
        target_entry = receipt.scheduled_entries[1]
        filtered = manager.query_transfers(entry_id=target_entry)
        assert len(filtered) == 1 and filtered[0].entry_id == target_entry
        assert len(manager.query_transfers()) == 3

    def test_销毁_等待在途排空后成功(self, pool, tier_backend, manager):
        _one_in_flight_out(manager, pool, tier_backend)
        report = destroy_tier_manager(manager)  # 默认 wait_for_drain=True
        assert report.timed_out is False and report.in_flight == ()
        assert len(report.residual_entries) == 1  # 排空中确认的那笔落了账本
        assert manager.destroyed


class TestTier3SlotNormal:
    def test_级三留位_处置_LRU丢弃_计入审计(self, pool, tier_backend, manager):
        blocks = _full_pool_blocks(pool, tier_backend)
        # 三笔先后换出 ⇒ evicted_at 递增，LRU 序 = 换出序
        for block in blocks[:3]:
            manager.evict([block.block_id], pressure=PressureLevel.HIGH)
        manager.set_tier_policy(
            TierPolicy(**{**DEFAULT_POLICY.__dict__, "level2_high_watermark": 0.15})
        )
        report = manager.apply_r13_disposition()
        assert isinstance(report, Tier3DispositionReport)
        # 留位语义：目标是级三、后端未注册（恒 False——留位不是实现）
        assert report.target is TierLevel.TIER3 and report.backend_registered is False
        # LRU 最旧先丢，丢到高水位以下即停（1.5 MiB → 阈值 1.2 MiB ⇒ 只丢最旧一笔）
        assert report.dropped_entries == (f"b{blocks[0].block_id}",)
        assert report.dropped_bytes == BLOCK_BYTES
        assert manager.query_tier_audit().tier3_missing_drops == 1
        assert f"b{blocks[0].block_id}" in tier_backend.dropped
        assert manager.query_tier_watermark().level2.entry_count == 2

    def test_级三留位_未到高水位_空处置显式给原因(self, manager):
        report = manager.apply_r13_disposition()
        assert report.dropped_entries == () and report.dropped_bytes == 0
        assert "未到高水位" in report.reason  # 空处置不报成功，原因可见（T-4）
        assert manager.query_tier_audit().tier3_missing_drops == 0

    def test_级三留位_能力面与水位面槽位(self, manager):
        cap = manager.query_tier_capability()
        # 留位槽：级三后端恒未注册；水位面级三给「未实现」；枚举有 TIER3
        assert cap.tier3_backend_registered is False
        wm = manager.query_tier_watermark()
        assert "未实现" in wm.level3
        assert TierLevel.TIER3.value.startswith("级三")


# ======================================================================
# 边界
# ======================================================================


class TestAsyncBoundary:
    def test_在途块同步换出被拒_计审计(self, pool, tier_backend, manager):
        blocks = _full_pool_blocks(pool, tier_backend)
        in_flight_id = blocks[0].block_id
        other_id = blocks[1].block_id
        assert isinstance(
            manager.evict_async([in_flight_id], pressure=PressureLevel.HIGH),
            AsyncEvictReceipt,
        )
        result = manager.evict(
            [in_flight_id, other_id], pressure=PressureLevel.HIGH
        )
        # 在途块被跳过，其余候选照常；拒绝次数进审计
        assert result.evicted_entries == (f"b{other_id}",)
        assert manager.query_tier_audit().inflight_reschedule_rejected == 1

    def test_在途块再次异步换出被拒(self, pool, tier_backend, manager):
        _, block_id, _ = _one_in_flight_out(manager, pool, tier_backend)
        second = manager.evict_async([block_id], pressure=PressureLevel.HIGH)
        assert second.scheduled_entries == () and not second.met_target
        assert "在途" in second.reason and "二次调度被拒" in second.reason
        assert manager.query_tier_audit().inflight_reschedule_rejected == 1
        assert len(manager.query_transfers()) == 1  # 没有第二笔在途

    def test_在途条目重复换入被拒(self, pool, tier_backend, manager):
        _, _, entry_id = _one_in_flight_out(manager, pool, tier_backend)
        manager.poll_transfers()
        assert isinstance(manager.promote_async(entry_id), AsyncPromoteReceipt)
        with pytest.raises(TierError, match="已有换入在途"):
            manager.promote_async(entry_id)
        with pytest.raises(TierError, match="已有换入在途"):
            manager.promote(entry_id)

    def test_换出在途的条目换入被拒_不静默当未命中(
        self, pool, tier_backend, manager
    ):
        _, _, entry_id = _one_in_flight_out(manager, pool, tier_backend)
        with pytest.raises(TierError, match="正在换出在途"):
            manager.promote(entry_id)
        with pytest.raises(TierError, match="正在换出在途"):
            manager.promote_async(entry_id)

    def test_确认超时_可恢复态_不算失败_后续仍可确认(
        self, pool, tier_backend, manager
    ):
        _, block_id, _ = _one_in_flight_out(manager, pool, tier_backend)
        tier_backend.never_confirm = True
        for _ in range(manager_module._PENDING_TIMEOUT_POLLS):
            report = manager.poll_transfers()
            assert report.outcomes[0].state in (
                TransferState.PENDING,
                TransferState.TIMED_OUT,
            )
        # 问询预算耗尽 ⇒ 确认超时（可恢复）：留在途表、不算失败、归属未变
        tickets = manager.query_transfers()
        assert len(tickets) == 1 and tickets[0].state is TransferState.TIMED_OUT
        assert manager.query_tier_audit().copy_failures == 0
        assert manager.query_tier_watermark().level2.entry_count == 0
        assert pool.block_view(block_id).state is BlockState.IN_USE
        # 后端恢复 ⇒ 下一轮问询即确认（可恢复态的真实含义）
        tier_backend.never_confirm = False
        assert len(manager.poll_transfers().confirmed) == 1
        assert manager.query_tier_watermark().level2.entry_count == 1

    def test_销毁_排空超时_可恢复上报_不标销毁(self, pool, tier_backend, manager):
        _one_in_flight_out(manager, pool, tier_backend)
        tier_backend.never_confirm = True
        report = destroy_tier_manager(manager, wait_for_drain=True, timeout_s=0.05)
        assert report.timed_out is True and report.in_flight
        assert manager.destroyed is False  # 超时不销毁，排空后可重试（T-7 可恢复态）
        assert manager.query_tier_capability().available
        tier_backend.never_confirm = False
        final = destroy_tier_manager(manager, wait_for_drain=True, timeout_s=0.5)
        assert final.timed_out is False and manager.destroyed

    def test_销毁_不等在途_报错附清单(self, pool, tier_backend, manager):
        _, _, entry_id = _one_in_flight_out(manager, pool, tier_backend)
        with pytest.raises(TierError, match="不静默丢弃") as excinfo:
            destroy_tier_manager(manager, wait_for_drain=False)
        assert entry_id in str(excinfo.value)
        assert not manager.destroyed

    def test_级二容量按承诺量判读_在途换出计入(self, pool, tier_backend, manager, monkeypatch):
        monkeypatch.setattr(manager, "_level2_capacity", BLOCK_BYTES)
        blocks = _full_pool_blocks(pool, tier_backend)
        first = manager.evict_async(
            [blocks[0].block_id], pressure=PressureLevel.HIGH
        )
        assert first.scheduled_entries  # 承诺量 512 KiB，装得下一笔
        # 已确认 0 + 在途 512 KiB = 承诺量满 ⇒ 第二笔拒发，如实标注未达目标
        second = manager.evict_async(
            [blocks[1].block_id],
            target_bytes=BLOCK_BYTES,
            pressure=PressureLevel.HIGH,
        )
        assert second.scheduled_entries == () and not second.met_target
        assert "级二容量不足" in second.reason
        assert manager.query_tier_audit().unmet_target_evicts == 1

    def test_预算闸_异步发起即记账(self, pool, device_backend, tier_backend):
        policy = TierPolicy(**{**DEFAULT_POLICY.__dict__, "budget_bytes": BLOCK_BYTES})
        pool = create_pool(
            device_backend,
            form=PoolForm.HANDLE,
            workload=WorkloadType.GENERATIVE,
            tier_spec=TierSpec(base_bytes=BLOCK_BYTES, step_bytes=4096, tier_count=6),
        )
        manager = create_tier_manager(
            pool, tier_backend, level2_capacity_bytes=LEVEL2_CAP, policy=policy
        )
        blocks = _full_pool_blocks(pool, tier_backend)
        first = manager.evict_async(
            [blocks[0].block_id], pressure=PressureLevel.HIGH
        )
        assert first.scheduled_entries
        second = manager.evict_async(
            [blocks[1].block_id], pressure=PressureLevel.HIGH
        )
        assert second.scheduled_entries == () and "预算闸" in second.reason
        assert manager.query_tier_audit().budget_gate_deferred == 1

    def test_最小驻留_异步换出同样生效_E2可越窗计审计(
        self, pool, tier_backend
    ):
        policy = TierPolicy(
            **{**DEFAULT_POLICY.__dict__, "min_residency_s": 3600.0}
        )
        manager = create_tier_manager(
            pool, tier_backend, level2_capacity_bytes=LEVEL2_CAP, policy=policy
        )
        blocks = _full_pool_blocks(pool, tier_backend)
        manager.evict([blocks[0].block_id], pressure=PressureLevel.HIGH)
        outcome = manager.promote(f"b{blocks[0].block_id}")
        assert isinstance(outcome, PromoteResult)
        # E2 压力路同样受最小驻留约束：刚换入的块不作候选
        deferred = manager.evict_async(
            [outcome.block_id], pressure=PressureLevel.HIGH
        )
        assert deferred.scheduled_entries == ()
        assert manager.query_tier_audit().min_residency_deferred == 1
        # E2 + 显式允许越窗 ⇒ 可发起，越窗计入回执与审计
        bypassed = manager.evict_async(
            [outcome.block_id],
            pressure=PressureLevel.HIGH,
            allow_bypass_min_residency=True,
        )
        assert bypassed.scheduled_entries and bypassed.bypassed_residency == 1
        manager.poll_transfers()
        assert manager.query_tier_audit().bypass_window_evicts == 1

    def test_级三处置_换入在途的条目不丢(self, pool, tier_backend, manager):
        blocks = _full_pool_blocks(pool, tier_backend)
        for block in blocks[:2]:
            manager.evict([block.block_id], pressure=PressureLevel.HIGH)
        manager.set_tier_policy(
            TierPolicy(**{**DEFAULT_POLICY.__dict__, "level2_high_watermark": 0.05})
        )
        in_flight_entry = f"b{blocks[0].block_id}"
        assert isinstance(
            manager.promote_async(in_flight_entry, keep_tier2_copy=True),
            AsyncPromoteReceipt,
        )
        report = manager.apply_r13_disposition()
        # 有换入在途的条目不丢（级二副本是 H2D 的数据源）；只丢无在途的那笔
        assert report.dropped_entries == (f"b{blocks[1].block_id}",)
        assert "换入在途" in report.reason
        assert in_flight_entry not in tier_backend.dropped


# ======================================================================
# 异常
# ======================================================================


class TestAsyncError:
    def test_异步换出_发起失败_不落账本块留在用_此前在途保留(
        self, pool, tier_backend, manager, monkeypatch
    ):
        blocks = _full_pool_blocks(pool, tier_backend)
        original = tier_backend.copy_out
        calls: list[tuple] = []

        def flaky_copy_out(*args, **kwargs):
            calls.append(args)
            if len(calls) == 2:
                raise RuntimeError("注入：第二笔发起失败")
            return original(*args, **kwargs)

        monkeypatch.setattr(tier_backend, "copy_out", flaky_copy_out)
        with pytest.raises(TierCopyFailed, match="发起失败"):
            manager.evict_async(
                [b.block_id for b in blocks[:2]], pressure=PressureLevel.HIGH
            )
        # 第一笔已发起 ⇒ 仍在途；第二笔失败 ⇒ 不落账本、不还块、块留在用
        tickets = manager.query_transfers()
        assert len(tickets) == 1
        assert tickets[0].entry_id == f"b{blocks[0].block_id}"
        assert pool.block_view(blocks[0].block_id).state is BlockState.IN_USE
        assert pool.block_view(blocks[1].block_id).state is BlockState.IN_USE
        assert manager.query_tier_watermark().level2.entry_count == 0
        assert manager.query_tier_audit().copy_failures == 1

    def test_异步换出_确认期失败_回收路径_可重新调度(
        self, pool, tier_backend, manager
    ):
        _, block_id, entry_id = _one_in_flight_out(manager, pool, tier_backend)
        tier_backend.fail_on_confirm = True
        report = manager.poll_transfers()
        failed = report.failed
        assert len(failed) == 1 and failed[0].state is TransferState.FAILED
        assert entry_id in (failed[0].error or "")
        # 3.5 回收：条目不落账本、不调 free_block，块留在用
        assert manager.query_tier_watermark().level2.entry_count == 0
        assert pool.block_view(block_id).state is BlockState.IN_USE
        assert manager.query_tier_audit().copy_failures == 1
        assert manager.query_transfers() == ()  # 出表 ⇒ 可重新调度
        tier_backend.fail_on_confirm = False
        retry = manager.evict_async([block_id], pressure=PressureLevel.HIGH)
        assert retry.scheduled_entries == (entry_id,)
        assert manager.poll_transfers().confirmed[0].entry_id == entry_id

    def test_异步换入_发起失败_归还预留块_条目可重试(
        self, pool, tier_backend, manager, monkeypatch
    ):
        _, _, entry_id = _one_in_flight_out(manager, pool, tier_backend)
        manager.poll_transfers()
        original_free = pool.free_block
        free_calls: list[int] = []

        def counting_free(bid: int):
            free_calls.append(bid)
            return original_free(bid)

        monkeypatch.setattr(pool, "free_block", counting_free)
        tier_backend.fail_copy_in = True
        with pytest.raises(TierCopyFailed, match="归还预留块"):
            manager.promote_async(entry_id)
        # 预留块恰好归还一次（在用量与待回收来回抵消，用归还次数钉），级二条目保留可重试（3.5）
        assert len(free_calls) == 1
        assert manager.query_tier_watermark().level2.entry_count == 1
        assert manager.query_tier_audit().copy_failures == 1
        tier_backend.fail_copy_in = False
        assert isinstance(manager.promote_async(entry_id), AsyncPromoteReceipt)

    def test_异步换入_确认期失败_归还块_条目保留(
        self, pool, tier_backend, manager
    ):
        _, _, entry_id = _one_in_flight_out(manager, pool, tier_backend)
        manager.poll_transfers()
        assert isinstance(manager.promote_async(entry_id), AsyncPromoteReceipt)
        in_use_before = manager.query_tier_watermark().level1.pool_in_use
        tier_backend.fail_on_confirm = True
        report = manager.poll_transfers()
        failed = report.failed
        assert len(failed) == 1 and failed[0].direction is TransferDirection.PROMOTE
        wm = manager.query_tier_watermark()
        assert wm.level1.pool_in_use < in_use_before  # 预留块归还
        assert wm.level2.entry_count == 1  # 条目保留可重试
        assert manager.query_tier_audit().copy_failures == 1
        tier_backend.fail_on_confirm = False
        assert isinstance(manager.promote(entry_id).block_id, int)  # 同步重试成功

    def test_发起无完成凭据_报错_不挂起(self, pool, tier_backend, manager, monkeypatch):
        blocks = _full_pool_blocks(pool, tier_backend)
        monkeypatch.setattr(tier_backend, "copy_out", lambda *a, **k: None)
        with pytest.raises(TierCopyFailed, match="未返回完成凭据"):
            manager.evict_async([blocks[0].block_id], pressure=PressureLevel.HIGH)
        assert manager.query_transfers() == ()  # 没有挂起的在途记录
        assert pool.block_view(blocks[0].block_id).state is BlockState.IN_USE

    def test_确认只结算一次_不重复还块(self, pool, tier_backend, manager, monkeypatch):
        _, block_id, _ = _one_in_flight_out(manager, pool, tier_backend)
        original_free = pool.free_block
        free_calls: list[int] = []
        monkeypatch.setattr(
            pool, "free_block", lambda bid: (free_calls.append(bid), original_free(bid))[1]
        )
        manager.poll_transfers()
        manager.poll_transfers()  # 第二轮已无在途，什么都不该发生
        assert free_calls == [block_id]
        assert manager.query_tier_audit().evicted_entries == 1
