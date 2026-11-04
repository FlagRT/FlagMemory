"""回收屏障与跨流保护。全部不依赖真卡。

判据来源：`docs/design/显存池概念与设计基础.md` 第二节 2.4（跨流保护两条路径）、
第三节 3.3（回收与屏障约束）、不变量 `I-2` / `I-11`。
"""

from __future__ import annotations

import pytest

from flagmemory.caps.capabilities import Capability, L0_CAPABILITIES, L1_CAPABILITIES
from flagmemory.caps.fake import FakeStream
from flagmemory.pool import (
    BlockState,
    InvalidArgument,
    OwnershipViolation,
    PoolError,
    PoolForm,
    Purpose,
    TierSpec,
    create_pool,
)
from fakes import CompletingBackend


class _RecordingCrossStream(CompletingBackend):
    """声明并实现了跨流内存保护登记口的后端（= 三家那一档）。"""

    @property
    def recorded(self) -> list[tuple[int, int | None]]:
        log = getattr(self, "_recorded", None)
        if log is None:
            log = []
            setattr(self, "_recorded", log)
        return log

    def record_stream(self, base_ref, stream) -> None:
        self.recorded.append((base_ref.handle_id, getattr(stream, "stream_id", None)))


class TestNormal:
    """正常：屏障等所属流 + 跨流借用集；两条跨流路径对外语义相同。"""

    def test_跨流声明把流并入借用集(self, handle_pool):
        view = handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        other = handle_pool.register_stream(FakeStream(stream_id=3))
        borrowers = handle_pool.declare_cross_stream_use(view.block_id, other)
        assert borrowers == frozenset({other})
        assert handle_pool.block_view(view.block_id).borrowers == frozenset({other})

    def test_有跨流保护能力时经统一层登记(self, tier_spec: TierSpec):
        backend = _RecordingCrossStream(
            name="with-protection", declared=frozenset(L0_CAPABILITIES | L1_CAPABILITIES)
        )
        pool = create_pool(backend, form=PoolForm.HANDLE, tier_spec=tier_spec)
        view = pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        other = pool.register_stream(FakeStream(stream_id=5))
        pool.declare_cross_stream_use(view.block_id, other)
        assert backend.recorded == [(1, 5)]

    def test_无跨流保护能力时只记池内借用集并在降级清单里说明(self, tensor_pool):
        """寒武纪那一档：对外语义相同，差别只在池内部的屏障成本。"""
        view = tensor_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        other = tensor_pool.register_stream(FakeStream(stream_id=4))
        assert tensor_pool.declare_cross_stream_use(view.block_id, other) == frozenset({other})
        keys = {d.key for d in tensor_pool.query_capability().degradations}
        assert "tensor_level_pool" in keys

    def test_屏障要等借用集里每条流各一个事件(self, handle_pool):
        view = handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        for sid in (11, 12):
            handle_pool.declare_cross_stream_use(
                view.block_id, handle_pool.register_stream(FakeStream(stream_id=sid))
            )
        pending = handle_pool.free_block(view.block_id)
        # 所属流 + 两条借用流 = 三条相关流，各取一个事件凭据，全部确认才放行
        assert len(pending.borrowers) == 2
        result = handle_pool.drain_stream()
        assert result.promoted_blocks == 1
        assert handle_pool._ledger._blocks[view.storage_id].state is BlockState.REUSABLE

    def test_按流推进只动相关流上的块(self, handle_pool):
        stream_a = handle_pool.register_stream(FakeStream(stream_id=21))
        stream_b = handle_pool.register_stream(FakeStream(stream_id=22))
        block_a = handle_pool.alloc_on_stream(1 << 20, stream_id=stream_a, purpose=Purpose.ACTIVATION)
        block_b = handle_pool.alloc_on_stream(2 << 20, stream_id=stream_b, purpose=Purpose.ACTIVATION)
        handle_pool.free_block(block_a.block_id)
        handle_pool.free_block(block_b.block_id)

        result = handle_pool.drain_stream(stream_id=stream_a)
        assert result.promoted_blocks == 1
        assert handle_pool._ledger._blocks[block_a.storage_id].state is BlockState.REUSABLE
        assert handle_pool._ledger._blocks[block_b.storage_id].state is BlockState.PENDING_RECYCLE


class TestBoundary:
    """边界：超时是可恢复态；超阈才转隔离；没有待回收时推进是空操作。"""

    def test_屏障超时不等于失败(self, pending_pool):
        """`I-11`：超时按可恢复态上报，块留在待回收，不抛异常。"""
        view = pending_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        pending_pool.free_block(view.block_id)
        result = pending_pool.drain_stream(timeout_s=0.001)
        assert result.recoverable_timeout is True
        assert result.promoted_blocks == 0
        assert result.still_pending == 1
        assert view.block_id in result.timed_out_blocks
        assert pending_pool._ledger._blocks[view.storage_id].state is BlockState.PENDING_RECYCLE

    def test_零等待观察不隔离健康工作(self, pending_pool):
        view = pending_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        pending_pool.free_block(view.block_id)
        for _ in range(5):
            result = pending_pool.drain_stream(timeout_s=0.0)
            assert result.quarantined_blocks == ()
            assert not result.recoverable_timeout
        assert pending_pool.query_audit().pending_depth == 1

    def test_没有待回收时推进是空操作(self, handle_pool):
        result = handle_pool.drain_stream()
        assert (result.promoted_blocks, result.still_pending) == (0, 0)
        assert result.recoverable_timeout is False

    def test_同一条流上多块一起放行(self, handle_pool):
        views = [
            handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION) for _ in range(3)
        ]
        for v in views:
            handle_pool.free_block(v.block_id)
        result = handle_pool.drain_stream()
        assert result.promoted_blocks == 3
        assert result.promoted_bytes == 3 * (1 << 20)


class TestError:
    """异常：事件适配器报错 → 隔离；取不到凭据 → 不给空凭据；流未登记 → 报错。"""

    def test_事件适配器报错时转隔离(self, tier_spec: TierSpec):
        class FaultyQuery(CompletingBackend):
            def query_event(self, event):
                raise RuntimeError("事件查询失败")

        backend = FaultyQuery(
            name="faulty-query", declared=frozenset(L0_CAPABILITIES | L1_CAPABILITIES)
        )
        pool = create_pool(backend, form=PoolForm.HANDLE, tier_spec=tier_spec)
        view = pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        pool.free_block(view.block_id)
        result = pool.drain_stream()
        assert view.block_id in result.quarantined_blocks
        assert pool.query_audit().quarantine_reasons == {"事件适配器报错": 1}

    def test_取不到屏障凭据时归还失败并转隔离(self, tier_spec: TierSpec):
        """宁可归还失败并上报，也不给一张空凭据 —— 空凭据会让在用显存立刻变可复用。"""

        class NoEvent(CompletingBackend):
            def create_event(self, device_index: int = 0):
                raise RuntimeError("取不到事件")

        backend = NoEvent(name="no-event", declared=frozenset(L0_CAPABILITIES | L1_CAPABILITIES))
        pool = create_pool(backend, form=PoolForm.HANDLE, tier_spec=tier_spec)
        view = pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        with pytest.raises(PoolError) as exc:
            pool.free_block(view.block_id)
        assert "空凭据" in str(exc.value)
        assert pool._ledger._blocks[view.storage_id].state is BlockState.QUARANTINED

    def test_跨流声明的块必须在用(self, handle_pool):
        view = handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        handle_pool.free_block(view.block_id)
        other = handle_pool.register_stream(FakeStream(stream_id=9))
        with pytest.raises(OwnershipViolation):
            handle_pool.declare_cross_stream_use(view.block_id, other)

    def test_跨流声明的流必须已登记(self, handle_pool):
        view = handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        with pytest.raises(InvalidArgument):
            handle_pool.declare_cross_stream_use(view.block_id, 404)

    def test_推进未登记的流报错(self, handle_pool):
        with pytest.raises(InvalidArgument):
            handle_pool.drain_stream(stream_id=404)

    def test_登记空流报错(self, handle_pool):
        with pytest.raises(InvalidArgument):
            handle_pool.register_stream(None)

    def test_跨流保护能力声明了但后端没实现那个口时不报错只记池内(self, tier_spec: TierSpec):
        """声明与实现不一致是后端的问题，池这边按「没有那个口」走，不崩。"""
        backend = CompletingBackend(
            name="declared-not-implemented",
            declared=frozenset(L0_CAPABILITIES | L1_CAPABILITIES),
        )
        assert Capability.CROSS_STREAM_PROTECTION in backend.declared_capabilities()
        pool = create_pool(backend, form=PoolForm.HANDLE, tier_spec=tier_spec)
        view = pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        other = pool.register_stream(FakeStream(stream_id=6))
        assert pool.declare_cross_stream_use(view.block_id, other) == frozenset({other})
