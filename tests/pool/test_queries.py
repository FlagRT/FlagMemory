"""水位 / 碎片 / 整理 / 压力回收 / 预算 / 能力 / 审计 / 销毁。全部不依赖真卡。

判据来源：`docs/design/显存池概念与设计基础.md` 第三节 3.3 / 3.4 与第四节 4.2 的契约表，
以及不变量 `I-5` / `I-6` / `I-7` / `I-10` / `I-12`。
"""

from __future__ import annotations

import pytest

from flagmemory.caps.backend import CapabilityNotDeclared
from flagmemory.caps.capabilities import Capability, L0_CAPABILITIES, L1_CAPABILITIES
from flagmemory.pool import (
    BlockState,
    InvalidArgument,
    LeakDetected,
    NotWired,
    PoolError,
    PoolForm,
    PressureLevel,
    Purpose,
    TierSpec,
    WorkloadType,
    create_pool,
)
from fakes import CompletingBackend, FakeBaseAllocator


class TestNormal:
    """正常：双指标齐全、整理交回底座、回收按标签次序、预算按负载判读。"""

    def test_水位双指标两侧都给(self, handle_pool):
        """`I-6`：设备侧 + 引擎侧缺一不可。"""
        handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        wm = handle_pool.query_watermark()
        assert wm.device_total > 0 and wm.device_free >= 0        # 设备侧三键
        assert wm.pool_in_use == 1 << 20                           # 引擎侧
        assert wm.pool_total == wm.pool_in_use

    def test_碎片给各档空闲分布与可立刻满足的最大单块(self, handle_pool):
        small = handle_pool.alloc_on_stream(4096, purpose=Purpose.ACTIVATION)
        big = handle_pool.alloc_on_stream(2 << 20, purpose=Purpose.ACTIVATION)
        for v in (small, big):
            handle_pool.free_block(v.block_id)
        handle_pool.drain_stream()
        frag = handle_pool.query_fragmentation()
        by_tier = {t.tier_bytes: t.reusable_blocks for t in frag.tiers}
        assert by_tier[512 << 10] == 1
        assert by_tier[2 << 20] == 1
        assert frag.largest_servable_bytes == 2 << 20
        assert 0.0 < frag.engine_fragmentation <= 1.0

    def test_整理把可复用块交回底座(self, tensor_pool, base_allocator: FakeBaseAllocator):
        view = tensor_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        tensor_pool.free_block(view.block_id)
        tensor_pool.drain_stream()
        result = tensor_pool.compact()
        assert result.improved is True
        assert result.returned_blocks == 1
        assert result.returned_bytes == 1 << 20
        assert base_allocator.used_bytes == 0
        assert result.after.watermark.pool_reusable == 0

    def test_压力回收按用途次序走(self, handle_pool):
        """临时工作区最优先，然后激活，然后 KV 缓存。"""
        # 先全部分配再全部归还：交错着做会命中机会性复用，三个标签会落在同一块上。
        blocks = {
            purpose: handle_pool.alloc_on_stream(1 << 20, purpose=purpose).block_id
            for purpose in (Purpose.KV_CACHE, Purpose.ACTIVATION, Purpose.SCRATCH)
        }
        for block_id in blocks.values():
            handle_pool.free_block(block_id)
        handle_pool.drain_stream()

        result = handle_pool.reclaim(target_bytes=1 << 20)
        assert result.met_target is True
        assert list(result.released_by_purpose) == [Purpose.SCRATCH]
        assert handle_pool._ledger._blocks[blocks[Purpose.ACTIVATION]].state is BlockState.REUSABLE

    def test_按压力等级回收(self, handle_pool):
        views = [
            handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION) for _ in range(4)
        ]
        for v in views:
            handle_pool.free_block(v.block_id)
        handle_pool.drain_stream()
        result = handle_pool.reclaim(pressure=PressureLevel.HIGH)
        assert result.released_bytes == 4 << 20
        assert result.met_target is True

    def test_预算按负载类型给不同判读(self, handle_pool, tensor_pool):
        """`I-7`：生成式的池是并发容量，向量化的池是纯余量。"""
        assert handle_pool.workload is WorkloadType.GENERATIVE
        assert "并发容量" in handle_pool.query_budget().compressible_reading
        assert tensor_pool.workload is WorkloadType.EMBEDDING
        assert "纯余量" in tensor_pool.query_budget().compressible_reading

    def test_预算按标签聚合已分配量(self, handle_pool):
        handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.WEIGHT)
        handle_pool.alloc_on_stream(2 << 20, purpose=Purpose.KV_CACHE)
        budget = handle_pool.query_budget()
        assert budget.allocated_by_purpose[Purpose.WEIGHT] == 1 << 20
        assert budget.allocated_by_purpose[Purpose.KV_CACHE] == 2 << 20
        assert budget.allocated_by_purpose[Purpose.ACTIVATION] == 0
        assert "池自统计" in budget.source

    def test_能力视图给形态分支与降级清单(self, handle_pool):
        view = handle_pool.query_capability()
        assert view.form is PoolForm.HANDLE
        assert view.native_stream_ordered is False      # 昆仑芯那一档没有原生流有序
        assert view.core_path_ready is True
        assert Capability.MEMORY_HANDLE in view.branches_enabled
        assert view.tier_source == "caller_config"
        assert view.degradations                        # 降级必须可观测（I-10）

    def test_审计视图给复用命中率(self, handle_pool):
        v = handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        handle_pool.free_block(v.block_id)
        handle_pool.drain_stream()
        handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        audit = handle_pool.query_audit()
        assert (audit.alloc_requests, audit.reuse_hits, audit.base_fetches) == (2, 1, 1)
        assert audit.reuse_hit_rate == 0.5

    def test_干净销毁交回全部块并放开流强引用(self, tensor_pool, base_allocator: FakeBaseAllocator):
        v = tensor_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        tensor_pool.free_block(v.block_id)
        report = tensor_pool.destroy_pool()
        assert report.returned_blocks == 1
        assert report.leaked == ()
        assert base_allocator.used_bytes == 0
        assert tensor_pool.destroyed is True


class TestBoundary:
    """边界：无改善如实说、未达目标如实说、空池的各视图、不允许交回底座。"""

    def test_没有可交回的块时如实报无改善(self, handle_pool):
        result = handle_pool.compact()
        assert result.improved is False
        assert result.returned_blocks == 0
        assert "无改善" in result.note

    def test_不允许交回底座时说明本层做不了合并(self, handle_pool):
        """`I-9`：池不持有地址空间，合并做不到 —— 这是现状而非待优化项。"""
        v = handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        handle_pool.free_block(v.block_id)
        handle_pool.drain_stream()
        result = handle_pool.compact(allow_return_to_base=False)
        assert result.improved is False
        assert "合并" in result.note and "地址空间" in result.note

    def test_回收达不到目标量时如实返回实际量(self, handle_pool):
        v = handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        handle_pool.free_block(v.block_id)
        handle_pool.drain_stream()
        result = handle_pool.reclaim(target_bytes=100 << 20)
        assert result.met_target is False
        assert result.released_bytes == 1 << 20
        assert "未达目标" in result.note

    def test_只整理指定档(self, handle_pool):
        small = handle_pool.alloc_on_stream(4096, purpose=Purpose.ACTIVATION)
        big = handle_pool.alloc_on_stream(2 << 20, purpose=Purpose.ACTIVATION)
        for v in (small, big):
            handle_pool.free_block(v.block_id)
        handle_pool.drain_stream()
        result = handle_pool.compact(tier_bytes=512 << 10)
        assert result.returned_blocks == 1
        assert handle_pool._ledger._blocks[big.storage_id].state is BlockState.REUSABLE

    def test_空池的视图都能取(self, handle_pool):
        assert handle_pool.query_audit().live_blocks == 0
        assert handle_pool.query_fragmentation().largest_servable_bytes == 0
        assert handle_pool.query_budget().compressible_bytes == 0
        assert handle_pool.query_audit().reuse_hit_rate == 0.0

    def test_在用块不参与整理也不参与回收(self, handle_pool):
        """`在用` 与 `待回收` 一概不动（设计文档 3.4 前置条件）。"""
        in_use = handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        pending = handle_pool.alloc_on_stream(2 << 20, purpose=Purpose.ACTIVATION)
        handle_pool.free_block(pending.block_id)        # 停在待回收
        result = handle_pool.compact()
        assert result.returned_blocks == 0
        assert handle_pool._ledger._blocks[in_use.storage_id].state is BlockState.IN_USE
        assert handle_pool._ledger._blocks[pending.storage_id].state is BlockState.PENDING_RECYCLE

    def test_销毁时排空超时按可恢复态上报(self, pending_pool):
        v = pending_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        pending_pool.free_block(v.block_id)
        report = pending_pool.destroy_pool(timeout_s=0.001)
        assert report.drain_timed_out is True
        assert report.leaked == ()            # 待回收不算泄漏，在用才算

    def test_按标签筛选预算视图(self, handle_pool):
        handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.WEIGHT)
        budget = handle_pool.query_budget(purposes=[Purpose.WEIGHT])
        assert list(budget.allocated_by_purpose) == [Purpose.WEIGHT]


class TestError:
    """异常：设备侧查不到即报错、分层换出没接线、回收入参互斥、泄漏不静默释放。"""

    def test_设备侧三键查不到时报错不用引擎侧充抵(self, tier_spec: TierSpec):
        """`I-6`：双指标缺一不可，不用单边数充抵。"""

        class NoDeviceInfo(CompletingBackend):
            def device_memory_info(self, device_index: int = 0):
                raise CapabilityNotDeclared(Capability.DEVICE_MEMORY_INFO, self.name)

        backend = NoDeviceInfo(
            name="no-devinfo", declared=frozenset(L0_CAPABILITIES | L1_CAPABILITIES)
        )
        pool = create_pool(backend, form=PoolForm.HANDLE, tier_spec=tier_spec)
        with pytest.raises(PoolError) as exc:
            pool.query_watermark()
        assert "双指标缺一不可" in str(exc.value)

    def test_分层换出没接线时报错而不是给空壳(self, handle_pool):
        with pytest.raises(NotWired) as exc:
            handle_pool.reclaim(target_bytes=1 << 20, allow_tier_offload=True)
        assert "没有接线" in str(exc.value)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"target_bytes": 1 << 20, "pressure": PressureLevel.LOW},
        ],
    )
    def test_回收的两种入参必须给且只给一个(self, handle_pool, kwargs):
        with pytest.raises(InvalidArgument):
            handle_pool.reclaim(**kwargs)

    @pytest.mark.parametrize("bad", [0, -1, True])
    def test_回收目标量必须是正整数(self, handle_pool, bad):
        with pytest.raises(InvalidArgument):
            handle_pool.reclaim(target_bytes=bad)

    def test_压力等级必须取自枚举(self, handle_pool):
        with pytest.raises(InvalidArgument):
            handle_pool.reclaim(pressure="高")  # type: ignore[arg-type]

    def test_整理不在档位表里的档报错(self, handle_pool):
        with pytest.raises(InvalidArgument):
            handle_pool.compact(tier_bytes=12345)

    def test_销毁时有在用块则抛泄漏并带清单(self, handle_pool):
        """池不静默释放在用块 —— 下面可能还在跑。"""
        v = handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        with pytest.raises(LeakDetected) as exc:
            handle_pool.destroy_pool()
        report = exc.value.report
        assert [b.block_id for b in report.leaked] == [v.block_id]
        assert "不静默释放在用块" in str(exc.value)

    def test_底座释放报错时块转隔离并上报(self, l0_backend_done, tier_spec: TierSpec):
        allocator = FakeBaseAllocator()
        pool = create_pool(
            l0_backend_done,
            form=PoolForm.TENSOR,
            base_allocator=allocator,
            tier_spec=tier_spec,
        )
        v = pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        pool.free_block(v.block_id)
        pool.drain_stream()
        allocator.fail_release = True
        with pytest.raises(PoolError) as exc:
            pool.compact()
        assert "转隔离" in str(exc.value)
        assert pool._ledger._blocks[v.storage_id].state is BlockState.QUARANTINED
        assert pool.query_audit().quarantine_reasons == {"底座释放报错": 1}

    def test_预算筛选的标签必须取自枚举(self, handle_pool):
        with pytest.raises(InvalidArgument):
            handle_pool.query_budget(purposes=["权重"])  # type: ignore[list-item]
