"""分配 / 归还 / 复用与六态状态机。全部不依赖真卡。

判据来源：`docs/design/显存池概念与设计基础.md` 第二节 2.2（六态与迁移表）与第三节
3.1 / 3.2（分配与复用的前置后置与失败行为）。
"""

from __future__ import annotations

import pytest

from flagmemory.caps.capabilities import L0_CAPABILITIES, L1_CAPABILITIES
from flagmemory.caps.fake import FakeStream
from flagmemory.pool import (
    BlockState,
    CapacityExhausted,
    FormUnavailable,
    InvalidArgument,
    OwnershipViolation,
    PoolError,
    PoolForm,
    Purpose,
    ReusePolicy,
    TierSpec,
    WorkloadType,
    create_pool,
)
from fakes import CompletingBackend, FakeBaseAllocator


class TestNormal:
    """正常：定档、绑流绑标签、归还只到待回收、确认后才可复用、复用命中。"""

    def test_分配后状态是在用且定了档(self, handle_pool):
        view = handle_pool.alloc_on_stream(300 << 10, purpose=Purpose.ACTIVATION)
        assert view.state is BlockState.IN_USE
        assert view.tier_bytes == 512 << 10      # 定档到第一个 ≥ 请求的档
        assert view.nbytes == 300 << 10          # 实际字节数照记
        assert view.purpose is Purpose.ACTIVATION
        assert view.owner_stream == 0            # 不给流就绑默认流

    def test_归还只到待回收不直接可复用(self, handle_pool):
        """调用方看到的「还回去了」只到待回收；流序未确认时池不把块交给下一个人。"""
        view = handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        after = handle_pool.free_block(view.block_id)
        assert after.state is BlockState.PENDING_RECYCLE

    def test_屏障确认后转可复用(self, handle_pool):
        view = handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        handle_pool.free_block(view.block_id)
        result = handle_pool.drain_stream()
        assert result.promoted_blocks == 1
        assert handle_pool.block_view(view.block_id).state is BlockState.REUSABLE

    def test_复用命中同一块且改绑流与标签(self, handle_pool):
        first = handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        handle_pool.free_block(first.block_id)
        handle_pool.drain_stream()
        other = handle_pool.register_stream(FakeStream(stream_id=7))

        second = handle_pool.alloc_on_stream(
            900 << 10, stream_id=other, purpose=Purpose.KV_CACHE
        )
        assert second.block_id == first.block_id        # 命中复用，没向底座再拿
        assert second.owner_stream == other             # 改绑本次请求的流
        assert second.purpose is Purpose.KV_CACHE       # 改写标签
        assert second.borrowers == frozenset()          # 跨流借用集清空
        audit = handle_pool.query_audit()
        assert audit.reuse_hits == 1
        assert audit.base_fetches == 1

    def test_张量级形态走注入的底座(self, tensor_pool, base_allocator: FakeBaseAllocator):
        view = tensor_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        assert base_allocator.used_bytes == view.tier_bytes
        assert len(base_allocator.live) == 1

    def test_不给标签时按未声明处理(self, handle_pool):
        view = handle_pool.alloc_on_stream(4096)
        assert view.purpose is Purpose.UNDECLARED


class TestBoundary:
    """边界：最小/最大档、档位不匹配不复用、释放阈值、机会性复用、块标识不复用。"""

    def test_不同档的空闲块不会被跨档复用(self, handle_pool):
        small = handle_pool.alloc_on_stream(4096, purpose=Purpose.ACTIVATION)
        handle_pool.free_block(small.block_id)
        handle_pool.drain_stream()
        big = handle_pool.alloc_on_stream(2 << 20, purpose=Purpose.ACTIVATION)
        assert big.block_id != small.block_id
        assert big.tier_bytes == 2 << 20

    def test_块标识归还给底座后不再复用(self, handle_pool):
        view = handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        handle_pool.free_block(view.block_id)
        handle_pool.drain_stream()
        handle_pool.compact()                       # 交回底座，标识退役
        again = handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        assert again.block_id != view.block_id      # 审计对账时标识永不含糊

    def test_释放阈值为零时归还直接交回底座(self, handle_backend_done, tier_spec: TierSpec):
        pool = create_pool(
            handle_backend_done,
            form=PoolForm.HANDLE,
            workload=WorkloadType.GENERATIVE,
            tier_spec=tier_spec,
            reuse=ReusePolicy(release_threshold_bytes=0),
        )
        view = pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        pool.free_block(view.block_id)
        pool.drain_stream()
        frag = pool.query_fragmentation()
        assert frag.watermark.pool_reusable == 0     # 没留在空闲表
        assert all(t.reusable_blocks == 0 for t in frag.tiers)

    def test_关掉机会性复用后分配不替你推进待回收(self, handle_backend_done, tier_spec: TierSpec):
        pool = create_pool(
            handle_backend_done,
            form=PoolForm.HANDLE,
            workload=WorkloadType.GENERATIVE,
            tier_spec=tier_spec,
            reuse=ReusePolicy(opportunistic=False),
        )
        first = pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        pool.free_block(first.block_id)             # 停在待回收，没 drain
        second = pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        assert second.block_id != first.block_id    # 没复用，去底座拿了新的
        assert pool.query_audit().reuse_hits == 0

    def test_开着机会性复用则分配时顺手推进一次(self, handle_pool):
        first = handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        handle_pool.free_block(first.block_id)      # 停在待回收，不手动 drain
        second = handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        assert second.block_id == first.block_id    # 机会性复用命中
        assert handle_pool.query_audit().reuse_hits == 1


class TestError:
    """异常：容量不足、参数非法、重复归还、非本池块、形态不可用 —— 一律报错不静默。"""

    def test_容量不足时报错且带双指标水位(self, l0_backend_done, tier_spec: TierSpec):
        pool = create_pool(
            l0_backend_done,
            form=PoolForm.TENSOR,
            workload=WorkloadType.GENERATIVE,
            base_allocator=FakeBaseAllocator(capacity_bytes=1 << 20),
            tier_spec=tier_spec,
        )
        pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        with pytest.raises(CapacityExhausted) as exc:
            pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        err = exc.value
        assert err.watermark.device_total > 0            # 设备侧
        assert err.watermark.pool_in_use == 1 << 20      # 引擎侧
        assert err.tier_free.tier_bytes == 1 << 20
        assert "不自动缩小请求" in str(err)

    @pytest.mark.parametrize("nbytes", [0, -1])
    def test_字节数非法时报错(self, handle_pool, nbytes: int):
        with pytest.raises(InvalidArgument):
            handle_pool.alloc_on_stream(nbytes)

    def test_标签不在枚举内时报错(self, handle_pool):
        with pytest.raises(InvalidArgument) as exc:
            handle_pool.alloc_on_stream(4096, purpose="权重")  # type: ignore[arg-type]
        assert "Purpose" in str(exc.value)

    def test_流未登记时报错(self, handle_pool):
        with pytest.raises(InvalidArgument) as exc:
            handle_pool.alloc_on_stream(4096, stream_id=99)
        assert "未登记" in str(exc.value)

    def test_重复归还被池账本拦下并转隔离(self, handle_pool):
        """`I-1`：不等厂商报 —— 昇腾的裸释放对二次释放静默返回成功。"""
        view = handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        handle_pool.free_block(view.block_id)
        with pytest.raises(OwnershipViolation) as exc:
            handle_pool.free_block(view.block_id)
        assert "重复归还" in str(exc.value)
        assert handle_pool.block_view(view.block_id).state is BlockState.QUARANTINED

    def test_非本池块报错(self, handle_pool):
        with pytest.raises(OwnershipViolation) as exc:
            handle_pool.free_block(123456)
        assert "不是本池发出的" in str(exc.value)

    def test_归还已交回底座的块报重复归还(self, handle_pool):
        view = handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        handle_pool.free_block(view.block_id)
        handle_pool.drain_stream()
        handle_pool.compact()
        with pytest.raises(OwnershipViolation) as exc:
            handle_pool.free_block(view.block_id)
        assert "已归还给底座" in str(exc.value)

    def test_隔离块不再被复用也不被释放(self, handle_pool):
        view = handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        handle_pool.free_block(view.block_id)
        with pytest.raises(OwnershipViolation):
            handle_pool.free_block(view.block_id)          # 转隔离
        reused = handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        assert reused.block_id != view.block_id            # 隔离块不进复用
        assert handle_pool.query_audit().quarantined_blocks == 1

    def test_强制句柄形态而后端未声明句柄能力时报错(self, l0_backend_done, tier_spec: TierSpec):
        with pytest.raises(FormUnavailable) as exc:
            create_pool(l0_backend_done, form=PoolForm.HANDLE, tier_spec=tier_spec)
        assert "不静默退化" in str(exc.value)

    def test_强制张量级而没注入底座时报错(self, l0_backend_done, tier_spec: TierSpec):
        with pytest.raises(InvalidArgument) as exc:
            create_pool(l0_backend_done, form=PoolForm.TENSOR, tier_spec=tier_spec)
        assert "base_allocator" in str(exc.value)

    def test_自动形态两条路都不通时报错(self, l0_backend_done, tier_spec: TierSpec):
        with pytest.raises(InvalidArgument) as exc:
            create_pool(l0_backend_done, form=PoolForm.AUTO, tier_spec=tier_spec)
        assert "定不下来" in str(exc.value)

    def test_注入的底座不满足协议时报错(self, l0_backend_done, tier_spec: TierSpec):
        class NotAnAllocator:
            pass

        with pytest.raises(InvalidArgument) as exc:
            create_pool(
                l0_backend_done,
                form=PoolForm.TENSOR,
                base_allocator=NotAnAllocator(),  # type: ignore[arg-type]
                tier_spec=tier_spec,
            )
        assert "BaseAllocator" in str(exc.value)

    def test_缺L0核心能力时池不起(self, tier_spec: TierSpec):
        """`I-8`：核心路径只许用四家齐备的 L0；缺了是「不支持」而不是降级。"""
        crippled = CompletingBackend(
            name="missing-l0",
            declared=frozenset((L0_CAPABILITIES | L1_CAPABILITIES) - {
                next(iter(L0_CAPABILITIES))
            }),
        )
        with pytest.raises(PoolError) as exc:
            create_pool(crippled, form=PoolForm.HANDLE, tier_spec=tier_spec)
        assert "L0" in str(exc.value)

    def test_池销毁后不接受任何操作(self, handle_pool):
        handle_pool.destroy_pool()
        with pytest.raises(PoolError):
            handle_pool.alloc_on_stream(4096)
