"""流有序分配的两条路：自管路（基线）与原生路（优化路径）。全部不依赖真卡。

判据：池的对外语义是「**按流分配 + 后端能力标记**」。标记为假走自管路（自己记流序、
空闲链复用、事件做回收屏障），为真则包原生原语。**两条路对上层暴露同一套接口** ——
所以本文件的大部分用例是「同一段操作在两条路上结果相同」。

四家的档位（只读符号探查结论，登记处 = 指挥台仓
`inbox/调研-四芯片SDK显存原语面补探-20261103.md`）：寒武纪有、平头哥有（能力挂在设备属性
门禁上、属性值未取证）、昇腾要较新的 CANN 大版本才有、昆仑芯没有（运行时层那套异步接口是
同步实现的空壳，不得当原生实现接进来）。
"""

from __future__ import annotations

import pytest

from flagmemory.caps.capabilities import L0_CAPABILITIES, L1_CAPABILITIES
from flagmemory.caps.fake import FakeStream
from flagmemory.pool import (
    BlockState,
    InvalidArgument,
    PoolForm,
    Purpose,
    ReusePolicy,
    TierSpec,
    WorkloadType,
    create_pool,
)
from fakes import CompletingBackend, NativeOrderedBackend


@pytest.fixture
def native_backend() -> NativeOrderedBackend:
    return NativeOrderedBackend(
        name="native-ordered", declared=frozenset(L0_CAPABILITIES | L1_CAPABILITIES)
    )


@pytest.fixture
def native_pool(native_backend: NativeOrderedBackend, tier_spec: TierSpec):
    return create_pool(
        native_backend,
        form=PoolForm.HANDLE,
        workload=WorkloadType.GENERATIVE,
        tier_spec=tier_spec,
    )


class TestNormal:
    """正常：能力标记如实反映；原生路按流取块还块；自管路是无条件可用的基线。"""

    def test_能力标记如实反映有没有原生原语(self, native_pool, handle_pool):
        assert native_pool.native_stream_ordered is True
        assert native_pool.query_capability().native_stream_ordered is True
        # 昆仑芯那一档：原生没有，走自管路
        assert handle_pool.native_stream_ordered is False
        assert handle_pool.query_capability().native_stream_ordered is False

    def test_原生路取块时把流带下去(self, native_pool, native_backend: NativeOrderedBackend):
        stream = native_pool.register_stream(FakeStream(stream_id=31))
        native_pool.alloc_on_stream(1 << 20, stream_id=stream, purpose=Purpose.ACTIVATION)
        assert native_backend.native_log == [("alloc", 1 << 20, 31)]

    def test_原生路还块时也走原生释放(self, native_pool, native_backend: NativeOrderedBackend):
        stream = native_pool.register_stream(FakeStream(stream_id=32))
        view = native_pool.alloc_on_stream(1 << 20, stream_id=stream, purpose=Purpose.ACTIVATION)
        native_pool.free_block(view.block_id)
        native_pool.drain_stream()
        native_pool.compact()
        assert native_backend.native_log == [
            ("alloc", 1 << 20, 32),
            ("free", 1 << 20, 32),
        ]

    def test_自管路无需任何原生能力就能跑完整生命周期(self, tensor_pool):
        """= 昆仑芯 P800 与老 CANN 昇腾机器那条必经路，只用 L0 能力。"""
        view = tensor_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        tensor_pool.free_block(view.block_id)
        assert tensor_pool.drain_stream().promoted_blocks == 1
        again = tensor_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        assert again.block_id == view.block_id


class TestBoundary:
    """边界：两条路的对外结果逐项相同；三个旋钮在两条路上同样生效。"""

    @pytest.mark.parametrize("native", [False, True])
    def test_两条路的对外语义相同(self, tier_spec: TierSpec, native: bool):
        cls = NativeOrderedBackend if native else CompletingBackend
        backend = cls(name=f"native={native}", declared=frozenset(L0_CAPABILITIES | L1_CAPABILITIES))
        pool = create_pool(backend, form=PoolForm.HANDLE, tier_spec=tier_spec)

        stream = pool.register_stream(FakeStream(stream_id=41))
        view = pool.alloc_on_stream(300 << 10, stream_id=stream, purpose=Purpose.ACTIVATION)
        assert (view.state, view.tier_bytes, view.owner_stream) == (
            BlockState.IN_USE,
            512 << 10,
            stream,
        )
        assert pool.free_block(view.block_id).state is BlockState.PENDING_RECYCLE
        assert pool.drain_stream().promoted_blocks == 1
        assert pool.block_view(view.block_id).state is BlockState.REUSABLE

        reused = pool.alloc_on_stream(512 << 10, stream_id=stream, purpose=Purpose.ACTIVATION)
        assert reused.block_id == view.block_id
        assert pool.query_audit().reuse_hits == 1
        assert pool.compact().returned_blocks == 0   # 已被复用走，没有可交回的

    @pytest.mark.parametrize("native", [False, True])
    def test_释放阈值旋钮在两条路上都生效(self, tier_spec: TierSpec, native: bool):
        cls = NativeOrderedBackend if native else CompletingBackend
        backend = cls(name=f"native={native}", declared=frozenset(L0_CAPABILITIES | L1_CAPABILITIES))
        pool = create_pool(
            backend,
            form=PoolForm.HANDLE,
            tier_spec=tier_spec,
            reuse=ReusePolicy(release_threshold_bytes=0),
        )
        view = pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        pool.free_block(view.block_id)
        pool.drain_stream()
        assert pool.query_watermark().pool_reusable == 0

    def test_原生路也受容量不足报错约束(self, tier_spec: TierSpec):
        class TightNative(NativeOrderedBackend):
            def alloc_from_pool_async(self, nbytes: int, stream, device_index: int = 0):
                raise MemoryError("原生池满")

        backend = TightNative(name="tight", declared=frozenset(L0_CAPABILITIES | L1_CAPABILITIES))
        pool = create_pool(backend, form=PoolForm.HANDLE, tier_spec=tier_spec)
        from flagmemory.pool import CapacityExhausted

        with pytest.raises(CapacityExhausted):
            pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)


class TestError:
    """异常：按事件依赖复用这个旋钮不许关 —— 关掉就违反状态机。"""

    def test_按事件依赖复用不提供关闭(self):
        with pytest.raises(InvalidArgument) as exc:
            ReusePolicy(follow_event_dependencies=False)
        assert "不提供关闭" in str(exc.value)

    @pytest.mark.parametrize("bad", [-1, "0", 1.5])
    def test_释放阈值必须是非负整数或None(self, bad):
        with pytest.raises(InvalidArgument):
            ReusePolicy(release_threshold_bytes=bad)  # type: ignore[arg-type]
