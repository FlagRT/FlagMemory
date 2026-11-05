"""自管流序与未启用原生优化的边界。全部不依赖真卡。

后端即使有原生方法也不得自动绕过当前自管回收屏障；启用优化须另验完整释放寿命。
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
    """正常：能力标记反映实际启用路径，底座生命周期均走自管回收屏障。"""

    def test_能力标记如实反映原生优化未启用(self, native_pool, handle_pool):
        assert native_pool.native_stream_ordered is False
        assert native_pool.query_capability().native_stream_ordered is False
        # 昆仑芯那一档：原生没有，走自管路
        assert handle_pool.native_stream_ordered is False
        assert handle_pool.query_capability().native_stream_ordered is False

    def test_后端有原生方法也不自动启用(self, native_pool, native_backend: NativeOrderedBackend):
        stream = native_pool.register_stream(FakeStream(stream_id=31))
        native_pool.alloc_on_stream(1 << 20, stream_id=stream, purpose=Purpose.ACTIVATION)
        assert native_backend.native_log == []

    def test_释放不走未验证原生原语(self, native_pool, native_backend: NativeOrderedBackend):
        stream = native_pool.register_stream(FakeStream(stream_id=32))
        view = native_pool.alloc_on_stream(1 << 20, stream_id=stream, purpose=Purpose.ACTIVATION)
        native_pool.free_block(view.block_id)
        native_pool.drain_stream()
        native_pool.compact()
        assert native_backend.native_log == []
        assert native_backend.live_handles() == 0

    def test_自管路无需任何原生能力就能跑完整生命周期(self, tensor_pool):
        """= 昆仑芯 P800 与老 CANN 昇腾机器那条必经路，只用 L0 能力。"""
        view = tensor_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        tensor_pool.free_block(view.block_id)
        assert tensor_pool.drain_stream().promoted_blocks == 1
        again = tensor_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        assert again.storage_id == view.storage_id
        assert again.block_id != view.block_id


class TestBoundary:
    """边界：后端是否带原生方法不影响自管语义和回收策略。"""

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
        assert pool._ledger._blocks[view.storage_id].state is BlockState.REUSABLE

        reused = pool.alloc_on_stream(512 << 10, stream_id=stream, purpose=Purpose.ACTIVATION)
        assert reused.storage_id == view.storage_id
        assert reused.block_id != view.block_id
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

    def test_未验证原生分配器不会绕过自管底座(self, tier_spec: TierSpec):
        class TightNative(NativeOrderedBackend):
            def alloc_from_pool_async(self, nbytes: int, stream, device_index: int = 0):
                raise MemoryError("原生池满")

        backend = TightNative(name="tight", declared=frozenset(L0_CAPABILITIES | L1_CAPABILITIES))
        pool = create_pool(backend, form=PoolForm.HANDLE, tier_spec=tier_spec)
        lease = pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        assert lease.state is BlockState.IN_USE
        assert backend.native_log == []


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
