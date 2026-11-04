"""池用例的夹具。假后端与假底座在 `tests/pool/fakes.py`，这里只装配夹具。

四家的能力声明照抄 `flagmemory.caps.fake` 的预设（调研 §5.1 的分级），本文件不重新论证。
"""

from __future__ import annotations

import pytest

from flagmemory.caps.capabilities import Capability, L0_CAPABILITIES, L1_CAPABILITIES
from flagmemory.caps.fake import FakeBackend
from flagmemory.pool import PoolForm, TierSpec, WorkloadType, create_pool
from fakes import (
    TEST_BASE_BYTES,
    TEST_STEP_BYTES,
    TEST_TIER_COUNT,
    CompletingBackend,
    FakeBaseAllocator,
    ObservingBackend,
)


@pytest.fixture
def tier_spec() -> TierSpec:
    """用例的档位配置项（= 昆仑芯那一格只能从外面喂粒度的那条路）。"""
    return TierSpec(
        base_bytes=TEST_BASE_BYTES, step_bytes=TEST_STEP_BYTES, tier_count=TEST_TIER_COUNT
    )


@pytest.fixture
def base_allocator() -> FakeBaseAllocator:
    return FakeBaseAllocator()


@pytest.fixture
def handle_backend_done() -> CompletingBackend:
    """句柄形态 + 事件立刻完成（= 昆仑芯 P800 那一档：L0 + L1，无原生流有序）。"""
    return CompletingBackend(
        name="kunlunxin-like", declared=frozenset(L0_CAPABILITIES | L1_CAPABILITIES)
    )


@pytest.fixture
def l0_backend_done() -> CompletingBackend:
    """只有 L0 的后端（= 寒武纪 MLU590 那一档：句柄、跨流保护、上下文观测全未声明）。"""
    return CompletingBackend(name="cambricon-like", declared=frozenset(L0_CAPABILITIES))


@pytest.fixture
def pending_backend() -> FakeBackend:
    """事件永不完成的后端：用来验屏障超时是可恢复态、不是失败。"""
    return FakeBackend(name="never-done", declared=frozenset(L0_CAPABILITIES | L1_CAPABILITIES))


@pytest.fixture
def all_caps_backend() -> ObservingBackend:
    """四级全声明（= 昇腾 910C 那一档），带上下文只读观测。"""
    return ObservingBackend(
        name="ascend-like",
        declared=frozenset(
            L0_CAPABILITIES
            | L1_CAPABILITIES
            | {Capability.CONTEXT_LIFECYCLE, Capability.TRUE_BOUNDED_SYNC}
        ),
    )


@pytest.fixture
def handle_pool(handle_backend_done: CompletingBackend, tier_spec: TierSpec):
    """句柄形态的池，生成式负载。"""
    return create_pool(
        handle_backend_done,
        form=PoolForm.HANDLE,
        workload=WorkloadType.GENERATIVE,
        tier_spec=tier_spec,
    )


@pytest.fixture
def tensor_pool(
    l0_backend_done: CompletingBackend,
    base_allocator: FakeBaseAllocator,
    tier_spec: TierSpec,
):
    """张量级形态的池（四家通用的默认路径），只有 L0 的后端。"""
    return create_pool(
        l0_backend_done,
        form=PoolForm.TENSOR,
        workload=WorkloadType.EMBEDDING,
        base_allocator=base_allocator,
        tier_spec=tier_spec,
    )


@pytest.fixture
def pending_pool(pending_backend: FakeBackend, tier_spec: TierSpec):
    """事件永不完成的池：屏障一定超时。"""
    return create_pool(
        pending_backend,
        form=PoolForm.HANDLE,
        workload=WorkloadType.GENERATIVE,
        tier_spec=tier_spec,
        barrier_timeout_s=0.0,
        timeout_threshold=2,
    )
