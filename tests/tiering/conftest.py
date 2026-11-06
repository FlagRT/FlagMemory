"""分层存储管理器用例的夹具：真池本体（假设备后端）+ 假搬运后端。

池用 tests/pool/fakes.py 的 `CompletingBackend`（record 即完成，L0+L1 声明照抄
调研 §5.1 的分级预设），搬运后端用本目录 `fakes.py` 的 `FakeTierBackend`
（主机内存字典模拟级二）。全部主机可跑，零真卡。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 复用池用例的假设备后端（tests/pool/fakes.py —— 刻意是普通模块而不是 conftest）
_POOL_TESTS = Path(__file__).resolve().parent.parent / "pool"
if str(_POOL_TESTS) not in sys.path:
    sys.path.insert(0, str(_POOL_TESTS))

from fakes import TEST_BASE_BYTES, CompletingBackend  # noqa: E402

from flagmemory.caps.capabilities import L0_CAPABILITIES, L1_CAPABILITIES
from flagmemory.pool import (
    PoolForm,
    Purpose,
    TierSpec,
    WorkloadType,
    create_pool,
)
from flagmemory.tiering.manager import TierPolicy, create_tier_manager
from tier_fakes import FakeTierBackend

#: 设备侧假显存 4 MiB（8 块 × 512 KiB 基档）；级二容量必须大于它（T-8）
DEVICE_TOTAL = 4 << 20
#: 级二假容量：设备总量的一倍（8 MiB，容纳 16 个基档块）
LEVEL2_CAP = DEVICE_TOTAL * 2
BLOCK_BYTES = TEST_BASE_BYTES  # 512 KiB

DEFAULT_POLICY = TierPolicy(
    h_dev=0.5,
    h_eng=0.7,
    l_dev=0.2,
    l_eng=0.2,
    hysteresis_n=1,
    cooldown_s=0.05,
    min_residency_s=0.05,
    budget_bytes=64 << 20,
    budget_window_s=1.0,
    backoff_trigger_max=3,
)


@pytest.fixture
def device_backend() -> CompletingBackend:
    return CompletingBackend(
        name="kunlunxin-like",
        declared=frozenset(L0_CAPABILITIES | L1_CAPABILITIES),
        total_bytes=DEVICE_TOTAL,
    )


@pytest.fixture
def pool(device_backend: CompletingBackend):
    return create_pool(
        device_backend,
        form=PoolForm.HANDLE,
        workload=WorkloadType.GENERATIVE,
        tier_spec=TierSpec(base_bytes=BLOCK_BYTES, step_bytes=4096, tier_count=6),
    )


@pytest.fixture
def tier_backend() -> FakeTierBackend:
    return FakeTierBackend()


@pytest.fixture
def manager(pool, tier_backend: FakeTierBackend):
    return create_tier_manager(
        pool, tier_backend, level2_capacity_bytes=LEVEL2_CAP, policy=DEFAULT_POLICY
    )


def alloc_kv_blocks(pool, tier_backend: FakeTierBackend, count: int):
    """铺 count 个 KV 缓存块（铺满设备侧即双指标到高位），并把假内容放进级一模拟。"""
    blocks = [
        pool.alloc_on_stream(BLOCK_BYTES, purpose=Purpose.KV_CACHE)
        for _ in range(count)
    ]
    for i, view in enumerate(blocks):
        tier_backend.device_blocks[view.block_id] = (
            f"kv-payload-{i}".encode() * (BLOCK_BYTES // 16)
        )
    return blocks
