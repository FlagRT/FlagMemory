"""PPU 分层存储管理器端到端真卡编排用例 —— **要卡**，默认不跑。

在真卡上经 PPU 搬运适配层完整编排一轮：
create_pool → T-8 负例 → create_tier_manager → register_tier_backend（PPU 能力
探测实值落账）→ 注册真实 KV 条目形态的块 → classify_placement（断言命中预期
规则号）→ evict（账本先行 + free_block + 实际换出字节如实）→ promote（完成确认
后才交块、内容与换出前逐字节一致）。覆盖设计文档 §3.4 换出六步 / 换入五步在
真卡上的时序。

跑法（PTG_0 一次性容器，PPU 0 号卡，仓根在 PYTHONPATH 上）：

    docker run --rm --device /dev/alixpu_ppu0 --device /dev/alixpu \
        --device /dev/alixpu_ctl -v <repo>:/ws -w /ws \
        <vllm024 镜像> bash -c "source /usr/local/PPU_SDK/envsetup.sh && \
        UMD_PLATFORM_TYPE=1 HGGC_DRIVER_CANDIDATE=UMD \
        FLAGMEMORY_ENABLE_LEGACY_CUDA=I_ACCEPT_LEGACY_RISKS \
        python -m pytest -v tests/tiering/test_ppu_tier_manager_realcard.py"

池侧真卡底座（RealCardBackend / RealCardAllocator / measure_granularity）与
`tests/pool/test_pool_realcard.py` 同构（torch.cuda 兼容命名空间 + 张量级底座 +
实测粒度），刻意复制而不跨测试目录 import（对方模块带模块级 skip 守卫，import
即炸）。块号 → 缓存行号的重排由 `_RowMappedPpuTierBackend` 承担：池的块号是
租约号，与 connector 缓存张量的行下标是两套编号，测试侧维护映射。
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

# tier_fakes 是 tests/tiering 下的普通模块（同 conftest 的路径注入法）
_TIERING_TESTS = Path(__file__).resolve().parent
if str(_TIERING_TESTS) not in sys.path:
    sys.path.insert(0, str(_TIERING_TESTS))

from flagmemory.caps.backend import EventNotRecorded, MemoryInfo
from flagmemory.caps.capabilities import L0_CAPABILITIES
from flagmemory.pool import (
    PoolForm,
    Purpose,
    TierSpec,
    WorkloadType,
    create_pool,
)
from flagmemory.tiering.backends.ppu_backend import (
    PINNED_LEVEL_HOST_REGISTER,
    PINNED_LEVEL_PIN_MEMORY,
    PpuTierBackend,
)
from flagmemory.tiering.manager import (
    TierConfigError,
    TierPolicy,
    create_tier_manager,
)
from tier_fakes import FakeTierBackend

pytestmark = pytest.mark.realcard

if (os.environ.get("FLAGMEMORY_ENABLE_LEGACY_CUDA") != "I_ACCEPT_LEGACY_RISKS"
        or not any(arg.replace("\\", "/").endswith(
               "/test_ppu_tier_manager_realcard.py")
               or arg == "test_ppu_tier_manager_realcard.py"
               for arg in sys.argv[1:])):
    pytest.skip("legacy CUDA tier-manager suite disabled", allow_module_level=True)

torch = None  # 真卡夹具内才 import

DEVICE_INDEX = int(os.environ.get("FLAGMEMORY_REALCARD_DEVICE", "0"))

#: 缓存张量行数：池租约号与行下标两套编号，测试侧映射，行数给足余量
N_CACHE_ROWS = 64


# ======================================================================
# 池侧真卡底座（与 tests/pool/test_pool_realcard.py 同构，说明见模块头）
# ======================================================================


@dataclass
class RealCardBackend:
    device_index: int = 0
    _recorded: set[int] = field(default_factory=set, repr=False)

    @property
    def name(self) -> str:
        return "realcard-tensor-level"

    def declared_capabilities(self):
        return frozenset(L0_CAPABILITIES)

    def device_memory_info(self, device_index: int = 0) -> MemoryInfo:
        free, total = torch.cuda.mem_get_info(device_index)
        return MemoryInfo(total=int(total), used=int(total - free), free=int(free))

    def current_stream(self, device_index: int = 0) -> Any:
        return torch.cuda.current_stream(device_index)

    def streams_equal(self, left: Any, right: Any) -> bool:
        return left == right

    def validate_stream_device(self, stream: Any, device_index: int) -> None:
        device = getattr(stream, "device", None)
        index = getattr(device, "index", None)
        if index is None or int(index) != int(device_index):
            raise ValueError(
                f"流对象 {stream!r} 无法确认属于设备 {device_index}"
                f"（读到的 device={device!r}）"
            )

    def create_event(self, device_index: int = 0) -> Any:
        return torch.cuda.Event()

    def record_event(self, event: Any, stream: Any) -> None:
        event.record(stream)
        self._recorded.add(id(event))

    def query_event(self, event: Any) -> bool:
        if id(event) not in self._recorded:
            raise EventNotRecorded("事件未 record 就查询（该栈原生误报已完成）。")
        return bool(event.query())

    def synchronize_event(self, event: Any, timeout_s: float) -> bool:
        raise AssertionError("池不该在屏障里调有界同步（I-11）")


@dataclass
class RealCardAllocator:
    device_index: int = 0
    live: dict[int, Any] = field(default_factory=dict, repr=False)
    _next: int = field(default=1, repr=False)

    def allocate(self, nbytes: int, device_index: int = 0, *, stream: Any) -> Any:
        try:
            with torch.cuda.stream(stream):
                tensor = torch.empty(
                    nbytes, dtype=torch.int8, device=f"cuda:{device_index}"
                )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise MemoryError(str(exc)) from exc
            raise
        key = self._next
        self._next += 1
        self.live[key] = tensor
        return ("tensor", key, nbytes)

    def release(self, base_ref: Any) -> None:
        key = base_ref[1]
        if key not in self.live:
            raise RuntimeError(f"底座引用 {base_ref} 不在底座的在世表里")
        del self.live[key]


def measure_granularity(device_index: int, probe_bytes: int = 1) -> tuple[int, int]:
    torch.cuda.empty_cache()
    before = torch.cuda.memory_reserved(device_index)
    probe = torch.empty(probe_bytes, dtype=torch.int8, device=f"cuda:{device_index}")
    after = torch.cuda.memory_reserved(device_index)
    del probe
    torch.cuda.empty_cache()
    grain = max(after - before, 0)
    if grain <= 0:
        pytest.skip("量不到底座块粒度（预留量没涨），须改用显式档位配置项")
    return grain, grain


# ======================================================================
# PPU 适配层的行号重排包装
# ======================================================================


class _RowMappedPpuTierBackend(PpuTierBackend):
    """池租约块号 → connector 缓存行号 的映射包装（测试侧账）。

    管理器把池的块号原样递给 copy_out / copy_in；connector 的
    DmaCopyBackend 把同一编号当缓存张量行下标用（且 load 方向 src/dst 行号
    相同 —— gpu/cpu 缓存共用一套编号）。两套编号在真卡测试里由本类翻译：
    换出时按池块号领行并记 entry_id → 行；换入按 entry_id 找回同一行
    （connector 语义 = 级二副本原行回到级一同行）。
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.row_of: dict[int, int] = {}
        self.entry_rows: dict[str, int] = {}
        self._next_row = 0

    def map_block(self, block_id: int) -> int:
        if block_id not in self.row_of:
            self.row_of[block_id] = self._next_row
            self._next_row += 1
        row = self.row_of[block_id]
        assert row < N_CACHE_ROWS, "缓存行数给少了（测试配置项）"
        return row

    def copy_out(self, entry_id: str, block_id: int, nbytes: int, *, stream_id: int):
        row = self.map_block(block_id)
        self.entry_rows[entry_id] = row
        return super().copy_out(entry_id, row, nbytes, stream_id=stream_id)

    def copy_in(self, entry_id: str, block_id: int, nbytes: int, *, stream_id: int):
        row = self.entry_rows[entry_id]  # 级二副本所在行 = 级一落点行（connector 语义）
        return super().copy_in(entry_id, row, nbytes, stream_id=stream_id)

    def confirm(self, ticket: Any) -> bool:
        """管理器侧完成确认轮询无间隔（8 次即刻轮询），真卡 DMA 线程可能来不及
        入事件表 ⇒ 未确认时同步设备 + 小睡重问（与 backend 真卡用例同一校准
        手段，不改语义：只信 record 后入表的事件）。"""
        if super().confirm(ticket):
            return True
        for _ in range(200):
            torch.cuda.synchronize()
            if super().confirm(ticket):
                return True
            time.sleep(0.01)
        return super().confirm(ticket)


# ======================================================================
# 夹具
# ======================================================================


@pytest.fixture(scope="module")
def device_index() -> int:
    global torch
    import torch as torch_module
    torch = torch_module
    if not torch.cuda.is_available() or torch.cuda.device_count() <= DEVICE_INDEX:
        pytest.skip(f"这台机器上取不到 {DEVICE_INDEX} 号卡")
    torch.cuda.set_device(DEVICE_INDEX)
    return DEVICE_INDEX


@pytest.fixture(scope="module")
def measured_tier_spec(device_index: int) -> TierSpec:
    minimum, recommended = measure_granularity(device_index)
    return TierSpec(base_bytes=recommended, step_bytes=minimum, tier_count=8)


@pytest.fixture
def real_pool(device_index: int, measured_tier_spec: TierSpec):
    pool = create_pool(
        RealCardBackend(device_index=device_index),
        device_index=device_index,
        form=PoolForm.TENSOR,
        workload=WorkloadType.GENERATIVE,
        base_allocator=RealCardAllocator(device_index=device_index),
        tier_spec=measured_tier_spec,
    )
    yield pool
    if not pool.destroyed:
        try:
            pool.destroy_pool()
        except Exception:
            pass
    torch.cuda.empty_cache()


@pytest.fixture
def bound_ppu_backend(device_index: int, measured_tier_spec: TierSpec):
    """bind 好的 PPU 适配层 + 行号映射（gpu/cpu 缓存张量一并交给用例）。"""
    dev = torch.device(f"cuda:{DEVICE_INDEX}")
    block_bytes = measured_tier_spec.base_bytes
    gpu = torch.zeros(
        N_CACHE_ROWS, block_bytes, dtype=torch.int8, device=dev
    )
    cpu = torch.zeros(
        N_CACHE_ROWS, block_bytes, dtype=torch.int8, pin_memory=True
    )
    adapter = _RowMappedPpuTierBackend()  # backend_id 在用例里对齐池侧名
    adapter.bind(
        {"layer.kv": gpu},
        {"layer.kv": cpu},
        dev,
        torch.cuda.Stream(),
        torch.cuda.Stream(),
        block_bytes=block_bytes,
    )
    adapter.gpu_cache = gpu
    adapter.cpu_cache = cpu
    yield adapter
    adapter.shutdown()


# ======================================================================
# 用例
# ======================================================================


def test_T8负例_级二容量不大于级一池容量报错不启(real_pool, bound_ppu_backend):
    bound_ppu_backend.backend_id = real_pool.query_capability().backend_name
    watermark = real_pool.query_watermark()
    with pytest.raises(TierConfigError, match="T-8"):
        create_tier_manager(
            real_pool,
            bound_ppu_backend,
            level2_capacity_bytes=watermark.device_total,
        )
    # 级二水位也不该有账（管理器没起来）
    assert bound_ppu_backend._bound


def test_真卡端到端编排_注册分类换出换入全链路(real_pool, bound_ppu_backend):
    from flagmemory.pool import OwnershipViolation, PressureLevel

    torch.cuda.synchronize(torch.device(f"cuda:{DEVICE_INDEX}"))
    pool_backend = real_pool.query_capability().backend_name
    bound_ppu_backend.backend_id = pool_backend

    # —— create_tier_manager：先以假后端起壳，再走 register_tier_backend 真路径 ——
    policy = TierPolicy(
        h_dev=0.5, h_eng=0.5, l_dev=0.0, l_eng=0.0,
        hysteresis_n=1, cooldown_s=0.05, min_residency_s=0.05,
    )
    watermark = real_pool.query_watermark()
    manager = create_tier_manager(
        real_pool,
        FakeTierBackend(backend_id=pool_backend),
        level2_capacity_bytes=watermark.device_total * 2,
        policy=policy,
    )
    # register_tier_backend：PPU 能力探测实值落账（锁页三级链在真卡上实测）
    registration = manager.register_tier_backend(pool_backend, bound_ppu_backend)
    assert registration.backend_id == pool_backend
    assert registration.probed.supports_pinned is True
    assert bound_ppu_backend.probed_pinned_level in (
        PINNED_LEVEL_HOST_REGISTER,
        PINNED_LEVEL_PIN_MEMORY,
    )
    assert registration.probed.has_batch_copy is False  # PPU 无批量原语是登记事实
    assert registration.degradations == ()  # 声明探测一致，无降级项

    # 搬运流登记（换出第 1 步）
    store_stream = torch.cuda.Stream()
    manager.attach_stream(store_stream)

    # —— 注册真实 KV 条目形态的块 ——
    block_bytes = bound_ppu_backend._bound_block_bytes
    kv = real_pool.alloc_on_stream(block_bytes, purpose=Purpose.KV_CACHE)
    weight = real_pool.alloc_on_stream(block_bytes, purpose=Purpose.WEIGHT)
    kv_row = bound_ppu_backend.map_block(kv.block_id)

    # 级一放进已知内容（真实 KV 块的字节形态：确定性模式）
    pattern = torch.arange(block_bytes, dtype=torch.int32).to(torch.int8)
    bound_ppu_backend.gpu_cache[kv_row].copy_(
        pattern.to(device=bound_ppu_backend.gpu_cache.device)
    )
    torch.cuda.synchronize(torch.device(f"cuda:{DEVICE_INDEX}"))
    expected = pattern.clone()

    # —— classify_placement：命中预期规则号 ——
    v_weight = manager.classify_placement(block_id=weight.block_id)
    assert v_weight.rule == "R1"  # 权重永不下放（T-3），与水位无关的稳定规则
    v_kv = manager.classify_placement(block_id=kv.block_id)
    zone = v_kv.reading.zone
    if zone == "高":
        assert v_kv.rule == "R7"  # 判据 4/5 不可用（block_activity 恒 None）+ 高位
        assert v_kv.degraded  # 降级如实标出，不静默
    else:
        assert v_kv.rule == "R8"  # 未同时到高水位，搬运是净成本
    assert v_kv.target.value in ("级一·设备显存", "级二·主机内存")

    # —— evict：换出六步（E2 压力口径，不依赖真卡整机水位）——
    result = manager.evict([kv.block_id], pressure=PressureLevel.HIGH)
    entry_id = f"b{kv.block_id}"
    assert result.evicted_entries == (entry_id,)
    assert result.evicted_bytes == kv.tier_bytes  # 实际换出字节如实
    assert result.met_target is True

    # 账本先行（T-7）：条目先落级二账本
    tier_wm = manager.query_tier_watermark()
    assert tier_wm.level2.used_bytes == kv.tier_bytes
    assert tier_wm.level2.entry_count == 1
    # 后 free_block：旧租约号必须被池账本拒绝（块已进池内流转）
    with pytest.raises(OwnershipViolation):
        real_pool.block_view(kv.block_id)

    # D2H 实际字节核对：级二副本与级一换出前内容逐字节一致
    torch.cuda.synchronize(torch.device(f"cuda:{DEVICE_INDEX}"))
    assert torch.equal(bound_ppu_backend.cpu_cache[kv_row].cpu(), expected)

    # —— promote：换入五步（完成确认后才交块）——
    promoted = manager.promote(entry_id)
    assert promoted.nbytes == kv.tier_bytes
    assert promoted.kept_tier2_copy is False
    torch.cuda.synchronize(torch.device(f"cuda:{DEVICE_INDEX}"))
    # connector 语义：级二副本行 = 级一落点行（gpu/cpu 缓存共用行编号）
    assert torch.equal(
        bound_ppu_backend.gpu_cache[kv_row].cpu(), expected
    ), "换入后内容必须与换出前逐字节一致"
    # 级二账已清（未保留副本）
    assert manager.query_tier_watermark().level2.used_bytes == 0
    assert manager.query_tier_watermark().level2.entry_count == 0

    # —— 审计：换出/换入字节对账、零搬运失败 ——
    audit = manager.query_tier_audit()
    assert audit.evicted_bytes == audit.promoted_bytes == kv.tier_bytes
    assert audit.evicted_entries == audit.promote_calls == 1
    assert audit.copy_failures == 0

    # 收尾：交还其余块，销毁管理器
    real_pool.free_block(weight.block_id)
    real_pool.free_block(promoted.block_id)
    real_pool.drain_stream(timeout_s=5.0)
    report = manager.destroy()
    assert report.in_flight == ()
