"""昆仑芯分层存储管理器「异步在途轨」端到端真卡用例 —— **要卡**，默认不跑。

体例对齐 `test_ppu_tier_manager_realcard.py`（管理器 E2E 真卡套件样板）与昇腾
同构件 `test_npu_tier_manager_realcard.py`，覆盖异步增量：`evict_async` /
`promote_async` **发起即返回**，`poll_transfers` 完成确认后才落级二账 / 还块 /
交块（T-7 的异步形态）。2026-11-09 P800 真卡冒烟钉死的事实固化：

- 真卡张量级池（`RealCardBackend` / `RealCardAllocator`，torch.cuda 命名空间即
  torch_xmlir 符号重写后的 XPU 面）+ `XpuTierBackend.bind`（懒加载真
  `XpuCopyBackend`、锁页实测 True）+ `create_tier_manager` 全链编排；
- `evict_async` → `poll_transfers` 确认：确认前不落账不还块（异步轨与同步轨的
  分界判据）、确认后级二账本先行 + 旧租约失效、D2H 主机侧逐字节一致；
- `promote_async` → poll 确认后交块：新旧块号分叉（租约号不复用，设计 2.1）、
  copy_in 源行按条目级二落点行定位（目标行 = 新块号行，两套编号）、H2D 主机侧
  逐字节一致；
- 审计对账：evicted_bytes == promoted_bytes == tier_bytes、条目 1/1、零失败；
- destroy 双形态：无在途直接销毁 / 有在途先拒（不静默丢弃）后排空，in_flight 恒 ()。

⚠️ 本栈 API 面铁则（与 `test_xpu_backend_realcard.py` 同源）：`torch.xpu.*` 恒
不可用，一切设备操作按 `torch.cuda.*` 写（torch_xmlir 符号重写到 XPU）。两个
已钉死的坑一并遵守：主线程紧 synchronize 会饿死后台拷贝线程的 H2D 提交 ⇒ 轮询
每轮让出 ~10ms；torch.equal 跨设备不可靠 ⇒ 比较一律取回主机侧做。

跑法（P800 vllm0.20.2 系镜像一次性容器 —— operator-runtime v2 镜像里 connector
import 链断是已登记环境事实，勿用；空闲物理卡 <N> 映射为容器内 cuda:0，仓根在
PYTHONPATH 上）：

    docker run --rm --network none \
        --device /dev/xpu<N> --device /dev/xpuctrl \
        -v <repo>:/ws:ro -w /ws -e PYTHONPATH=/ws \
        <P800 vllm0.20.2 系镜像> \
        python -m pytest -m realcard -p no:cacheprovider \
        tests/tiering/test_xpu_tier_manager_realcard.py -v

池侧真卡底座与 PPU 套件（`test_ppu_tier_manager_realcard.py`）同构，刻意复制而
不跨测试目录 import（对方模块带模块级 skip 守卫，import 即炸；理由见对方模块
头）。与 PPU 不同：昆仑芯接线层把池块号直接当 connector 缓存张量行下标用
（`xpu_backend.copy_out / copy_in` 契约），无需行号重排包装 —— 但池租约号是
进程级单调发号，缓存行数给足余量并由 `_row()` 守卫。
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
    OwnershipViolation,
    PoolForm,
    PressureLevel,
    Purpose,
    TierSpec,
    WorkloadType,
    create_pool,
)
from flagmemory.tiering.backends.xpu_backend import XpuTierBackend
from flagmemory.tiering.manager import (
    AsyncEvictReceipt,
    AsyncPromoteReceipt,
    TierConfigError,
    TierError,
    TierPolicy,
    TransferDirection,
    TransferState,
    create_tier_manager,
)
from tier_fakes import FakeTierBackend

pytestmark = pytest.mark.realcard

torch = None  # 真卡夹具内才 import

DEVICE_INDEX = int(os.environ.get("FLAGMEMORY_REALCARD_DEVICE", "0"))

#: 缓存张量行数：池租约号是进程级单调发号（不回收复用，设计 2.1），行数给足
#: 余量；本文件独占进程跑（真卡容器一次一文件），实际发号远小于该值
N_CACHE_ROWS = 64


def _row(block_id: int) -> int:
    """昆仑芯接线层把池块号直接当缓存行下标用（bind 契约）；行数守卫。"""
    assert 0 <= block_id < N_CACHE_ROWS, (
        f"池租约号 {block_id} 超出缓存张量行数 {N_CACHE_ROWS}（测试配置项）"
    )
    return block_id


def _poll_until_confirmed(manager, *, transfer_id: str):
    """轮询推进在途搬运至完成确认（T-7：超时 + 轮询，不阻塞等待）。

    每轮 poll_transfers 之间同步设备 + **让出 10ms** —— 本栈已钉死的坑
    （2026-11-09 P800 真卡实测）：主线程紧密 synchronize 会饿死后台拷贝线程的
    H2D 提交，轮询间隔是完成语义的一部分；未确认的轮次顺手钉住「未确认不交块」
    判据。"""
    for _ in range(2000):
        report = manager.poll_transfers()
        for outcome in report.outcomes:
            if outcome.state is not TransferState.CONFIRMED:
                assert outcome.promoted is None, (
                    "未确认完成就交块 = 交出一块内容不全的显存（T-7 第 5 步）"
                )
        hit = [o for o in report.confirmed if o.transfer_id == transfer_id]
        if hit:
            return hit[0]
        torch.cuda.synchronize()
        time.sleep(0.01)
    pytest.fail(f"在途搬运 {transfer_id} 约 20s 内未完成确认（真卡拷贝异常）")


# ======================================================================
# 池侧真卡底座（与 tests/pool 同构 / PPU 套件同款，说明见模块头）
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
# 夹具
# ======================================================================


@pytest.fixture(scope="module")
def device_index() -> int:
    global torch
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")  # connector 依赖 vllm 面（import 链在 vllm0.20.2 系镜像才通）
    # 本栈铁则：torch.cuda.* 即 torch_xmlir 符号重写后的 XPU 面（torch.xpu.* 恒不可用）
    if not torch.cuda.is_available() or torch.cuda.device_count() <= DEVICE_INDEX:
        pytest.skip(f"这台机器上取不到 {DEVICE_INDEX} 号 XPU（torch_xmlir 重写面缺件）")
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
def bound_xpu_backend(device_index: int, measured_tier_spec: TierSpec):
    """bind 好的昆仑芯适配层（真 connector 懒加载）+ 块表张量一并交给用例。"""
    dev = torch.device(f"cuda:{DEVICE_INDEX}")
    block_bytes = measured_tier_spec.base_bytes
    gpu = torch.zeros(N_CACHE_ROWS, block_bytes, dtype=torch.int8, device=dev)
    cpu = torch.zeros(N_CACHE_ROWS, block_bytes, dtype=torch.int8, pin_memory=True)
    adapter = XpuTierBackend()  # backend_id 在用例里对齐池侧名
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


def _write_pattern(adapter, block_id: int, block_bytes: int):
    """级一放进已知内容（真实 KV 块的确定性字节形态）。

    本栈设备侧 int8 arange 可用（`test_xpu_backend_realcard.py` 实测路线），
    设备上构造再拷入目标行；判据 = 逐字节确定性内容，与构造路线无关。"""
    dev = adapter.gpu_cache.device
    pattern = torch.arange(block_bytes, dtype=torch.int8, device=dev)
    adapter.gpu_cache[_row(block_id)].copy_(pattern)
    torch.cuda.synchronize()
    return pattern.cpu().clone()


# ======================================================================
# 用例
# ======================================================================


def test_T8负例_级二容量不大于级一池容量报错不启(real_pool, bound_xpu_backend):
    bound_xpu_backend.backend_id = real_pool.query_capability().backend_name
    watermark = real_pool.query_watermark()
    with pytest.raises(TierConfigError, match="T-8"):
        create_tier_manager(
            real_pool,
            bound_xpu_backend,
            level2_capacity_bytes=watermark.device_total,
        )
    # 启动失败不回滚 bind，也不该留半个管理器
    assert bound_xpu_backend._bound


def test_真卡异步轨全链_发起即返回确认后落账交块(real_pool, bound_xpu_backend):
    torch.cuda.synchronize()
    pool_backend = real_pool.query_capability().backend_name
    bound_xpu_backend.backend_id = pool_backend

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
    # register_tier_backend：昆仑芯能力探测实值落账（锁页走 torch.cuda 符号重写
    # 面真锁，W1 实测 True）
    registration = manager.register_tier_backend(pool_backend, bound_xpu_backend)
    assert registration.backend_id == pool_backend
    assert registration.probed.supports_pinned is True
    assert registration.probed.has_batch_copy is False  # 昆仑芯无批量原语是登记事实
    assert registration.degradations == ()  # 声明与探测一致，无降级项

    # 搬运流登记（换出第 1 步）
    store_stream = torch.cuda.Stream()
    stream_id = manager.attach_stream(store_stream)

    # —— 注册真实 KV 条目形态的块 ——
    block_bytes = bound_xpu_backend._bound_block_bytes
    kv = real_pool.alloc_on_stream(block_bytes, purpose=Purpose.KV_CACHE)
    weight = real_pool.alloc_on_stream(block_bytes, purpose=Purpose.WEIGHT)
    kv_row = _row(kv.block_id)
    expected = _write_pattern(bound_xpu_backend, kv.block_id, block_bytes)

    # —— classify_placement：命中预期规则号 ——
    v_weight = manager.classify_placement(block_id=weight.block_id)
    assert v_weight.rule == "R1"  # 权重永不下放（T-3），与水位无关的稳定规则
    v_kv = manager.classify_placement(block_id=kv.block_id)
    if v_kv.reading.zone == "高":
        assert v_kv.rule == "R7"  # 判据 4/5 不可用（block_activity 恒 None）+ 高位
        assert v_kv.degraded  # 降级如实标出，不静默
    else:
        assert v_kv.rule == "R8"  # 未同时到高水位，搬运是净成本

    # —— evict_async：发起即返回（换出六步只做第 2、3 步）——
    receipt = manager.evict_async(
        [kv.block_id], pressure=PressureLevel.HIGH, stream_id=stream_id
    )
    entry_id = f"b{kv.block_id}"
    assert isinstance(receipt, AsyncEvictReceipt)
    assert receipt.met_target is True
    assert receipt.scheduled_entries == (entry_id,)
    assert receipt.scheduled_bytes == kv.tier_bytes
    # 昆仑芯接线层 block_activity 恒 None ⇒ 中性序（登记事实，确定性判据）
    assert receipt.ordering_basis == ("R7",)
    assert len(receipt.transfers) == 1
    ticket = receipt.transfers[0]
    assert ticket.state is TransferState.PENDING
    assert ticket.direction is TransferDirection.EVICT
    assert ticket.entry_id == entry_id
    assert ticket.nbytes == kv.tier_bytes

    # 未确认前：不落级二账本、不还块（异步轨与同步轨的分界判据，T-7）
    tier_wm = manager.query_tier_watermark()
    assert tier_wm.level2.used_bytes == 0
    assert tier_wm.level2.entry_count == 0
    real_pool.block_view(kv.block_id)  # 旧租约仍在用（未确认不还块）
    assert len(manager.query_transfers()) == 1

    # —— poll_transfers：完成确认后才落账 / 还块（账本先行，T-7）——
    outcome = _poll_until_confirmed(manager, transfer_id=ticket.transfer_id)
    assert outcome.direction is TransferDirection.EVICT
    assert outcome.entry_id == entry_id
    assert outcome.nbytes == kv.tier_bytes
    tier_wm = manager.query_tier_watermark()
    assert tier_wm.level2.used_bytes == kv.tier_bytes  # 条目先落级二账本
    assert tier_wm.level2.entry_count == 1
    with pytest.raises(OwnershipViolation):
        real_pool.block_view(kv.block_id)  # 后 free_block：旧租约号失效
    # D2H 实际字节核对：级二副本与级一换出前内容逐字节一致（跨设备 equal 不可靠，
    # 比较一律取回主机侧做 —— 2026-11-09 登记的栈级坑）
    assert torch.equal(bound_xpu_backend.cpu_cache[kv_row].cpu(), expected)
    assert len(manager.query_transfers()) == 0

    # 在条目真正的级二落点行上改数：证换入源行按条目定位（新旧行分叉判据）
    bound_xpu_backend.cpu_cache[kv_row][:8] = 7
    expected_after_edit = expected.clone()
    expected_after_edit[:8] = 7

    # —— promote_async：发起即返回，块标识未交付（T-7 第 5 步）——
    promo = manager.promote_async(entry_id)
    assert isinstance(promo, AsyncPromoteReceipt)
    assert promo.entry_id == entry_id
    assert promo.nbytes == kv.tier_bytes
    inflight = manager.query_transfers(entry_id=entry_id)
    assert len(inflight) == 1
    assert inflight[0].state is TransferState.PENDING

    outcome = _poll_until_confirmed(manager, transfer_id=promo.transfer_id)
    delivered = outcome.promoted
    assert delivered is not None  # 完成确认后才交块
    new_block_id = delivered.block_id
    assert new_block_id != kv.block_id  # 新旧块号分叉：租约号不复用（设计 2.1）
    real_pool.block_view(new_block_id)  # 新租约已生效（交给调用方）
    assert delivered.nbytes == kv.tier_bytes
    assert delivered.kept_tier2_copy is False
    # H2D 实际字节核对：目标行 = 新块号行，内容 = 条目源行改数后的内容（源行按
    # 条目定位，绝非 block_id 当源行 —— 两套编号契约）
    torch.cuda.synchronize()
    assert torch.equal(
        bound_xpu_backend.gpu_cache[_row(new_block_id)].cpu(), expected_after_edit
    )
    # 级二账已清（未保留副本）
    tier_wm = manager.query_tier_watermark()
    assert tier_wm.level2.used_bytes == 0
    assert tier_wm.level2.entry_count == 0

    # —— 审计对账：换出/换入字节相等、条目 1/1、零搬运失败 ——
    audit = manager.query_tier_audit()
    assert audit.evicted_bytes == audit.promoted_bytes == kv.tier_bytes
    assert audit.evicted_entries == audit.promoted_entries == 1
    assert audit.copy_failures == 0

    # 收尾：交还其余块，销毁管理器（无在途形态）
    real_pool.free_block(weight.block_id)
    real_pool.free_block(new_block_id)
    real_pool.drain_stream(timeout_s=5.0)
    report = manager.destroy()
    assert report.in_flight == ()
    assert report.timed_out is False


def test_destroy_有在途搬运_不静默丢弃排空后才销毁(real_pool, bound_xpu_backend):
    pool_backend = real_pool.query_capability().backend_name
    bound_xpu_backend.backend_id = pool_backend
    watermark = real_pool.query_watermark()
    manager = create_tier_manager(
        real_pool,
        bound_xpu_backend,
        level2_capacity_bytes=watermark.device_total * 2,
        policy=TierPolicy(
            h_dev=0.5, h_eng=0.5, l_dev=0.0, l_eng=0.0,
            hysteresis_n=1, cooldown_s=0.05, min_residency_s=0.05,
        ),
    )
    stream_id = manager.attach_stream(torch.cuda.Stream())
    block_bytes = bound_xpu_backend._bound_block_bytes

    # 第一块：发起 → poll 确认 → 审计可读（销毁后审计口关闭，先读）
    kv1 = real_pool.alloc_on_stream(block_bytes, purpose=Purpose.KV_CACHE)
    _write_pattern(bound_xpu_backend, kv1.block_id, block_bytes)
    receipt1 = manager.evict_async(
        [kv1.block_id], pressure=PressureLevel.HIGH, stream_id=stream_id
    )
    assert receipt1.scheduled_entries == (f"b{kv1.block_id}",)
    _poll_until_confirmed(manager, transfer_id=receipt1.transfers[0].transfer_id)
    audit = manager.query_tier_audit()
    assert audit.evicted_entries == 1
    assert audit.copy_failures == 0

    # 第二块：发起后**留在途**，验 destroy 双形态
    kv2 = real_pool.alloc_on_stream(block_bytes, purpose=Purpose.KV_CACHE)
    _write_pattern(bound_xpu_backend, kv2.block_id, block_bytes)
    receipt2 = manager.evict_async(
        [kv2.block_id], pressure=PressureLevel.HIGH, stream_id=stream_id
    )
    assert len(manager.query_transfers()) == 1

    # 有在途 + 不等待 ⇒ 报错并附清单，不静默丢弃（6.1 第 2 条）
    with pytest.raises(TierError, match="不静默丢弃"):
        manager.destroy(wait_for_drain=False)

    # 排空形态：destroy 自轮询排空在途，之后才标销毁；in_flight 恒 ()
    report = manager.destroy(timeout_s=30.0)
    assert report.in_flight == ()
    assert report.timed_out is False
    # 排空 = 第二笔换出确认落账：两笔都成级二残留条目（未被换入，如实上报）
    assert {e.entry_id for e in report.residual_entries} == {
        f"b{kv1.block_id}", f"b{kv2.block_id}"
    }
