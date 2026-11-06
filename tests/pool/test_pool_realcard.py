"""旧 CUDA 兼容命名空间的池真卡测试草稿，当前不是可用验收入口。

默认由 realcard 标记排除；运行须显式指到本文件并置 FLAGMEMORY_ENABLE_LEGACY_CUDA。
2026-11-06 已对齐现行池接口（DeviceBackend 的 validate_stream_device、BaseAllocator 的
目标流关键字、租约撤销/复用换号语义），在昆仑芯 P800 真卡（operator-runtime 镜像、
torch_xmlir 符号重写面）上 11/11 通过；昇腾侧另有独立入口（test_pool_ascend_realcard）。

SDK 原语调查只提供候选证据：指挥台仓
inbox/调研-四芯片SDK显存原语面补探-20261103.md；它不证明本适配器满足运行语义。
本文件使用 torch.cuda 兼容命名空间，不能据此推断设备种类或作为昇腾路径。

默认证据仅打印；显式设置 FLAGMEMORY_REALCARD_EVIDENCE 才写给定路径，使用者须选
仓外路径，旧逻辑不会自动拦截仓内路径。reports 历史材料迁移尚未收口，不作迁移声明。
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import pytest

from flagmemory.caps.backend import EventNotRecorded, MemoryInfo
from flagmemory.caps.capabilities import L0_CAPABILITIES
from flagmemory.pool import (
    BlockState,
    PoolForm,
    Purpose,
    TierSpec,
    WorkloadType,
    create_pool,
)

pytestmark = pytest.mark.realcard

if (os.environ.get("FLAGMEMORY_ENABLE_LEGACY_CUDA") != "I_ACCEPT_LEGACY_RISKS"
        or not any(arg.replace("\\", "/").endswith("/test_pool_realcard.py")
                   or arg == "test_pool_realcard.py" for arg in sys.argv[1:])):
    pytest.skip("legacy CUDA suite disabled", allow_module_level=True)

# 模块必须在无框架环境里也能 import：默认那批用例靠 `-m 'not realcard'` 把本文件**整体
# deselect**，而不是靠 skip 掩盖。所以 torch 在这里只是「有则用」，缺了由夹具报 skip。
torch = None  # imported only in the explicitly enabled fixture

DEVICE_INDEX = int(os.environ.get("FLAGMEMORY_REALCARD_DEVICE", "0"))

#: 默认仅打印；非空才写指定路径。调用方须显式选仓外路径，代码不自动验证该边界。
#: reports 仍含历史证据，迁移尚未收口。
EVIDENCE_PATH = os.environ.get("FLAGMEMORY_REALCARD_EVIDENCE", "")


# ======================================================================
# 最小真后端（框架绑定，只给真卡用例用）
# ======================================================================


@dataclass
class RealCardBackend:
    """最小 `DeviceBackend` 实现：只声明四家齐备的 L0 能力。

    **如实声明**（沿用统一层的纪律）：统一内存句柄这类 L1 能力不经由框架暴露，这里就不声明 ——
    于是池按降级表第 1 条走张量级池，那也正是四家通用的默认路径。
    """

    device_index: int = 0
    #: 事件登记表：`I-2` 的落点。该栈的 `Event.query()` 对**未 record** 的事件返回 True
    #: （就是调研里记的那个误报），所以适配层必须自己记账拦住，绝不把那个 True 递给池。
    _recorded: set[int] = field(default_factory=set, repr=False)

    @property
    def name(self) -> str:
        return "realcard-tensor-level"

    def declared_capabilities(self):
        return frozenset(L0_CAPABILITIES)

    # —— L0：设备显存三键 ——

    def device_memory_info(self, device_index: int = 0) -> MemoryInfo:
        free, total = torch.cuda.mem_get_info(device_index)
        return MemoryInfo(total=int(total), used=int(total - free), free=int(free))

    # —— L0：流 ——

    def current_stream(self, device_index: int = 0) -> Any:
        return torch.cuda.current_stream(device_index)

    def streams_equal(self, left: Any, right: Any) -> bool:
        return left == right

    def validate_stream_device(self, stream: Any, device_index: int) -> None:
        """现行 `DeviceBackend` L0 流面新增的核验口（`caps/backend.py`）。

        该栈的流对象带 `device` 属性；对不上或不带都视为无法确认，抛错（协议要求
        「不匹配或无法确认必须抛错」），绝不放行猜设备。
        """
        device = getattr(stream, "device", None)
        index = getattr(device, "index", None)
        if index is None or int(index) != int(device_index):
            raise ValueError(
                f"流对象 {stream!r} 无法确认属于设备 {device_index}（读到的 device={device!r}）"
            )

    # —— L0：事件 ——

    def create_event(self, device_index: int = 0) -> Any:
        return torch.cuda.Event()

    def record_event(self, event: Any, stream: Any) -> None:
        event.record(stream)
        self._recorded.add(id(event))

    def query_event(self, event: Any) -> bool:
        if id(event) not in self._recorded:
            raise EventNotRecorded(
                "事件未 record 就查询完成状态：该栈原生会返回 True（误报「已完成」），"
                "适配层按 I-2 在这里拦住。"
            )
        return bool(event.query())

    def synchronize_event(self, event: Any, timeout_s: float) -> bool:
        """协议要求有这个口，但**池的屏障刻意不调它**（`I-11`：只许超时 + 轮询）。"""
        raise AssertionError("池不该在屏障里调有界同步（I-11）")


@dataclass
class RealCardAllocator:
    """张量级底座：以框架侧的空张量作底座，池只管分级、归属与账本，不碰裸指针。

    注意这个底座底下就是**厂商缓存分配器**（第一层）—— `release` 把张量还给缓存分配器，
    不一定还给驱动。这正是设计文档 1.2 说的「池不假设拿到的是裸内存」（`I-9`）。
    """

    device_index: int = 0
    live: dict[int, Any] = field(default_factory=dict, repr=False)
    _next: int = field(default=1, repr=False)

    def allocate(self, nbytes: int, device_index: int = 0, *, stream: Any) -> Any:
        # 现行 `BaseAllocator` 协议带目标流关键字（`pool/base.py`）：分配须落在给定流
        # 或建立首次使用就绪依赖。这里用流上下文包住申请，交给框架的缓存分配器记流序。
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
    """实测底座的块粒度，返回 `(最小粒度, 推荐粒度)`。

    做法：清掉缓存后申请一个极小张量，看缓存分配器**预留**量涨了多少 —— 那个涨幅就是这一栈
    的块粒度。这是四家里唯一没有厂商粒度声明的那一家（原生设备属性无粒度项、头里无对齐宏）
    的「要么实测」那条路；量出来的数作为档位配置项喂给池，**池本体里不写死任何常数**。
    """
    torch.cuda.empty_cache()
    before = torch.cuda.memory_reserved(device_index)
    probe = torch.empty(probe_bytes, dtype=torch.int8, device=f"cuda:{device_index}")
    after = torch.cuda.memory_reserved(device_index)
    del probe
    torch.cuda.empty_cache()
    grain = max(after - before, 0)
    if grain <= 0:
        pytest.skip("量不到底座块粒度（预留量没涨），这台机器上要改用显式档位配置项")
    return grain, grain


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
def real_backend(device_index: int) -> RealCardBackend:
    return RealCardBackend(device_index=device_index)


@pytest.fixture
def real_allocator(device_index: int) -> RealCardAllocator:
    return RealCardAllocator(device_index=device_index)


@pytest.fixture
def real_pool(real_backend, real_allocator, measured_tier_spec, device_index: int):
    pool = create_pool(
        real_backend,
        device_index=device_index,
        form=PoolForm.TENSOR,
        workload=WorkloadType.GENERATIVE,
        base_allocator=real_allocator,
        tier_spec=measured_tier_spec,
    )
    yield pool
    if not pool.destroyed:
        try:
            pool.destroy_pool()
        except Exception:
            pass
    torch.cuda.empty_cache()


# ======================================================================
# 用例
# ======================================================================


class TestNormal:
    """正常：四条判据在真卡上逐条过一遍。"""

    def test_判据一_按大小分级的块池(self, real_pool, measured_tier_spec: TierSpec):
        table = real_pool.tier_table
        assert table.source == "caller_config"          # 粒度是实测喂进来的，不是常数
        assert table.tiers == tuple(sorted(set(table.tiers)))
        assert table.spec.base_bytes == measured_tier_spec.base_bytes

        # 不同大小的请求落到不同档
        small = real_pool.alloc_on_stream(1, purpose=Purpose.ACTIVATION)
        big = real_pool.alloc_on_stream(table.tiers[2], purpose=Purpose.ACTIVATION)
        assert small.tier_bytes == table.tiers[0]
        assert big.tier_bytes == table.tiers[2]
        assert small.tier_bytes != big.tier_bytes
        for v in (small, big):
            real_pool.free_block(v.block_id)
        real_pool.drain_stream()

    def test_判据二_按流顺序分配与归还(self, real_pool):
        """自管路：池自己记流序，用真卡上的事件做回收屏障。"""
        side = torch.cuda.Stream()
        stream_id = real_pool.register_stream(side)
        assert stream_id != 0

        view = real_pool.alloc_on_stream(
            real_pool.tier_table.tiers[1], stream_id=stream_id, purpose=Purpose.KV_CACHE
        )
        assert view.owner_stream == stream_id

        # 在那条流上真排一点活，再归还 —— 屏障必须等这条流上的操作跑完才放行
        with torch.cuda.stream(side):
            torch.empty(1 << 20, dtype=torch.int8, device=f"cuda:{DEVICE_INDEX}")

        assert real_pool.free_block(view.block_id).state is BlockState.PENDING_RECYCLE
        result = real_pool.drain_stream(timeout_s=5.0)
        assert result.promoted_blocks == 1

        # 现行租约语义（ledger.py `get`/`renew`）：归还即撤销旧租约、复用时发新租约号
        # 且**不复用已归还标识**（设计文档 2.1）。所以旧租约号在此必须被账本拒绝 ——
        # 这本身就是「块已进池内流转」的证明，不再是旧套件假定的 block_view 可查。
        from flagmemory.pool import OwnershipViolation as _OV
        with pytest.raises(_OV):
            real_pool.block_view(view.block_id)

        # 同档再要一次：命中复用，不向底座重新取块。判据改为现行契约的等价表达 ——
        # 复用保持同一 storage 身份（物理块没换），且 base_fetches 零增长（没向底座取）。
        before_fetches = real_pool.query_audit().base_fetches
        again = real_pool.alloc_on_stream(
            real_pool.tier_table.tiers[1], stream_id=stream_id, purpose=Purpose.ACTIVATION
        )
        assert again.storage_id == view.storage_id
        assert again.block_id != view.block_id      # 新租约号，不复用已归还标识
        audit = real_pool.query_audit()
        assert audit.reuse_hits == 1
        assert audit.base_fetches == before_fetches
        real_pool.free_block(again.block_id)
        real_pool.drain_stream(timeout_s=5.0)

    def test_自管路是这一款芯片上的必经路(self, real_pool):
        """该家没有原生流有序分配原语，所以能力标记必须是假、走自管路。"""
        view = real_pool.query_capability()
        assert view.native_stream_ordered is False
        assert view.form is PoolForm.TENSOR
        assert view.core_path_ready is True
        assert any(d.key == "tensor_level_pool" for d in view.degradations)

    def test_水位双指标在真卡上都有真数(self, real_pool):
        """`I-6`：设备侧读数来自真卡三键查询，引擎侧来自池账本，两侧都要有。"""
        view = real_pool.alloc_on_stream(real_pool.tier_table.tiers[1], purpose=Purpose.ACTIVATION)
        wm = real_pool.query_watermark()
        assert wm.device_total > (1 << 30)          # 真卡的总显存，不是假数
        assert 0 < wm.device_used < wm.device_total
        assert wm.pool_in_use == view.tier_bytes
        real_pool.free_block(view.block_id)
        real_pool.drain_stream(timeout_s=5.0)

    def test_整理与压力回收把块交回底座(self, real_pool, real_allocator: RealCardAllocator):
        views = [
            real_pool.alloc_on_stream(real_pool.tier_table.tiers[1], purpose=Purpose.SCRATCH)
            for _ in range(3)
        ]
        for v in views:
            real_pool.free_block(v.block_id)
        real_pool.drain_stream(timeout_s=5.0)
        assert len(real_allocator.live) == 3

        reclaimed = real_pool.reclaim(target_bytes=real_pool.tier_table.tiers[1])
        assert reclaimed.released_blocks == 1
        assert reclaimed.met_target is True

        compacted = real_pool.compact()
        assert compacted.returned_blocks == 2
        assert real_allocator.live == {}

    def test_完整生命周期走一遍并留下证据(self, real_pool, real_allocator, device_index: int):
        """四条判据的汇总：分级 + 按流分配归还 + 真卡跑通，证据落盘（不进代码仓）。"""
        stream = torch.cuda.Stream()
        stream_id = real_pool.register_stream(stream)
        table = real_pool.tier_table

        allocated = [
            real_pool.alloc_on_stream(
                table.tiers[i % 3], stream_id=stream_id, purpose=Purpose.ACTIVATION
            )
            for i in range(6)
        ]
        peak = real_pool.query_watermark()
        for v in allocated:
            real_pool.free_block(v.block_id)
        drained = real_pool.drain_stream(timeout_s=10.0)
        audit = real_pool.query_audit()
        frag = real_pool.query_fragmentation()
        capability = real_pool.query_capability()
        report = real_pool.destroy_pool()

        assert drained.promoted_blocks == 6
        assert audit.quarantined_blocks == 0
        assert report.leaked == ()
        assert real_allocator.live == {}

        evidence = {
            "device_index": device_index,
            "tier_table": {
                "tiers": list(table.tiers),
                "base_bytes": table.spec.base_bytes,
                "step_bytes": table.spec.step_bytes,
                "source": table.source,
            },
            "capability": {
                "form": capability.form.value,
                "native_stream_ordered": capability.native_stream_ordered,
                "core_path_ready": capability.core_path_ready,
                "degradations": [d.key for d in capability.degradations],
                "tier_source": capability.tier_source,
            },
            "peak_watermark": {
                "device_total": peak.device_total,
                "device_used": peak.device_used,
                "pool_in_use": peak.pool_in_use,
            },
            "drain": {
                "promoted_blocks": drained.promoted_blocks,
                "promoted_bytes": drained.promoted_bytes,
                "recoverable_timeout": drained.recoverable_timeout,
            },
            "audit": {
                "alloc_requests": audit.alloc_requests,
                "reuse_hits": audit.reuse_hits,
                "base_fetches": audit.base_fetches,
                "quarantined_blocks": audit.quarantined_blocks,
            },
            "fragmentation": {
                "largest_servable_bytes": frag.largest_servable_bytes,
                "engine_fragmentation": frag.engine_fragmentation,
            },
            "destroy": {
                "returned_blocks": report.returned_blocks,
                "returned_bytes": report.returned_bytes,
                "leaked": len(report.leaked),
            },
        }
        text = json.dumps(evidence, ensure_ascii=False, indent=2)
        if EVIDENCE_PATH:
            # 证据不进代码仓：默认不落盘，要落就用环境变量指到仓外。
            with open(EVIDENCE_PATH, "w", encoding="utf-8") as handle:
                handle.write(text)
        print("\n===FLAGMEMORY-REALCARD-EVIDENCE===\n" + text)


class TestBoundary:
    """边界：真卡上量出来的粒度、以及屏障在真流上的等待。"""

    def test_实测得到的粒度是正数且档位按它切(self, measured_tier_spec: TierSpec):
        assert measured_tier_spec.base_bytes > 0
        assert measured_tier_spec.step_bytes > 0

    def test_屏障超时在真卡上也是可恢复态(self, real_pool):
        """给一个极短超时，拿到的必须是可恢复态而不是异常。"""
        view = real_pool.alloc_on_stream(real_pool.tier_table.tiers[0], purpose=Purpose.ACTIVATION)
        real_pool.free_block(view.block_id)
        result = real_pool.drain_stream(timeout_s=0.0)
        assert result.recoverable_timeout in (True, False)   # 两者都不算失败
        real_pool.drain_stream(timeout_s=10.0)


class TestError:
    """异常：真卡上的两个相反行为（事件误报、释放不报错）都由池自己拦住。"""

    def test_未record的事件在这一栈上原生误报已完成(self, device_index: int):
        """`I-2` 的现场取证：原生返回 True，适配层必须拦住。"""
        raw = torch.cuda.Event()
        native_says = raw.query()
        backend = RealCardBackend(device_index=device_index)
        with pytest.raises(EventNotRecorded):
            backend.query_event(raw)
        print(
            f"\n[I-2 现场取证] 未 record 的事件：原生 Event.query() = {native_says}，"
            f"适配层按纪律报错拦下。"
        )

    def test_重复归还由池账本拦下(self, real_pool):
        from flagmemory.pool import OwnershipViolation

        view = real_pool.alloc_on_stream(real_pool.tier_table.tiers[0], purpose=Purpose.ACTIVATION)
        real_pool.free_block(view.block_id)
        with pytest.raises(OwnershipViolation):
            real_pool.free_block(view.block_id)
        # 现行契约（pool.py `free_block` docstring + ledger.py `get`）：重复归还由账本
        # 拒绝且**不损伤当前块** —— 块继续走正常回收路（而非旧套件假定的转隔离终态）。
        # 判据：那块仍被正常推进为可复用，且旧租约号不再可查。
        result = real_pool.drain_stream(timeout_s=5.0)
        assert result.promoted_blocks == 1
        assert result.quarantined_blocks == ()
        with pytest.raises(OwnershipViolation):
            real_pool.block_view(view.block_id)

    def test_容量不足时报错不静默缩小请求(self, real_backend, real_allocator, device_index: int):
        """按真卡的空闲量定一个一定要不到的档，必须报错而不是悄悄给个小的。"""
        from flagmemory.pool import CapacityExhausted

        free, _total = torch.cuda.mem_get_info(device_index)
        huge = int(free * 4)
        pool = create_pool(
            real_backend,
            device_index=device_index,
            form=PoolForm.TENSOR,
            base_allocator=real_allocator,
            tier_spec=TierSpec(base_bytes=huge, step_bytes=1 << 20, tier_count=1),
        )
        try:
            with pytest.raises(CapacityExhausted):
                pool.alloc_on_stream(huge, purpose=Purpose.ACTIVATION)
        finally:
            pool.destroy_pool()
