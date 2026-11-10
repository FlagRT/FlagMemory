"""分层存储管理器本体：放置决策、触发判读、抖动抑制、级二账本、搬运审计。

语义全部取自 `docs/design/分层存储概念与设计基础.md`：
- 放置决策表 R1..R14 与五条判据取数口径 = 该文档第二节；
- 触发条件 E1..E4 / I1..I3、抖动抑制四件、换出六步、换入五步 = 第三节；
- 不变量 T-1..T-11 = 该文档第五节（唯一登记处，命名空间 = 分层存储子系统内部）；
- 本模块新增的 10 条操作契约 = 该文档第六节第 1 小节；依赖池的操作 = 6.2，一条不改。

硬约束（T-5）：本模块**不 import 推理框架**、**不 import `flagmemory.tiering.offload`
的任何实现**——后端只经下面的 `TierBackend` 协议接入；芯片差异只许出现在协议声明里，
本体代码不出现「如果是某家芯片」的分支。向下只依赖池的 15 条对外接口，不读池内部结构。

增量的边界（真卡改造留给后续增量）：
- 搬运后端用协议对象接入；本仓测试用主机内存字典模拟级二（`tests/tiering/tier_fakes.py`），
  三套真实卸载适配层到本接口面的改造未做（设计第八节第 5 条）；
- 级三整层未实现，只在决策表（R13）、水位与审计里占位（设计第八节第 2 条）；
  本增量把占位接到可操作面：`apply_r13_disposition` 兑现「级三未实现 ⇒ 按级二
  淘汰策略丢弃并计入审计」的处置口，能力面给 `tier3_backend_registered` 留位槽
  （恒 False——留位不是实现第三级后端）；
- 搬运两轨：`evict` / `promote` 保持同步完成确认（发起即轮询判定）；
  `evict_async` / `promote_async` / `poll_transfers` / `query_transfers` 是
  异步在途轨——发起即返回，搬运中的块/条目在**在途账本**（`self._transfers`）里
  有明确的「在途」状态与跟踪，完成确认（poll）后才变更归属（T-7 的异步形态）。
  在途块不得被二次调度或重复归还（重复换出/换入一律显式拒绝并计审计）；
  搬运失败分两个时点：**发起期失败**（copy_* 抛错 / 无完成凭据）与**确认期失败**
  （confirm 抛错），两者的回收语义都按 3.5「换出不落账本不还块、换入归还预留块」；
  确认超时是可恢复态不是失败（T-7 / 池的 I-11），条目留在在途表可继续轮询。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterable, Protocol, runtime_checkable

from flagmemory.pool import (
    BlockPool,
    PressureLevel,
    Purpose,
    WorkloadType,
)
from flagmemory.pool.types import CapacityExhausted, DegradationNotice, Watermark

__all__ = [
    # 异常
    "TierError",
    "TierUnavailable",
    "TierConfigError",
    "CrossStreamUndeclared",
    "TierCopyFailed",
    "PromoteCapacityExhausted",
    # 协议与数据
    "TierBackend",
    "TierBackendCapability",
    "TierEntryStats",
    "TierLevel",
    "TierPolicy",
    # 视图
    "PlacementReading",
    "PlacementVerdict",
    "EvictResult",
    "PromoteResult",
    "CooldownRejected",
    "PrefetchNotTriggered",
    "Tier2Watermark",
    "TierWatermarkView",
    "TierRegistration",
    "TierCapabilityView",
    "TierAuditView",
    "Tier2Entry",
    "TierDestroyReport",
    # 异步在途轨（本增量）
    "TransferDirection",
    "TransferState",
    "TransferTicket",
    "TransferOutcome",
    "TransferPollReport",
    "AsyncEvictReceipt",
    "AsyncPromoteReceipt",
    # 级三留位（本增量）
    "Tier3DispositionReport",
    # 本体
    "TierManager",
    "create_tier_manager",
    "destroy_tier_manager",
]


# ======================================================================
# 异常（T-4 总纪律：报错或显式降级，二者之一，不静默退化）
# ======================================================================


class TierError(RuntimeError):
    """分层存储的异常基类。"""


class TierUnavailable(TierError):
    """该后端无已注册搬运适配层，分层存储显式不可用（T-11）。"""


class TierConfigError(TierError):
    """配置非法：级二容量不大于级一（T-8）、设备侧三键不可用（T-1 启动校验）、
    滞回带约束违反、试图关掉抖动抑制四件之一（T-2）。一律报错，不静默夹取。"""


class CrossStreamUndeclared(TierError):
    """搬运流未登记就跨流搬运（T-6）。"""


class TierCopyFailed(TierError):
    """搬运过程失败（拷贝报错 / 完成确认不通过）。

    换出：条目不落级二账本、不调 free_block，块留在用（3.5）；
    换入：归还已分的块后报错。
    """


class PromoteCapacityExhausted(TierError):
    """换入时池容量不足。**不许当作未命中**（T-4）：把池返回的双指标水位带在错误里。

    属性：`watermark`（池的双指标水位）、`nbytes`（本次要的块大小）。
    """

    def __init__(self, nbytes: int, watermark: Watermark) -> None:
        self.nbytes = nbytes
        self.watermark = watermark
        super().__init__(
            f"换入失败：池容量不足，请求 {nbytes} 字节。"
            f"设备侧 已用 {watermark.device_used}/{watermark.device_total}、空闲 "
            f"{watermark.device_free}；引擎侧 在用 {watermark.pool_in_use} / 可复用 "
            f"{watermark.pool_reusable} / 待回收 {watermark.pool_pending}。"
            f"按 T-4 报错，不当作未命中，不静默改小换入量。"
        )


# ======================================================================
# 级与协议
# ======================================================================


class TierLevel(Enum):
    """三级放置目标（设计第一节 1.1；写法沿用第〇节约定，不写 L1/L2/L3）。"""

    TIER1 = "级一·设备显存"
    TIER2 = "级二·主机内存"
    TIER3 = "级三·更慢介质（未实现，结构留位）"


class TransferDirection(Enum):
    """在途搬运的方向（异步轨）。"""

    EVICT = "换出（级一→级二）"
    PROMOTE = "换入（级二→级一）"


class TransferState(Enum):
    """在途搬运的状态。生命周期：PENDING →（确认）CONFIRMED 出表落账/交块；
    PENDING →（确认期报错）FAILED 出表回收；PENDING →（问询预算耗尽）TIMED_OUT
    **留在表里**（可恢复态，不算失败，T-7 / 池的 I-11「超时不等于失败」）。"""

    PENDING = "在途"
    TIMED_OUT = "确认超时（可恢复，仍在途）"
    CONFIRMED = "完成"
    FAILED = "失败"


@dataclass(frozen=True)
class TierBackendCapability:
    """搬运后端的能力声明（register_tier_backend 的入参之一）。

    芯片差异只许出现在这里与搬运后端实现里，不出现在管理器本体（T-5 / 设计 1.3）。
    """

    #: 一次搬运粒度（字节），由适配层按后端块大小声明，不写死常数（T-9）
    transfer_granularity_bytes: int
    #: 是否锁页（声明支持锁页主机内存）
    supports_pinned: bool
    #: 是否有批量搬运原语（没有则退逐块拷贝并上报降级态，3.5）
    has_batch_copy: bool


@dataclass(frozen=True)
class TierEntryStats:
    """判据 4 / 5 的取数：条目级访问频度与重用距离。

    只能来自**引擎侧条目索引**（设备侧读数看不见条目级访问，2.1 判据 4 的口径）；
    取不到时退化为判据 1+2+3 并上报降级，**不静默把频度当 0**（T-4）。
    """

    #: 最近统计窗口内命中次数
    hits: int
    #: 最近统计窗口内未命中次数
    misses: int
    #: 最后一次被访问时的全局访问序号
    last_access_seq: int
    #: 当前全局访问序号（重用距离 = last_access_seq 与它之差）
    current_seq: int

    @property
    def reuse_distance(self) -> int:
        return self.current_seq - self.last_access_seq


@runtime_checkable
class TierBackend(Protocol):
    """搬运后端的最小协议：能力声明 + 逐块拷贝 + 完成确认 + 条目级统计。

    真实实现 = 三套 KV→Host 卸载适配层改造到本协议（后续增量，设计第八节第 5 条）；
    本增量用 `tests/tiering/fakes.py` 的主机内存字典模拟级二。
    完成确认语义对齐池的 I-2：**经事件适配器，不许裸调厂商事件查询**——
    协议方法 `confirm` 就是那条事件适配器的占位。
    """

    #: 该后端在池侧 `query_capability().backend_name` 里的标识（不是芯片型号字符串）
    backend_id: str

    def declare_capabilities(self) -> TierBackendCapability:
        """适配层**声明**的能力（register 时与实测探测比对）。"""
        ...

    def probe_capabilities(self) -> TierBackendCapability:
        """运行期**实测**的能力。声明探测不到 ⇒ 按降级登记并上报，不静默（T-4）。"""
        ...

    def copy_out(
        self, entry_id: str, block_id: int, nbytes: int, *, stream_id: int
    ) -> Any:
        """级一 → 级二逐块拷贝，返回完成凭据（ticket）。失败抛异常。"""
        ...

    def copy_in(
        self, entry_id: str, block_id: int, nbytes: int, *, stream_id: int
    ) -> Any:
        """级二 → 级一逐块拷贝，返回完成凭据（ticket）。失败抛异常。"""
        ...

    def confirm(self, ticket: Any) -> bool:
        """完成确认：True = 搬完；False = 尚未完成（可恢复，轮询再问）。"""
        ...

    def block_activity(self, block_id: int) -> TierEntryStats | None:
        """判据 4 / 5 的取数口。取不到给 None（退用其余判据并标降级）。"""
        ...

    def drop_entry(self, entry_id: str) -> None:
        """释放级二主机侧副本（换入后按策略不保留时调用）。"""
        ...


# ======================================================================
# 抖动抑制四件的可配置面（T-2）
# ======================================================================


@dataclass(frozen=True)
class TierPolicy:
    """阈值与窗口。全部运行期可改（set_tier_policy），改完生效值立即可查。

    水位一律是**占用率**（0..1），设备侧与引擎侧各一份（第〇节阈值写法约定）。
    默认值由能力层推导（3.2：分配粒度 / 搬运粒度 / 池容量与单 token KV 成本），
    `create_tier_manager` 不给 policy 时按后端搬运粒度推导 —— 不写死常数（T-9）。
    """

    #: 换出高水位（设备侧 / 引擎侧），两侧**同时**越过才具备换出条件（E1，T-1）
    h_dev: float
    h_eng: float
    #: 预取低水位（设备侧 / 引擎侧），两侧**同时**在低水位以下才允许预取（I2）
    l_dev: float
    l_eng: float
    #: 滞回带约束的 N（带宽 H−L ≥ 一次搬运粒度 × N，N 可配默认 ≥ 1）
    hysteresis_n: int = 1
    #: 冷却窗口（秒）：刚换出的条目不作预取候选；真实请求召回不受限（3.3 第 2 条）
    cooldown_s: float = 1.0
    #: 最小驻留（秒）：刚换入的条目不作换出候选；E2 强制换出可越窗但计审计（3.3 第 3 条）
    min_residency_s: float = 1.0
    #: 搬运预算闸：budget_window_s 内最多搬 budget_bytes 字节（3.3 第 4 条）
    budget_bytes: int = 256 << 20
    budget_window_s: float = 1.0
    #: 空转退避：连续 backoff_trigger_max 次挑不出候选则拉长判读间隔（有候选立即复位）
    backoff_trigger_max: int = 3
    #: 级二高水位（R13：到高位该向级三换出；级三未实现 ⇒ 丢弃并计审计）
    level2_high_watermark: float = 0.9


def _derive_default_policy(
    granularity_bytes: int, device_total: int, pool_total: int
) -> TierPolicy:
    """按 3.2 现行方案推导默认阈值：输入 = 搬运粒度（适配层声明）+ 两侧容量基数。

    推导式（保证滞回带约束天然成立）：带宽 = 粒度 × N ÷ 基数；
    高水位 = 1 − 带宽，低水位 = 高水位 − 带宽。推导结果经 `query_tier_capability`
    可查（T-9 的「生效值必须可查」）。
    """
    n = 1
    dev_base = max(device_total, granularity_bytes * (n + 2), 1)
    eng_base = max(pool_total, granularity_bytes * (n + 2), 1)
    band_dev = granularity_bytes * n / dev_base
    band_eng = granularity_bytes * n / eng_base
    h_dev = max(1.0 - band_dev, band_dev * 2)
    h_eng = max(1.0 - band_eng, band_eng * 2)
    return TierPolicy(
        h_dev=h_dev,
        h_eng=h_eng,
        l_dev=h_dev - band_dev,
        l_eng=h_eng - band_eng,
        hysteresis_n=n,
    )


def _validate_policy(
    policy: TierPolicy, granularity_bytes: int, device_total: int, pool_total: int
) -> None:
    """滞回带与四件开关校验。违反即报错，**不静默夹取**（T-2 + T-4）。"""
    if policy.hysteresis_n < 1:
        raise TierConfigError(
            f"滞回带 N 必须 ≥ 1（当前 {policy.hysteresis_n}，T-2：单阈值视为缺陷）。"
        )
    if policy.cooldown_s <= 0 or policy.min_residency_s <= 0:
        raise TierConfigError(
            "冷却窗口与最小驻留必须 > 0：抖动抑制四件默认开，禁止整项关闭（T-2）。"
        )
    if policy.budget_bytes <= 0 or policy.budget_window_s <= 0 or policy.backoff_trigger_max < 1:
        raise TierConfigError(
            "搬运预算闸与空转退避必须启用（预算 > 0、窗口 > 0、退避阈值 ≥ 1）："
            "抖动抑制四件默认开，禁止整项关闭（T-2）。"
        )
    for name in ("h_dev", "h_eng", "l_dev", "l_eng"):
        value = getattr(policy, name)
        if not 0.0 <= value <= 1.0:
            raise TierConfigError(f"水位 {name}={value} 必须在 0..1（占用率口径）。")
    if not 0.0 <= policy.level2_high_watermark <= 1.0:
        raise TierConfigError(
            f"级二高水位 level2_high_watermark={policy.level2_high_watermark} "
            f"必须在 0..1（占用率口径）。"
        )
    # T-2 滞回带：L ≤ H − 一次搬运粒度 × N（按各自容量基数换算成占用率）。
    # 基数下限 = 粒度 × (N+2)：池还空着时（pool_total 小）不把带宽算成天文数字。
    dev_base = max(device_total, granularity_bytes * (policy.hysteresis_n + 2), 1)
    eng_base = max(pool_total, granularity_bytes * (policy.hysteresis_n + 2), 1)
    band_dev = granularity_bytes * policy.hysteresis_n / dev_base
    if policy.l_dev > policy.h_dev - band_dev:
        raise TierConfigError(
            f"滞回带约束违反：l_dev={policy.l_dev} > h_dev − 粒度×N/设备总量"
            f"（= {policy.h_dev} − {band_dev}）。按 T-2/T-4 报错，不静默夹取。"
        )
    band_eng = granularity_bytes * policy.hysteresis_n / eng_base
    if policy.l_eng > policy.h_eng - band_eng:
        raise TierConfigError(
            f"滞回带约束违反：l_eng={policy.l_eng} > h_eng − 粒度×N/池总量"
            f"（= {policy.h_eng} − {band_eng}）。按 T-2/T-4 报错，不静默夹取。"
        )


# ======================================================================
# 视图（纯数据）
# ======================================================================


@dataclass(frozen=True)
class PlacementReading:
    """一次放置判定用到的判据读数（2.3：判定的依据必须可逐条复核）。"""

    purpose: Purpose | None
    #: 双指标两侧读数（T-1：单指标不得作为唯一输入，两侧都给）
    watermark: Watermark
    device_occupancy: float
    engine_occupancy: float
    #: 水位区间：高 / 中 / 低（按双指标两侧同时判读）
    zone: str
    workload: WorkloadType | None
    #: 判据 4 / 5（取不到时 stats_available=False，判据退用 1+2+3 并标降级）
    hits: int | None = None
    misses: int | None = None
    reuse_distance: int | None = None
    stats_available: bool = False


@dataclass(frozen=True)
class PlacementVerdict:
    """classify_placement 的产出：只判定，不搬运（2.3）。"""

    target: TierLevel
    rule: str
    reading: PlacementReading
    note: str = ""
    degraded: tuple[DegradationNotice, ...] = ()


@dataclass(frozen=True)
class EvictResult:
    """evict 的产出。`met_target` 如实标注，部分换出不得报成功（T-4）。"""

    evicted_bytes: int
    evicted_entries: tuple[str, ...]
    met_target: bool
    #: 未达目标 / 空转的原因（人话，进日志）
    reason: str
    #: 候选排序依据（每个换出条目命中 R9 / R10 / R7 哪条）
    ordering_basis: tuple[str, ...] = ()
    #: 越过最小驻留窗的换出条数（E2 可越窗但要计审计，3.3 第 3 条）
    bypassed_residency: int = 0


@dataclass(frozen=True)
class PromoteResult:
    """promote 成功的产出：完成确认之后才交块（T-7）。"""

    entry_id: str
    block_id: int
    base_ref: Any
    nbytes: int
    kept_tier2_copy: bool


@dataclass(frozen=True)
class CooldownRejected:
    """预取被冷却窗拒绝（I2）。真实请求召回（I1）不受此限（T-2）。"""

    entry_id: str
    remaining_s: float


@dataclass(frozen=True)
class PrefetchNotTriggered:
    """预取未触发：双指标低水位未同时满足（I2 条件，T-1）。"""

    entry_id: str
    device_occupancy: float
    engine_occupancy: float


PromoteOutcome = PromoteResult | CooldownRejected | PrefetchNotTriggered


@dataclass(frozen=True)
class Tier2Entry:
    """级二账本的一条（主机侧块表 + 条目索引；账本先行是正确性要求，T-7）。"""

    entry_id: str
    #: 来源块（换出前的级一块标识；换入成功后按策略保留或释放）
    block_id: int
    nbytes: int
    evicted_at: float
    #: 换入后是否保留了级二副本
    replica_kept: bool = False


@dataclass(frozen=True)
class Tier2Watermark:
    """级二水位（取自管理器自己的主机侧账本）。"""

    capacity_bytes: int
    used_bytes: int
    entry_count: int

    @property
    def occupancy(self) -> float:
        return self.used_bytes / self.capacity_bytes if self.capacity_bytes else 0.0


@dataclass(frozen=True)
class TierWatermarkView:
    """query_tier_watermark 的产出：级一双指标（两侧都给）+ 级二账本 + 级三未实现。"""

    level1: Watermark
    level2: Tier2Watermark
    level3: str = "未实现（级三结构留位，设计第八节第 2 条）"


@dataclass(frozen=True)
class TierRegistration:
    """register_tier_backend 的产出：注册结论 + 实测能力与降级项（不静默，T-4）。"""

    backend_id: str
    declared: TierBackendCapability
    probed: TierBackendCapability
    degradations: tuple[DegradationNotice, ...]


@dataclass(frozen=True)
class TierCapabilityView:
    """query_tier_capability 的产出。该口恒可用，是降级可观测性的基础。"""

    available: bool
    unavailable_reason: str
    backend_id: str
    declared: TierBackendCapability | None
    probed: TierBackendCapability | None
    degradations: tuple[DegradationNotice, ...]
    #: 生效的阈值与窗口（T-9：推导出的实际生效值必须可查）
    effective_policy: TierPolicy | None
    #: 生效的级二容量与搬运粒度
    level2_capacity_bytes: int
    transfer_granularity_bytes: int
    #: 空转退避现状：连续空转轮数（有候选立即复位）
    backoff_rounds: int
    #: 级三后端是否已注册（留位槽，恒 False——设计第八节第 2 条：结构留位，不承诺实现）
    tier3_backend_registered: bool = False


@dataclass(frozen=True)
class TierAuditView:
    """query_tier_audit 的产出：抖动抑制生效本身必须可观测（3.3 末段）。"""

    evict_calls: int
    evicted_bytes: int
    evicted_entries: int
    promote_calls: int
    promoted_bytes: int
    promoted_entries: int
    # —— 抖动抑制四件的触发数 ——
    cooldown_rejected_prefetch: int
    min_residency_deferred: int
    budget_gate_deferred: int
    backoff_deferrals: int
    # —— 其余审计 ——
    bypass_window_evicts: int
    unmet_target_evicts: int
    copy_failures: int
    cross_stream_declarations: int
    #: 级三缺失丢弃（R13；级三未实现，本增量把处置口接到这个计数位）
    tier3_missing_drops: int
    #: 在途块被再次调度（换出/换入）而被拒的次数（在途块不得二次调度）
    inflight_reschedule_rejected: int


@dataclass(frozen=True)
class TierDestroyReport:
    """destroy 的产出：级二残留条目与未回级一的块清单；在途搬运不静默丢弃。"""

    residual_entries: tuple[Tier2Entry, ...]
    in_flight: tuple[str, ...]
    timed_out: bool


# ----------------------------------------------------------------------
# 异步在途轨的视图（本增量）：发起即返回的回执 + 在途账本的可查状态 + 轮询产出
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TransferTicket:
    """在途账本的一条（query_transfers 的产出）：搬运中的块/条目的明确「在途」状态与跟踪。"""

    transfer_id: str
    direction: TransferDirection
    entry_id: str
    block_id: int
    nbytes: int
    started_at: float
    polls: int
    state: TransferState
    error: str | None


@dataclass(frozen=True)
class AsyncEvictReceipt:
    """evict_async 的产出：**发起即返回**。

    `scheduled_*` 是「已发起在途」的量，不是已换出量——完成确认（poll_transfers）
    之前，条目不落级二账本、块不归还池（T-7）；实际换出量看审计与水位。
    未达目标如实标注（T-4）。"""

    scheduled_bytes: int
    scheduled_entries: tuple[str, ...]
    transfers: tuple[TransferTicket, ...]
    met_target: bool
    reason: str
    ordering_basis: tuple[str, ...] = ()
    bypassed_residency: int = 0


@dataclass(frozen=True)
class AsyncPromoteReceipt:
    """promote_async 的产出：块已向池预留但**未确认完成**——块标识与底座引用在
    完成确认时才交付（poll_transfers 的 TransferOutcome.promoted，T-7 第 5 步：
    未确认完成就交出去 = 交出一块内容不全的显存）。"""

    entry_id: str
    nbytes: int
    transfer_id: str


@dataclass(frozen=True)
class TransferOutcome:
    """poll_transfers 的一条：一次非阻塞轮询里对一笔在途搬运的判定。"""

    transfer_id: str
    direction: TransferDirection
    entry_id: str
    block_id: int
    nbytes: int
    state: TransferState
    polls: int = 0
    #: 换入完成时交付的块（direction = PROMOTE 且 state = CONFIRMED 才有）
    promoted: PromoteResult | None = None
    error: str | None = None


@dataclass(frozen=True)
class TransferPollReport:
    """poll_transfers 的产出：本次轮询的全部判定 + 仍在途清单（可观测面）。"""

    outcomes: tuple[TransferOutcome, ...]
    still_in_flight: tuple[TransferTicket, ...]

    @property
    def confirmed(self) -> tuple[TransferOutcome, ...]:
        return tuple(o for o in self.outcomes if o.state is TransferState.CONFIRMED)

    @property
    def failed(self) -> tuple[TransferOutcome, ...]:
        return tuple(o for o in self.outcomes if o.state is TransferState.FAILED)


# ----------------------------------------------------------------------
# 级三留位的处置视图（本增量；设计第八节第 2 条：只定位置，不承诺实现）
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Tier3DispositionReport:
    """apply_r13_disposition 的产出：R13 处置（级二高水位 ⇒ 本应换出到级三）。

    级三未实现 ⇒ 按级二淘汰策略（现状 LRU，evicted_at 最旧先丢——实现可调参数，
    不是冻结口径，设计第八节第 6 条）丢弃并计入审计（tier3_missing_drops）。
    `backend_registered` 恒 False：这是**留位**——槽位在、后端未实现。"""

    #: 处置目标级别（恒 TIER3，留位）
    target: TierLevel
    #: 级三后端是否已注册（恒 False；将来实现级三后端时此槽位即为接线点）
    backend_registered: bool
    dropped_entries: tuple[str, ...]
    dropped_bytes: int
    reason: str


# ======================================================================
# 本体
# ======================================================================


#: 完成确认的轮询上限（同步完成确认的实现细节；屏障只许超时+轮询，不许阻塞等待，T-7）
_CONFIRM_POLLS = 8
_CONFIRM_INTERVAL_S = 0.0

#: 异步在途轨的问询预算：同一笔搬运被问询满这个次数仍未确认 ⇒ 状态转 TIMED_OUT
#: （可恢复态：**留在在途表**可继续轮询，不算失败——T-7 / 池的 I-11「超时不等于失败」）。
#: 这是主机侧机制预算，不是芯片相关阈值（T-9 管的是搬运粒度与水位默认值）。
_PENDING_TIMEOUT_POLLS = 8

#: destroy 排空在途时的轮询间隔（秒）。只许超时+轮询，不许无限阻塞（T-7）。
_DRAIN_POLL_INTERVAL_S = 0.001


@dataclass
class _TransferRecord:
    """在途账本的内部记录（可变；对外只经 `TransferTicket` 视图暴露）。"""

    transfer_id: str
    direction: TransferDirection
    entry_id: str
    block_id: int
    nbytes: int
    ticket: Any
    started_at: float
    #: 发起搬运用的搬运流（finalize 时 drain 用同一条流）
    stream_id: int
    #: 候选排序依据（换出方向；R9 / R7 / R10）
    rule: str
    #: 越过最小驻留窗的换出（E2 可越窗但要计审计）
    bypassed_residency: int
    #: 换入方向：完成后是否保留级二副本
    keep_tier2_copy: bool
    #: 换入方向：完成时交付的底座引用（promote_async 发起时从池租约取得）
    base_ref: Any = None
    polls: int = 0
    state: TransferState = TransferState.PENDING
    error: str | None = None

    def view(self) -> TransferTicket:
        return TransferTicket(
            transfer_id=self.transfer_id,
            direction=self.direction,
            entry_id=self.entry_id,
            block_id=self.block_id,
            nbytes=self.nbytes,
            started_at=self.started_at,
            polls=self.polls,
            state=self.state,
            error=self.error,
        )

    def outcome(self, state: TransferState) -> TransferOutcome:
        return TransferOutcome(
            transfer_id=self.transfer_id,
            direction=self.direction,
            entry_id=self.entry_id,
            block_id=self.block_id,
            nbytes=self.nbytes,
            state=state,
            polls=self.polls,
            error=self.error,
        )

#: 审计计数器的字段集合（避免手写 dict 漂移）
_AUDIT_FIELDS = (
    "evict_calls",
    "evicted_bytes",
    "evicted_entries",
    "promote_calls",
    "promoted_bytes",
    "promoted_entries",
    "cooldown_rejected_prefetch",
    "min_residency_deferred",
    "budget_gate_deferred",
    "backoff_deferrals",
    "bypass_window_evicts",
    "unmet_target_evicts",
    "copy_failures",
    "cross_stream_declarations",
    "tier3_missing_drops",
    "inflight_reschedule_rejected",
)


class TierManager:
    """分层存储管理器。10 条操作的 2..10 是本类方法，第 1 条是 `create_tier_manager`。"""

    def __init__(
        self,
        pool: BlockPool,
        backend: TierBackend | None,
        *,
        level2_capacity_bytes: int,
        policy: TierPolicy | None = None,
        transfer_stream: Any | None = None,
    ) -> None:
        # T-11：无已注册适配层 ⇒ 报错不启，不伪装可用
        if backend is None:
            raise TierUnavailable(
                "该后端无已注册搬运适配层，分层存储显式不可用（T-11）；"
                "池本体照常工作（退一级），换入换出一律报错。"
            )
        # T-1（启动校验）：设备侧三键不可用 ⇒ 报错不启，不用引擎侧单边数充抵
        watermark = pool.query_watermark()
        if watermark.device_total <= 0:
            raise TierConfigError(
                "池的设备侧三键（总量/已用/空闲）不可用，报错不启分层存储（T-1）；"
                "不用引擎侧单边读数充抵。"
            )
        # T-8：级二容量必须大于级一池容量，否则报错不启，不降级成一级
        if level2_capacity_bytes <= 0:
            raise TierConfigError(f"级二容量必须为正（收到 {level2_capacity_bytes}）。")
        if level2_capacity_bytes <= watermark.device_total:
            raise TierConfigError(
                f"级二容量（{level2_capacity_bytes}）不大于级一池容量"
                f"（{watermark.device_total}）：不构成真两级缓存，报错不启（T-8），"
                f"不降级成一级。"
            )
        registration = _register(backend, pool)
        pool_backend = pool.query_capability().backend_name
        if registration.backend_id != pool_backend:
            raise TierError(
                f"后端标识 {registration.backend_id!r} 与池的后端 {pool_backend!r} "
                f"不一致，报错不启（6.1 第 3 条）。"
            )
        self._pool = pool
        self._backend = backend
        self._backend_id = registration.backend_id
        self._declared = registration.declared
        self._probed = registration.probed
        self._backend_degradations = registration.degradations
        self._level2_capacity = level2_capacity_bytes
        granularity = self._probed.transfer_granularity_bytes
        if granularity <= 0:
            raise TierConfigError(
                f"搬运后端声明的一次搬运粒度必须为正（收到 {granularity}，T-9）。"
            )
        self._policy = policy or _derive_default_policy(
            granularity, watermark.device_total, watermark.pool_total
        )
        _validate_policy(
            self._policy, granularity, watermark.device_total, watermark.pool_total
        )
        # 级二账本（主机侧块表 + 条目索引；账本先行，T-7）
        self._entries: dict[str, Tier2Entry] = {}
        # 换入时间戳（最小驻留判据用）
        self._promoted_at: dict[int, float] = {}
        # 搬运流：默认用池的默认流（单流场景）；给了 transfer_stream 则登记并强持有（T-6 第 1 步）
        self._known_streams: set[int] = set()
        self._copy_stream_id = 0
        if transfer_stream is not None:
            self._copy_stream_id = self.attach_stream(transfer_stream)
        # 预算闸的滑动窗口与空转退避
        self._budget_events: deque[tuple[float, int]] = deque()
        self._idle_rounds = 0
        # 在途账本（异步轨）：transfer_id -> 记录。搬运中的块/条目在这里有明确
        # 「在途」状态；完成确认后才变更归属（T-7 的异步形态）。
        self._transfers: dict[str, _TransferRecord] = {}
        self._transfer_seq = 0
        self._audit = {name: 0 for name in _AUDIT_FIELDS}
        self._destroyed = False

    # ------------------------------------------------------------------
    # 辅助（不在那 10 条里）
    # ------------------------------------------------------------------

    @property
    def destroyed(self) -> bool:
        return self._destroyed

    @property
    def _in_flight(self) -> set[str]:
        """在途搬运的条目号集合（destroy 的报错清单等粗粒度口径；详态走 query_transfers）。"""
        return {record.entry_id for record in self._transfers.values()}

    def attach_stream(self, stream: Any) -> int:
        """把一条搬运流经池 register_stream 登记并由池强持有（T-6 / 3.4 换出第 1 步）。"""
        self._require_open()
        stream_id = self._pool.register_stream(stream)
        self._known_streams.add(stream_id)
        return stream_id

    def _require_open(self) -> None:
        if self._destroyed:
            raise TierError("分层存储管理器已销毁。")

    def _require_backend(self) -> None:
        self._require_open()
        if self._backend is None:  # pragma: no cover - 构造期已拦，防御
            raise TierUnavailable("该后端无已注册搬运适配层（T-11）。")

    def _zone(self, watermark: Watermark) -> tuple[str, float, float]:
        """双指标水位区间：高 = 两侧同时越高水位；低 = 两侧同时低于低水位（T-1）。"""
        device_occ = watermark.device_used / watermark.device_total
        engine_occ = (
            watermark.pool_in_use / watermark.pool_total
            if watermark.pool_total
            else 1.0
        )
        if device_occ >= self._policy.h_dev and engine_occ >= self._policy.h_eng:
            return "高", device_occ, engine_occ
        if device_occ <= self._policy.l_dev and engine_occ <= self._policy.l_eng:
            return "低", device_occ, engine_occ
        return "中", device_occ, engine_occ

    def _budget_allows(self, nbytes: int) -> bool:
        """搬运预算闸：窗口内累计字节数有上限（3.3 第 4 条）。"""
        now = time.monotonic()
        while self._budget_events and now - self._budget_events[0][0] > self._policy.budget_window_s:
            self._budget_events.popleft()
        used = sum(n for _, n in self._budget_events)
        return used + nbytes <= self._policy.budget_bytes

    def _budget_charge(self, nbytes: int) -> None:
        self._budget_events.append((time.monotonic(), nbytes))

    def _confirm(self, ticket: Any) -> bool:
        """完成确认：轮询直至确认或放弃（超时 + 轮询，不阻塞等待，T-7）。"""
        for _ in range(_CONFIRM_POLLS):
            if self._backend.confirm(ticket):
                return True
            time.sleep(_CONFIRM_INTERVAL_S)
        return False

    def _declare_cross_stream(self, block_id: int, block_owner: int | None) -> None:
        """3.4 第 2 步：搬运流与分配流不同 ⇒ 先声明跨流使用（T-6）。"""
        if block_owner is not None and self._copy_stream_id != block_owner:
            self._pool.declare_cross_stream_use(block_id, self._copy_stream_id)
            self._audit["cross_stream_declarations"] += 1

    # —— 换出/换入的公共判读件（同步轨与异步轨共用，语义一字不差）——

    def _evict_gate(
        self, *, forced: bool, watermark: Watermark
    ) -> EvictResult | None:
        """E1 双指标触发判读。返回 None = 通过；返回 EvictResult = 不触发（含空转退避计数）。"""
        if not forced:
            zone, _, _ = self._zone(watermark)
            if zone != "高":
                self._idle_rounds += 1
                if self._idle_rounds >= self._policy.backoff_trigger_max:
                    self._audit["backoff_deferrals"] += 1
                return EvictResult(
                    evicted_bytes=0,
                    evicted_entries=(),
                    met_target=False,
                    reason=f"双指标水位未同时到高水位（当前区间 {zone}，T-1），不换出",
                )
        return None

    def _collect_candidates(
        self, candidates: Iterable[int]
    ) -> tuple[list[tuple[int, int]], int, bool, int]:
        """候选收集与 R9→R10 排序（2.2 / E1 条件③）。

        返回 (ordered[(排序键, block_id)], R1..R5 跳过数, 判据 4/5 缺失标记, 在途跳过数)。
        在途块不得被二次调度：有在途搬运记录的块跳过并计审计（异步轨的核心不变量）。
        副本保留产生的条目号冲突（b<block_id> 已在级二账本）同样跳过——宁可拒绝。"""
        ordered: list[tuple[int, int]] = []
        skipped_labels = 0
        stats_missing = False
        inflight_skipped = 0
        active_blocks = {r.block_id for r in self._transfers.values()}
        for block_id in candidates:
            view = self._pool.block_view(block_id)
            if view.purpose != Purpose.KV_CACHE:
                skipped_labels += 1  # R1..R5：不进分层（T-3）
                continue
            if block_id in active_blocks or f"b{block_id}" in self._entries:
                # 在途块不得被二次调度；副本保留中/已在级二的条目号不覆盖
                inflight_skipped += 1
                self._audit["inflight_reschedule_rejected"] += 1
                continue
            stats = self._backend.block_activity(block_id)
            if stats is None:
                stats_missing = True
                ordered.append((1, block_id))  # 判据 4/5 缺失：中性序（降级可查）
            elif stats.hits == 0 and stats.reuse_distance > 0:
                ordered.append((0, block_id))  # R9：冷且久未用，最前
            else:
                ordered.append((2, block_id))  # R10/R12：热条目最后
        ordered.sort(key=lambda item: item[0])
        return ordered, skipped_labels, stats_missing, inflight_skipped

    def _level2_committed(self) -> int:
        """级二承诺量 = 已确认条目 + 换出在途（确认后即落账本）。容量判读用它（保守：
        在途换出按将占用计，避免排空时超容量——T-4 不静默超发）。"""
        return self._level2_used() + sum(
            r.nbytes
            for r in self._transfers.values()
            if r.direction is TransferDirection.EVICT
        )

    def _require_promotable(self, entry_id: str, *, operation: str) -> Tier2Entry:
        """换入的条目查账（同步/异步共用）。在途条目显式拒绝，不静默当未命中（T-4）。"""
        entry = self._entries.get(entry_id)
        active = [r for r in self._transfers.values() if r.entry_id == entry_id]
        if entry is None:
            out_ids = [
                r.transfer_id
                for r in active
                if r.direction is TransferDirection.EVICT
            ]
            if out_ids:
                raise TierError(
                    f"条目 {entry_id!r} 正在换出在途（{out_ids}），完成确认前不在级二"
                    f"账本，{operation} 拒绝（不静默当未命中，T-4 / T-7）。"
                )
            raise TierError(
                f"条目不在级二：{entry_id!r}（{operation} 不静默当未命中，T-4）。"
            )
        in_ids = [
            r.transfer_id
            for r in active
            if r.direction is TransferDirection.PROMOTE
        ]
        if in_ids:
            raise TierError(
                f"条目 {entry_id!r} 已有换入在途（{in_ids}），重复调度被拒。"
            )
        return entry

    def _promote_prefetch_gate(
        self, entry: Tier2Entry, watermark: Watermark
    ) -> CooldownRejected | PrefetchNotTriggered | None:
        """I2 预取门禁（同步/异步共用）：双指标低水位 + 冷却窗 + 预算闸（T-1 / T-2）。"""
        zone, dev_occ, eng_occ = self._zone(watermark)
        if zone != "低":
            return PrefetchNotTriggered(
                entry_id=entry.entry_id,
                device_occupancy=dev_occ,
                engine_occupancy=eng_occ,
            )
        # 冷却窗：刚换出的条目不作预取候选；真实请求召回（I1）不受此限（T-2）
        cooldown = time.monotonic() - entry.evicted_at
        if cooldown < self._policy.cooldown_s:
            self._audit["cooldown_rejected_prefetch"] += 1
            return CooldownRejected(
                entry_id=entry.entry_id,
                remaining_s=self._policy.cooldown_s - cooldown,
            )
        if not self._budget_allows(entry.nbytes):
            self._audit["budget_gate_deferred"] += 1
            return PrefetchNotTriggered(
                entry_id=entry.entry_id,
                device_occupancy=dev_occ,
                engine_occupancy=eng_occ,
            )
        return None

    # ------------------------------------------------------------------
    # 操作 4：classify_placement（只判不动，2.3）
    # ------------------------------------------------------------------

    def classify_placement(
        self, *, block_id: int | None = None, entry_id: str | None = None
    ) -> PlacementVerdict:
        self._require_open()
        if (block_id is None) == (entry_id is None):
            raise TierError("classify_placement 需要 block_id 或 entry_id 二选一。")
        watermark = self._pool.query_watermark()
        if watermark.device_total <= 0:
            raise TierConfigError("池的设备侧三键不可用（T-1），无法做放置判定。")
        if entry_id is not None:
            return self._classify_entry(entry_id, watermark)
        assert block_id is not None
        return self._classify_block(block_id, watermark)

    def _classify_block(self, block_id: int, watermark: Watermark) -> PlacementVerdict:
        view = self._pool.block_view(block_id)
        zone, dev_occ, eng_occ = self._zone(watermark)
        stats = self._backend.block_activity(block_id)
        degraded: tuple[DegradationNotice, ...] = ()
        hits = misses = distance = None
        if stats is None:
            degraded = (
                DegradationNotice(
                    key="tier.stats-unavailable",
                    action="判据 4/5（访问频度/重用距离）取不到，退用判据 1+2+3 判定",
                    cost="候选排序退化为按档位大小序，抖动抑制强度下降",
                ),
            )
        else:
            hits, misses, distance = stats.hits, stats.misses, stats.reuse_distance
        reading = PlacementReading(
            purpose=view.purpose,
            watermark=watermark,
            device_occupancy=dev_occ,
            engine_occupancy=eng_occ,
            zone=zone,
            workload=self._pool.workload,
            hits=hits,
            misses=misses,
            reuse_distance=distance,
            stats_available=stats is not None,
        )
        # —— R1..R5：永不下放清单（T-3）——
        never = {
            Purpose.WEIGHT: ("R1", "权重永不下放（I-12 / T-3）"),
            Purpose.COMM_BUFFER: ("R2", "通信缓冲永不下放（I-12 / T-3）"),
            Purpose.UNDECLARED: ("R3", "用途标签未声明，按最保守处理永不下放（I-12 / T-3）"),
            Purpose.SCRATCH: ("R4", "临时工作区生命周期短于搬运往返，不进分层"),
            Purpose.ACTIVATION: ("R5", "激活是池块复用的主战场，不进分层"),
        }
        if view.purpose in never:
            rule, note = never[view.purpose]
            return PlacementVerdict(
                target=TierLevel.TIER1, rule=rule, reading=reading, note=note
            )
        # —— KV 缓存 ——
        if self._backend is None:  # pragma: no cover - 构造期已拦
            return PlacementVerdict(
                target=TierLevel.TIER1,
                rule="R6",
                reading=reading,
                note="本后端无搬运适配层，显式不可用（T-11）",
            )
        if zone != "高":
            return PlacementVerdict(
                target=TierLevel.TIER1,
                rule="R8",
                reading=reading,
                note="双指标水位未同时到高水位，搬运是净成本",
            )
        # 高水位：按判据 4/5 与负载类型分叉
        cold = stats is not None and stats.hits == 0 and stats.reuse_distance > 0
        hot = stats is not None and stats.hits > 0 and stats.reuse_distance == 0
        if stats is None:
            return PlacementVerdict(
                target=TierLevel.TIER2,
                rule="R7",
                reading=reading,
                note="判据 4/5 不可用，按普通换出（降级已标）",
                degraded=degraded,
            )
        if cold:
            return PlacementVerdict(
                target=TierLevel.TIER2,
                rule="R9",
                reading=reading,
                note="低频且重用距离远，候选序最前",
            )
        if hot and self._pool.workload == WorkloadType.EMBEDDING:
            return PlacementVerdict(
                target=TierLevel.TIER1,
                rule="R11",
                reading=reading,
                note="向量化负载：压第二层预分配余量优于搬运（I-7）",
            )
        if hot:
            return PlacementVerdict(
                target=TierLevel.TIER2,
                rule="R12" if self._pool.workload == WorkloadType.GENERATIVE else "R10",
                reading=reading,
                note=(
                    "生成式负载：按 R9→R10 序换出，不压预分配（I-7）"
                    if self._pool.workload == WorkloadType.GENERATIVE
                    else "热条目最后才换出（候选序最后）"
                ),
            )
        return PlacementVerdict(
            target=TierLevel.TIER2, rule="R7", reading=reading, note="普通换出候选"
        )

    def _classify_entry(self, entry_id: str, watermark: Watermark) -> PlacementVerdict:
        entry = self._entries.get(entry_id)
        if entry is None:
            raise TierError(f"级二没有条目 {entry_id!r}。")
        reading = PlacementReading(
            purpose=Purpose.KV_CACHE,
            watermark=watermark,
            device_occupancy=watermark.device_used / watermark.device_total,
            engine_occupancy=(
                watermark.pool_in_use / watermark.pool_total
                if watermark.pool_total
                else 1.0
            ),
            zone="中",
            workload=self._pool.workload,
        )
        level2 = self._level2_view()
        if level2.occupancy >= self._policy.level2_high_watermark:
            return PlacementVerdict(
                target=TierLevel.TIER3,
                rule="R13",
                reading=reading,
                note=f"级二水位到高位（{level2.occupancy:.2f}），应换出到级三；"
                f"级三未实现 ⇒ 按级二淘汰策略处置并计审计（设计第八节第 2 条）",
            )
        return PlacementVerdict(
            target=TierLevel.TIER1,
            rule="R14",
            reading=reading,
            note="引擎侧报命中，换入到级一（I1）",
        )

    # ------------------------------------------------------------------
    # 操作 5：evict（换出六步，3.4；顺序不可换，T-7）
    # ------------------------------------------------------------------

    def evict(
        self,
        candidates: Iterable[int],
        *,
        target_bytes: int | None = None,
        pressure: PressureLevel | None = None,
        stream_id: int | None = None,
        allow_bypass_min_residency: bool = False,
    ) -> EvictResult:
        self._require_backend()
        self._audit["evict_calls"] += 1
        if target_bytes is None and pressure is None:
            raise TierError("evict 需要 target_bytes 或 pressure 至少其一（E2 口径）。")
        if stream_id is not None and stream_id not in self._known_streams:
            # T-6：搬运流未登记就跨流搬运属调用方错误
            raise CrossStreamUndeclared(
                f"搬运流 {stream_id} 未登记（T-6）：请先 attach_stream / register_stream。"
            )
        if stream_id is not None:
            self._copy_stream_id = stream_id
        forced = pressure is not None
        watermark = self._pool.query_watermark()
        if watermark.device_total <= 0:
            raise TierConfigError("池的设备侧三键不可用（T-1），无法做换出判读。")
        # E1 触发判读：双指标两侧同时越高水位才触发（T-1）；E2（压力回收）不查水位
        gate = self._evict_gate(forced=forced, watermark=watermark)
        if gate is not None:
            return gate
        # 候选收集：R1..R5 的标签永不出现在候选里；按 R9→R10 排序；在途块拒入
        ordered, skipped_labels, stats_missing, inflight_skipped = (
            self._collect_candidates(candidates)
        )
        if not ordered:
            self._idle_rounds += 1
            if self._idle_rounds >= self._policy.backoff_trigger_max:
                self._audit["backoff_deferrals"] += 1
            reason = "无可换出候选"
            if skipped_labels:
                reason += f"（{skipped_labels} 个候选属 R1..R5 永不下放标签，T-3）"
            if inflight_skipped:
                reason += f"（{inflight_skipped} 个候选正在途，二次调度被拒）"
            return EvictResult(
                evicted_bytes=0,
                evicted_entries=(),
                met_target=False,
                reason=reason,
            )
        self._idle_rounds = 0  # 有候选，空转退避立即复位（3.3 第 4 条）
        evicted_bytes = 0
        evicted: list[str] = []
        basis: list[str] = []
        bypassed = 0
        reason = ""
        now = time.monotonic()
        for rank, block_id in ordered:
            if target_bytes is not None and evicted_bytes >= target_bytes:
                break
            view = self._pool.block_view(block_id)
            nbytes = view.tier_bytes
            # 级二容量不足 ⇒ 停，如实标注未达目标（T-4）。
            # 判读用承诺量（已确认 + 换出在途），异步在途按将占用计（保守）。
            if self._level2_committed() + nbytes > self._level2_capacity:
                reason = "级二容量不足，停止换出"
                break
            # 预算闸（3.3 第 4 条）
            if not self._budget_allows(nbytes):
                self._audit["budget_gate_deferred"] += 1
                reason = "搬运预算闸：窗口内字节数到上限"
                break
            # 最小驻留（E1 条件⑤ / 3.3 第 3 条）：E2 强制可越窗但计审计
            promoted_at = self._promoted_at.get(block_id)
            if (
                promoted_at is not None
                and now - promoted_at < self._policy.min_residency_s
            ):
                if forced and allow_bypass_min_residency:
                    bypassed += 1
                else:
                    self._audit["min_residency_deferred"] += 1
                    continue
            rule = {0: "R9", 1: "R7", 2: "R10"}[rank]
            entry_id = f"b{block_id}"
            # —— 换出六步（3.4；顺序不可换，T-7）——
            # 步 2：跨流声明（搬运流与分配流不同时）
            self._declare_cross_stream(block_id, view.owner_stream)
            # 步 3：在搬运流上发起 D2H 逐块拷贝
            try:
                ticket = self._backend.copy_out(
                    entry_id, block_id, nbytes, stream_id=self._copy_stream_id
                )
                # 步 4：完成确认（经协议 confirm = 事件适配器占位，不许裸调厂商事件，T-6）
                if ticket is None or not self._confirm(ticket):
                    raise TierCopyFailed(
                        f"换出条目 {entry_id} 完成确认不通过；条目不落级二账本、"
                        f"不调 free_block，块留在用（3.5 / T-7）。"
                    )
                # 步 5：条目**先**落级二账本，**后** free_block（账本先行，T-7）
                self._entries[entry_id] = Tier2Entry(
                    entry_id=entry_id,
                    block_id=block_id,
                    nbytes=nbytes,
                    evicted_at=time.monotonic(),
                )
                self._pool.free_block(block_id)
            except TierCopyFailed:
                self._audit["copy_failures"] += 1
                raise
            except Exception as exc:  # 后端抛的拷贝异常：同 3.5 搬运过程报错
                self._audit["copy_failures"] += 1
                raise TierCopyFailed(
                    f"换出条目 {entry_id} 拷贝失败（{exc!r}）；条目不落级二账本、"
                    f"不调 free_block，块留在用（3.5 / T-7）。"
                ) from exc
            # 步 6：推进待回收（drain 超时是可恢复态，不算失败）
            self._pool.drain_stream(stream_id=self._copy_stream_id, timeout_s=0.0)
            self._budget_charge(nbytes)
            evicted_bytes += nbytes
            evicted.append(entry_id)
            basis.append(rule)
        self._audit["evicted_bytes"] += evicted_bytes
        self._audit["evicted_entries"] += len(evicted)
        self._audit["bypass_window_evicts"] += bypassed
        met_target = target_bytes is None or evicted_bytes >= target_bytes
        if not met_target:
            self._audit["unmet_target_evicts"] += 1
            reason = reason or "换出量未达目标（如实标注，不谎报，T-4）"
        elif not reason and stats_missing:
            reason = "部分候选判据 4/5 不可用，按中性序换出（降级，见 query_tier_capability）"
        if reason == "" and met_target and target_bytes is not None:
            reason = "达标"
        return EvictResult(
            evicted_bytes=evicted_bytes,
            evicted_entries=tuple(evicted),
            met_target=met_target,
            reason=reason,
            ordering_basis=tuple(basis),
            bypassed_residency=bypassed,
        )

    # ------------------------------------------------------------------
    # 操作 6：promote（换入五步，3.4；完成确认后才交块，T-7）
    # ------------------------------------------------------------------

    def promote(
        self,
        entry_id: str,
        *,
        stream_id: int | None = None,
        keep_tier2_copy: bool = False,
        prefetch: bool = False,
    ) -> PromoteOutcome:
        self._require_backend()
        self._audit["promote_calls"] += 1
        entry = self._require_promotable(entry_id, operation="promote")
        watermark = self._pool.query_watermark()
        if watermark.device_total <= 0:
            raise TierConfigError("池的设备侧三键不可用（T-1），无法做换入判读。")
        if stream_id is not None and stream_id not in self._known_streams:
            raise CrossStreamUndeclared(
                f"搬运流 {stream_id} 未登记（T-6）：请先 attach_stream / register_stream。"
            )
        if prefetch:
            gate = self._promote_prefetch_gate(entry, watermark)
            if gate is not None:
                return gate
        # —— 换入五步（3.4）——
        # 步 1：向池要块，用途标签 = KV 缓存；容量不足 ⇒ 报错，不许当未命中（T-4）
        try:
            block = self._pool.alloc_on_stream(
                entry.nbytes, stream_id=stream_id, purpose=Purpose.KV_CACHE
            )
        except CapacityExhausted as exc:
            raise PromoteCapacityExhausted(entry.nbytes, exc.watermark) from exc
        try:
            # 步 2：跨流声明
            self._declare_cross_stream(block.block_id, block.owner_stream)
            # 步 3：H2D 逐块拷贝
            ticket = self._backend.copy_in(
                entry_id, block.block_id, entry.nbytes, stream_id=self._copy_stream_id
            )
            # 步 4：完成确认
            if ticket is None or not self._confirm(ticket):
                raise TierCopyFailed(
                    f"换入条目 {entry_id} 完成确认不通过；归还已分的块后报错（3.5 / T-7）。"
                )
        except Exception as exc:
            # 3.5 换入失败：归还已分的块，向上报错
            self._audit["copy_failures"] += 1
            self._pool.free_block(block.block_id)
            if isinstance(exc, TierCopyFailed):
                raise
            raise TierCopyFailed(
                f"换入条目 {entry_id} 拷贝失败（{exc!r}）；已归还块 {block.block_id}（3.5）。"
            ) from exc
        # 步 5：**确认完成后**才把块交调用方（T-7）；级二副本按策略保留或释放
        if keep_tier2_copy:
            self._entries[entry_id] = replace(entry, replica_kept=True)
        else:
            self._entries.pop(entry_id, None)
            self._backend.drop_entry(entry_id)
        self._budget_charge(entry.nbytes)
        self._promoted_at[block.block_id] = time.monotonic()
        self._audit["promoted_bytes"] += entry.nbytes
        self._audit["promoted_entries"] += 1
        return PromoteResult(
            entry_id=entry_id,
            block_id=block.block_id,
            base_ref=block.storage_id,
            nbytes=entry.nbytes,
            kept_tier2_copy=keep_tier2_copy,
        )

    # ------------------------------------------------------------------
    # 异步在途轨（本增量）：发起即返回，完成确认（poll）后才变更归属（T-7 的异步形态）
    # ------------------------------------------------------------------

    def evict_async(
        self,
        candidates: Iterable[int],
        *,
        target_bytes: int | None = None,
        pressure: PressureLevel | None = None,
        stream_id: int | None = None,
        allow_bypass_min_residency: bool = False,
    ) -> AsyncEvictReceipt:
        """异步换出：判读与候选排序与 `evict` 完全一致，但**发起即返回**。

        换出六步里只做第 2 步（跨流声明）与第 3 步（发起 D2H）；第 4..6 步
        （完成确认、账本先行、还块推进屏障）推迟到 `poll_transfers` 的完成确认。
        预算闸按**发起**记账（在途搬运占拷贝带宽，保守）；级二容量按承诺量判读
        （已确认 + 换出在途）。发起期失败（copy_out 抛错 / 无完成凭据）按 3.5：
        该条目不落账本、不还块，此前已发起的在途搬运不受影响、仍在表。"""
        self._require_backend()
        self._audit["evict_calls"] += 1
        if target_bytes is None and pressure is None:
            raise TierError("evict_async 需要 target_bytes 或 pressure 至少其一（E2 口径）。")
        if stream_id is not None and stream_id not in self._known_streams:
            raise CrossStreamUndeclared(
                f"搬运流 {stream_id} 未登记（T-6）：请先 attach_stream / register_stream。"
            )
        if stream_id is not None:
            self._copy_stream_id = stream_id
        forced = pressure is not None
        watermark = self._pool.query_watermark()
        if watermark.device_total <= 0:
            raise TierConfigError("池的设备侧三键不可用（T-1），无法做换出判读。")
        gate = self._evict_gate(forced=forced, watermark=watermark)
        if gate is not None:
            return AsyncEvictReceipt(
                scheduled_bytes=0,
                scheduled_entries=(),
                transfers=(),
                met_target=False,
                reason=gate.reason,
            )
        ordered, skipped_labels, stats_missing, inflight_skipped = (
            self._collect_candidates(candidates)
        )
        if not ordered:
            self._idle_rounds += 1
            if self._idle_rounds >= self._policy.backoff_trigger_max:
                self._audit["backoff_deferrals"] += 1
            reason = "无可换出候选"
            if skipped_labels:
                reason += f"（{skipped_labels} 个候选属 R1..R5 永不下放标签，T-3）"
            if inflight_skipped:
                reason += f"（{inflight_skipped} 个候选正在途，二次调度被拒）"
            return AsyncEvictReceipt(
                scheduled_bytes=0,
                scheduled_entries=(),
                transfers=(),
                met_target=False,
                reason=reason,
            )
        self._idle_rounds = 0  # 有候选，空转退避立即复位（3.3 第 4 条）
        scheduled_bytes = 0
        scheduled: list[str] = []
        basis: list[str] = []
        bypassed = 0
        reason = ""
        tickets: list[TransferTicket] = []
        now = time.monotonic()
        for rank, block_id in ordered:
            if target_bytes is not None and scheduled_bytes >= target_bytes:
                break
            view = self._pool.block_view(block_id)
            nbytes = view.tier_bytes
            # 级二容量按承诺量判读：在途换出按将占用计（保守，防排空时超容量）
            if self._level2_committed() + nbytes > self._level2_capacity:
                reason = "级二容量不足，停止换出"
                break
            if not self._budget_allows(nbytes):
                self._audit["budget_gate_deferred"] += 1
                reason = "搬运预算闸：窗口内字节数到上限"
                break
            promoted_at = self._promoted_at.get(block_id)
            bypassed_this = 0
            if (
                promoted_at is not None
                and now - promoted_at < self._policy.min_residency_s
            ):
                if forced and allow_bypass_min_residency:
                    bypassed += 1
                    bypassed_this = 1
                else:
                    self._audit["min_residency_deferred"] += 1
                    continue
            rule = {0: "R9", 1: "R7", 2: "R10"}[rank]
            entry_id = f"b{block_id}"
            # —— 换出六步的异步前段（3.4）：步 2 跨流声明 + 步 3 发起 D2H ——
            self._declare_cross_stream(block_id, view.owner_stream)
            try:
                ticket = self._backend.copy_out(
                    entry_id, block_id, nbytes, stream_id=self._copy_stream_id
                )
                if ticket is None:
                    # 无完成凭据：永远无法确认，按发起失败处置（T-4 不静默挂起）
                    raise TierCopyFailed(
                        f"换出条目 {entry_id} 发起失败：搬运后端未返回完成凭据；"
                        f"条目不落级二账本、不调 free_block，块留在用（3.5 / T-7）。"
                    )
            except TierCopyFailed:
                self._audit["copy_failures"] += 1
                raise
            except Exception as exc:
                self._audit["copy_failures"] += 1
                raise TierCopyFailed(
                    f"换出条目 {entry_id} 发起失败（{exc!r}）；条目不落级二账本、"
                    f"不调 free_block，块留在用（3.5 / T-7）；此前已发起的 "
                    f"{len(scheduled)} 笔仍在途，经 poll_transfers 处置。"
                ) from exc
            self._budget_charge(nbytes)  # 发起即记账：在途搬运占带宽（保守）
            record = _TransferRecord(
                transfer_id=self._next_transfer_id(),
                direction=TransferDirection.EVICT,
                entry_id=entry_id,
                block_id=block_id,
                nbytes=nbytes,
                ticket=ticket,
                started_at=time.monotonic(),
                stream_id=self._copy_stream_id,
                rule=rule,
                bypassed_residency=bypassed_this,
                keep_tier2_copy=False,
            )
            self._transfers[record.transfer_id] = record
            scheduled_bytes += nbytes
            scheduled.append(entry_id)
            basis.append(rule)
            tickets.append(record.view())
        met_target = target_bytes is None or scheduled_bytes >= target_bytes
        if not met_target:
            self._audit["unmet_target_evicts"] += 1
            reason = reason or "发起量未达目标（如实标注，实际换出量以完成确认为准，T-4）"
        elif not reason and stats_missing:
            reason = "部分候选判据 4/5 不可用，按中性序发起（降级，见 query_tier_capability）"
        if reason == "" and met_target and target_bytes is not None:
            reason = "达标（发起量；实际换出量以完成确认为准）"
        if not reason:
            reason = "发起完成（实际换出量以完成确认为准）"
        return AsyncEvictReceipt(
            scheduled_bytes=scheduled_bytes,
            scheduled_entries=tuple(scheduled),
            transfers=tuple(tickets),
            met_target=met_target,
            reason=reason,
            ordering_basis=tuple(basis),
            bypassed_residency=bypassed,
        )

    def promote_async(
        self,
        entry_id: str,
        *,
        stream_id: int | None = None,
        keep_tier2_copy: bool = False,
        prefetch: bool = False,
    ) -> AsyncPromoteReceipt | CooldownRejected | PrefetchNotTriggered:
        """异步换入：门禁与 `promote` 完全一致，但**发起即返回**。

        换入五步里做第 1..3 步（向池要块、跨流声明、发起 H2D）；第 4..5 步
        （完成确认、交块）推迟到 `poll_transfers`——块标识与底座引用在完成确认
        时才经 `TransferOutcome.promoted` 交付（T-7 第 5 步）。发起期失败按 3.5：
        归还已预留的块，级二条目保留可重试。"""
        self._require_backend()
        self._audit["promote_calls"] += 1
        entry = self._require_promotable(entry_id, operation="promote_async")
        watermark = self._pool.query_watermark()
        if watermark.device_total <= 0:
            raise TierConfigError("池的设备侧三键不可用（T-1），无法做换入判读。")
        if stream_id is not None and stream_id not in self._known_streams:
            raise CrossStreamUndeclared(
                f"搬运流 {stream_id} 未登记（T-6）：请先 attach_stream / register_stream。"
            )
        if prefetch:
            gate = self._promote_prefetch_gate(entry, watermark)
            if gate is not None:
                return gate
        # —— 换入五步的异步前段（3.4）：步 1 要块 + 步 2 跨流声明 + 步 3 发起 H2D ——
        try:
            block = self._pool.alloc_on_stream(
                entry.nbytes, stream_id=stream_id, purpose=Purpose.KV_CACHE
            )
        except CapacityExhausted as exc:
            raise PromoteCapacityExhausted(entry.nbytes, exc.watermark) from exc
        try:
            self._declare_cross_stream(block.block_id, block.owner_stream)
            ticket = self._backend.copy_in(
                entry_id, block.block_id, entry.nbytes, stream_id=self._copy_stream_id
            )
            if ticket is None:
                raise TierCopyFailed(
                    f"换入条目 {entry_id} 发起失败：搬运后端未返回完成凭据；"
                    f"归还预留块后报错（3.5 / T-7）。"
                )
        except Exception as exc:
            self._audit["copy_failures"] += 1
            self._pool.free_block(block.block_id)
            if isinstance(exc, TierCopyFailed):
                raise
            raise TierCopyFailed(
                f"换入条目 {entry_id} 发起失败（{exc!r}）；已归还预留块 "
                f"{block.block_id}（3.5）。"
            ) from exc
        self._budget_charge(entry.nbytes)  # 发起即记账（保守，与 evict_async 同口径）
        record = _TransferRecord(
            transfer_id=self._next_transfer_id(),
            direction=TransferDirection.PROMOTE,
            entry_id=entry_id,
            block_id=block.block_id,
            nbytes=entry.nbytes,
            ticket=ticket,
            started_at=time.monotonic(),
            stream_id=self._copy_stream_id,
            rule="",
            bypassed_residency=0,
            keep_tier2_copy=keep_tier2_copy,
            base_ref=block.storage_id,
        )
        self._transfers[record.transfer_id] = record
        return AsyncPromoteReceipt(
            entry_id=entry_id, nbytes=entry.nbytes, transfer_id=record.transfer_id
        )

    def poll_transfers(self) -> TransferPollReport:
        """在途搬运的一次非阻塞轮询（T-7：超时 + 轮询，不阻塞等待）。

        每笔在途搬运问询一次完成确认：
        - 确认通过 ⇒ **此刻才变更归属**——换出：条目先落级二账本、后 free_block、
          推进屏障（六步的第 4..6 步）；换入：交块（五步的第 4..5 步）；
        - 未确认 ⇒ 仍在途；问询预算耗尽转 TIMED_OUT（可恢复态，留在表里，不算失败）；
        - confirm 抛错 ⇒ 确认期失败，回收路径按 3.5（换出不落账不还块、
          换入归还预留块），条目出表后可重新调度。"""
        self._require_backend()
        outcomes: list[TransferOutcome] = []
        for record in list(self._transfers.values()):
            try:
                done = self._backend.confirm(record.ticket)
            except Exception as exc:
                outcomes.append(self._fail_transfer(record, exc))
                continue
            if done:
                outcomes.append(self._finalize_transfer(record))
            else:
                record.polls += 1
                if (
                    record.state is not TransferState.TIMED_OUT
                    and record.polls >= _PENDING_TIMEOUT_POLLS
                ):
                    record.state = TransferState.TIMED_OUT
                outcomes.append(record.outcome(record.state))
        return TransferPollReport(
            outcomes=tuple(outcomes),
            still_in_flight=tuple(r.view() for r in self._transfers.values()),
        )

    def query_transfers(
        self, *, entry_id: str | None = None
    ) -> tuple[TransferTicket, ...]:
        """在途账本的可查视图（「在途」状态与跟踪的观测面）；可按条目过滤。"""
        self._require_open()
        return tuple(
            record.view()
            for record in self._transfers.values()
            if entry_id is None or record.entry_id == entry_id
        )

    # —— 异步在途轨的内部结算 ——

    def _next_transfer_id(self) -> str:
        self._transfer_seq += 1
        return f"tx-{self._transfer_seq}"

    def _finalize_transfer(self, record: _TransferRecord) -> TransferOutcome:
        """完成确认通过后的归属变更（每笔只结算一次：先出表，再动账本/交块）。"""
        self._transfers.pop(record.transfer_id, None)
        record.state = TransferState.CONFIRMED
        if record.direction is TransferDirection.EVICT:
            # 步 5：条目**先**落级二账本，**后** free_block（账本先行，T-7）
            self._entries[record.entry_id] = Tier2Entry(
                entry_id=record.entry_id,
                block_id=record.block_id,
                nbytes=record.nbytes,
                evicted_at=time.monotonic(),
            )
            self._pool.free_block(record.block_id)
            # 步 6：推进待回收（drain 超时是可恢复态，不算失败）；用发起时的搬运流
            self._pool.drain_stream(stream_id=record.stream_id, timeout_s=0.0)
            self._audit["evicted_bytes"] += record.nbytes
            self._audit["evicted_entries"] += 1
            self._audit["bypass_window_evicts"] += record.bypassed_residency
            return record.outcome(TransferState.CONFIRMED)
        # 换入：确认完成后才把块交调用方（T-7 第 5 步）；级二副本按策略保留或释放
        if record.keep_tier2_copy:
            entry = self._entries.get(record.entry_id)
            if entry is not None:
                self._entries[record.entry_id] = replace(entry, replica_kept=True)
            else:  # pragma: no cover - 防御：处置口跳过在途换入，正常不会走到
                self._backend.drop_entry(record.entry_id)
        else:
            self._entries.pop(record.entry_id, None)
            self._backend.drop_entry(record.entry_id)
        self._promoted_at[record.block_id] = time.monotonic()
        self._audit["promoted_bytes"] += record.nbytes
        self._audit["promoted_entries"] += 1
        result = PromoteResult(
            entry_id=record.entry_id,
            block_id=record.block_id,
            base_ref=record.base_ref,
            nbytes=record.nbytes,
            kept_tier2_copy=record.keep_tier2_copy,
        )
        return TransferOutcome(
            transfer_id=record.transfer_id,
            direction=record.direction,
            entry_id=record.entry_id,
            block_id=record.block_id,
            nbytes=record.nbytes,
            state=TransferState.CONFIRMED,
            polls=record.polls,
            promoted=result,
        )

    def _fail_transfer(self, record: _TransferRecord, exc: Exception) -> TransferOutcome:
        """确认期失败的回收（3.5）：条目出表（可重新调度），预算不退（带宽已耗，保守）。"""
        self._transfers.pop(record.transfer_id, None)
        record.state = TransferState.FAILED
        record.error = repr(exc)
        self._audit["copy_failures"] += 1
        if record.direction is TransferDirection.EVICT:
            # 发起时就没动账本没还块（异步前段只发起拷贝），此处保持：块留在用
            message = (
                f"换出条目 {record.entry_id} 在确认期失败（{exc!r}）；条目不落级二"
                f"账本、不调 free_block，块留在用（3.5 / T-7）。"
            )
        else:
            # 换入：归还预留块，级二条目保留可重试
            self._pool.free_block(record.block_id)
            message = (
                f"换入条目 {record.entry_id} 在确认期失败（{exc!r}）；已归还预留块 "
                f"{record.block_id}，级二条目保留可重试（3.5 / T-7）。"
            )
        outcome = record.outcome(TransferState.FAILED)
        return TransferOutcome(
            transfer_id=outcome.transfer_id,
            direction=outcome.direction,
            entry_id=outcome.entry_id,
            block_id=outcome.block_id,
            nbytes=outcome.nbytes,
            state=TransferState.FAILED,
            polls=outcome.polls,
            error=message,
        )

    # ------------------------------------------------------------------
    # 操作 7..10：查询与策略
    # ------------------------------------------------------------------

    def query_tier_watermark(self) -> TierWatermarkView:
        self._require_open()
        watermark = self._pool.query_watermark()
        if watermark.device_total <= 0:
            # T-1：级一任一侧取不到 ⇒ 报错，不用单边数充抵
            raise TierConfigError("池的设备侧三键不可用（T-1），不用引擎侧单边数充抵。")
        return TierWatermarkView(level1=watermark, level2=self._level2_view())

    def query_tier_capability(self) -> TierCapabilityView:
        """该口恒可用（含 backend 被置换前的形态），是降级可观测性的基础。"""
        available = self._backend is not None and not self._destroyed
        reason = ""
        if self._destroyed:
            reason = "管理器已销毁"
        elif self._backend is None:
            reason = "该后端无已注册搬运适配层（T-11）"
        return TierCapabilityView(
            available=available,
            unavailable_reason=reason,
            backend_id=self._backend_id,
            declared=self._declared,
            probed=self._probed,
            degradations=self._backend_degradations,
            effective_policy=self._policy,
            level2_capacity_bytes=self._level2_capacity,
            transfer_granularity_bytes=(
                self._probed.transfer_granularity_bytes if self._probed else 0
            ),
            backoff_rounds=self._idle_rounds,
            # 级三后端留位槽：恒 False（结构留位，设计第八节第 2 条，不承诺实现）
            tier3_backend_registered=False,
        )

    def query_tier_audit(self) -> TierAuditView:
        self._require_open()
        return TierAuditView(**self._audit)

    def set_tier_policy(self, policy: TierPolicy) -> TierPolicy:
        """改阈值与窗口。违反滞回带约束或试图关掉抖动抑制四件之一 ⇒ 报错不夹取（T-2）。"""
        self._require_open()
        watermark = self._pool.query_watermark()
        granularity = self._probed.transfer_granularity_bytes
        _validate_policy(
            policy, granularity, watermark.device_total, watermark.pool_total
        )
        self._policy = policy
        return self._policy

    # ------------------------------------------------------------------
    # 级三留位（本增量）：R13 处置口（设计第八节第 2 条——只定位置，不承诺实现）
    # ------------------------------------------------------------------

    def apply_r13_disposition(self) -> Tier3DispositionReport:
        """R13 处置：级二高水位 ⇒ 本应换出到级三；级三未实现（留位）⇒ 按级二
        淘汰策略丢弃并计入审计（tier3_missing_drops，E4 的「级三缺失丢弃」计数）。

        淘汰策略现状 = LRU（evicted_at 最旧先丢）——它是实现可调参数，不是冻结
        口径（设计第八节第 6 条）。有换入在途的条目不丢（级二副本是 H2D 的数据源，
        丢了就是在途搬运读悬空内存）。空处置（未到高水位）显式给原因，不报成功（T-4）。"""
        self._require_backend()
        level2 = self._level2_view()
        if level2.occupancy < self._policy.level2_high_watermark:
            return Tier3DispositionReport(
                target=TierLevel.TIER3,
                backend_registered=False,
                dropped_entries=(),
                dropped_bytes=0,
                reason=(
                    f"级二未到高水位（{level2.occupancy:.2f} < "
                    f"{self._policy.level2_high_watermark}），无需 R13 处置"
                ),
            )
        active_in = {
            record.entry_id
            for record in self._transfers.values()
            if record.direction is TransferDirection.PROMOTE
        }
        skipped_inflight = sum(1 for entry_id in active_in if entry_id in self._entries)
        threshold_bytes = self._policy.level2_high_watermark * self._level2_capacity
        droppable = sorted(
            (e for e in self._entries.values() if e.entry_id not in active_in),
            key=lambda entry: entry.evicted_at,
        )
        dropped: list[str] = []
        dropped_bytes = 0
        for entry in droppable:
            if self._level2_used() - dropped_bytes <= threshold_bytes:
                break  # 丢到回到高水位以下即停
            self._entries.pop(entry.entry_id, None)
            self._backend.drop_entry(entry.entry_id)
            dropped.append(entry.entry_id)
            dropped_bytes += entry.nbytes
            self._audit["tier3_missing_drops"] += 1
        reason = (
            f"级三未实现（留位）：按级二淘汰策略（LRU）丢弃 {len(dropped)} 条、"
            f"共 {dropped_bytes} 字节，计入级三缺失丢弃审计"
        )
        if skipped_inflight:
            reason += f"；{skipped_inflight} 条有换入在途，暂不丢弃"
        return Tier3DispositionReport(
            target=TierLevel.TIER3,
            backend_registered=False,
            dropped_entries=tuple(dropped),
            dropped_bytes=dropped_bytes,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # 操作 2 / 3：销毁与后端注册
    # ------------------------------------------------------------------

    def destroy(
        self, *, wait_for_drain: bool = True, timeout_s: float | None = None
    ) -> TierDestroyReport:
        """销毁：排空在途搬运、给出级二残留条目与未回级一的块清单（6.1 第 2 条）。

        - `wait_for_drain=False` 且存在在途 ⇒ 报错并附清单，不静默丢弃；
        - `wait_for_drain=True`：轮询推进在途搬运（只许超时+轮询，T-7），
          `timeout_s=None` 表示不等待（单次轮询即判）；到时仍在途 ⇒ 返回
          `timed_out=True` 的可恢复态报告，**不标销毁**，调用方排空后重试；
        - 在途排空（或本就没有）才标销毁并返回残留清单。"""
        self._require_open()
        if self._transfers:
            if not wait_for_drain:
                raise TierError(
                    f"存在在途搬运 {sorted(self._in_flight)}，不静默丢弃；"
                    f"先 poll_transfers 排空，或以 wait_for_drain=True 重试（6.1 第 2 条）。"
                )
            if timeout_s is None:
                # 未给超时 = 不等待：单次轮询即判，不阻塞（T-7）
                self.poll_transfers()
            else:
                deadline = time.monotonic() + timeout_s
                while self._transfers:
                    self.poll_transfers()
                    if not self._transfers:
                        break
                    if time.monotonic() + _DRAIN_POLL_INTERVAL_S >= deadline:
                        break
                    time.sleep(_DRAIN_POLL_INTERVAL_S)
            if self._transfers:
                # 排空超时：可恢复态上报（T-7「超时不等于失败」），不标销毁
                return TierDestroyReport(
                    residual_entries=tuple(self._entries.values()),
                    in_flight=tuple(sorted(self._in_flight)),
                    timed_out=True,
                )
        self._destroyed = True
        return TierDestroyReport(
            residual_entries=tuple(self._entries.values()),
            in_flight=(),
            timed_out=False,
        )

    def register_tier_backend(
        self, backend_id: str, adapter: TierBackend
    ) -> TierRegistration:
        """登记 / 更换一个搬运适配层。后端标识与池的后端不一致 ⇒ 报错（6.1 第 3 条）。"""
        self._require_open()
        pool_backend = self._pool.query_capability().backend_name
        if backend_id != pool_backend:
            raise TierError(
                f"后端标识 {backend_id!r} 与池的后端 {pool_backend!r} 不一致，报错不登记。"
            )
        registration = _register(adapter, self._pool)
        self._backend = adapter
        self._backend_id = registration.backend_id
        self._declared = registration.declared
        self._probed = registration.probed
        self._backend_degradations = registration.degradations
        return registration

    # —— 内部 ——

    def _level2_used(self) -> int:
        return sum(entry.nbytes for entry in self._entries.values())

    def _level2_view(self) -> Tier2Watermark:
        return Tier2Watermark(
            capacity_bytes=self._level2_capacity,
            used_bytes=self._level2_used(),
            entry_count=len(self._entries),
        )


def _register(backend: TierBackend, pool: BlockPool) -> TierRegistration:
    """注册结论 + 声明与实测的比对：声明探测不到 ⇒ 按降级登记并上报，不静默（T-4）。"""
    declared = backend.declare_capabilities()
    probed = backend.probe_capabilities()
    degradations: list[DegradationNotice] = []
    if declared.supports_pinned and not probed.supports_pinned:
        degradations.append(
            DegradationNotice(
                key="tier.pinned-unavailable",
                action="声明锁页但运行期探测不到，按普通页登记（3.5）",
                cost="正确性不受影响，搬运带宽下降",
            )
        )
    if declared.has_batch_copy and not probed.has_batch_copy:
        degradations.append(
            DegradationNotice(
                key="tier.batch-copy-unavailable",
                action="声明批量搬运原语但运行期探测不到，退逐块拷贝（3.5）",
                cost="搬运次数上升，带宽利用率下降",
            )
        )
    return TierRegistration(
        backend_id=backend.backend_id,
        declared=declared,
        probed=probed,
        degradations=tuple(degradations),
    )


# ----------------------------------------------------------------------
# 操作 1：create_tier_manager / 操作 2 的模块级别名
# ----------------------------------------------------------------------


def create_tier_manager(
    pool: BlockPool,
    backend: TierBackend | None,
    *,
    level2_capacity_bytes: int,
    policy: TierPolicy | None = None,
    transfer_stream: Any | None = None,
) -> TierManager:
    """在一个既有池上创建分层存储管理器（6.1 第 1 条）。

    报错不启的三种情形：级二容量 ≤ 级一池容量（T-8）；后端无已注册适配层（T-11）；
    池的设备侧三键不可用（T-1）。启动校验全部通过才返回。
    """
    return TierManager(
        pool,
        backend,
        level2_capacity_bytes=level2_capacity_bytes,
        policy=policy,
        transfer_stream=transfer_stream,
    )


def destroy_tier_manager(
    manager: TierManager, *, wait_for_drain: bool = True, timeout_s: float | None = None
) -> TierDestroyReport:
    """销毁：排空在途搬运、给出级二残留条目与未回级一的块清单（6.1 第 2 条）。"""
    return manager.destroy(wait_for_drain=wait_for_drain, timeout_s=timeout_s)
