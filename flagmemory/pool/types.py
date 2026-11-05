"""池的类型面：块状态、用途标签、负载类型、实现形态、结果视图与异常。

纯标准库、零框架依赖（`tools/check_layering.py` 规则 1 静态把关）。

本文件只放**数据与语义**，不放行为。语义全部取自 `docs/design/显存池概念与设计基础.md`：
六态状态机见该文件第二节 2.2，用途标签六类见 4.3，不变量 `I-1`..`I-12` 见 3.5
（登记处 = 那份设计文档第三节第 5 小节，命名空间 = 显存池组件内部）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    # 枚举
    "BlockState",
    "Purpose",
    "PurposePolicy",
    "PURPOSE_POLICY",
    "RECLAIM_ORDER",
    "WorkloadType",
    "PoolForm",
    "PressureLevel",
    # 视图
    "DegradationNotice",
    "BlockView",
    "Watermark",
    "TierFree",
    "FragmentationView",
    "CompactResult",
    "ReclaimResult",
    "DrainResult",
    "OwnershipVerdict",
    "BudgetView",
    "CapabilityView",
    "AuditView",
    "DestroyReport",
    "CustodyView",
    # 异常
    "PoolError",
    "InvalidArgument",
    "CapacityExhausted",
    "TierResolutionError",
    "OwnershipViolation",
    "LedgerInconsistent",
    "BlockQuarantined",
    "FormUnavailable",
    "NotWired",
    "LeakDetected",
]


# ======================================================================
# 枚举
# ======================================================================


class BlockState(Enum):
    """块的六态（设计文档 2.2）。`QUARANTINED` 是终态，不自动迁出。"""

    ALLOCATING = "申请中"
    IN_USE = "在用"
    PENDING_RECYCLE = "待回收"
    REUSABLE = "可复用"
    RETURNED = "已归还"
    QUARANTINED = "隔离"


class Purpose(Enum):
    """用途标签六类（设计文档 4.3；枚举可增，语义不可改）。

    标签落在句柄/请求上而不是池内部，原因是**通信缓冲那一类由别的方向申请** ——
    池只有从标签上看到它，才知道「这块别动」。
    """

    WEIGHT = "权重"
    KV_CACHE = "KV缓存"
    ACTIVATION = "激活"
    COMM_BUFFER = "通信缓冲"
    SCRATCH = "临时工作区"
    UNDECLARED = "未声明"


@dataclass(frozen=True)
class PurposePolicy:
    """池对某个用途标签的处理策略。"""

    #: 归还后能否进复用池
    reusable: bool
    #: 压力回收能否动它
    reclaimable: bool
    #: 复用/回收的优先序，数越小越优先被回收（仅对 reclaimable 的有意义）
    reclaim_rank: int
    #: 一句人话，进 `query_budget` / 报错文本
    note: str


#: 用途标签 → 策略。`I-12`：权重与通信缓冲永不被池回收或复用，只记账；
#: 「未声明」按最保守处理（不回收、不复用）。
PURPOSE_POLICY: dict[Purpose, PurposePolicy] = {
    Purpose.WEIGHT: PurposePolicy(
        reusable=False,
        reclaimable=False,
        reclaim_rank=99,
        note="模型参数，整个服务生命周期常驻，永不回收只记账（I-12）",
    ),
    Purpose.COMM_BUFFER: PurposePolicy(
        reusable=False,
        reclaimable=False,
        reclaim_rank=99,
        note="集合通信收发缓冲，由通信方向申请，池动它会打架，只记账（I-12）",
    ),
    Purpose.UNDECLARED: PurposePolicy(
        reusable=False,
        reclaimable=False,
        reclaim_rank=99,
        note="申请方未给标签，按最保守处理：不回收、不复用（I-12）",
    ),
    Purpose.SCRATCH: PurposePolicy(
        reusable=True,
        reclaimable=True,
        reclaim_rank=0,
        note="算子工作空间，最优先回收",
    ),
    Purpose.ACTIVATION: PurposePolicy(
        reusable=True,
        reclaimable=True,
        reclaim_rank=1,
        note="前向/反向中间结果，短生命周期，块复用的主战场",
    ),
    Purpose.KV_CACHE: PurposePolicy(
        reusable=True,
        reclaimable=True,
        reclaim_rank=2,
        note="推理上下文缓存，可分页、可降级到主机侧（分层存储的主对象）",
    ),
}

#: 压力回收的候选顺序：生命周期短的先走（设计文档 3.2「复用优先级」）。
RECLAIM_ORDER: tuple[Purpose, ...] = tuple(
    p for p, pol in sorted(PURPOSE_POLICY.items(), key=lambda kv: kv[1].reclaim_rank)
    if pol.reclaimable
)


class WorkloadType(Enum):
    """负载类型。`I-7`：预算仲裁必须区分它，只看水位数字的仲裁算缺陷。

    同一个「显存利用率」参数在两类负载下语义不同：向量化服务的池是**纯余量**（可压），
    生成式服务的池是**真实并发容量**（压了就掉并发）。
    """

    EMBEDDING = "向量化"
    GENERATIVE = "生成式"
    MIXED = "混合"


class PoolForm(Enum):
    """实现形态。形态在建池时一次定下，**不在单次分配里偷换**（设计文档 3.1）。"""

    AUTO = "自动"
    TENSOR = "张量级"   # 四家通用的默认路径
    HANDLE = "句柄形态"  # 声明了统一内存句柄（L1）的三家上的优化路径


class PressureLevel(Enum):
    """压力等级（`reclaim` 的两种入参之一，另一种是目标字节数）。"""

    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"

    @property
    def fraction(self) -> float:
        """该等级要回收掉可复用量的几成。"""
        return {"低": 0.25, "中": 0.5, "高": 1.0}[self.value]


# ======================================================================
# 视图（对外返回的纯数据，可直接打进日志）
# ======================================================================


@dataclass(frozen=True)
class DegradationNotice:
    """一条生效中的降级。`I-10`：降级必须可观测，静默降级算缺陷。"""

    key: str
    action: str
    cost: str

    def __str__(self) -> str:  # pragma: no cover - 便于报错时直读
        return f"{self.key}: {self.action}（代价：{self.cost}）"


@dataclass(frozen=True)
class BlockView:
    """租约元数据快照；block_id 每次租用更新，storage_id 标识底座。"""

    block_id: int
    storage_id: int
    state: BlockState
    tier_bytes: int
    nbytes: int
    purpose: Purpose
    owner_stream: int | None
    borrowers: frozenset[int]
    device_index: int


@dataclass(frozen=True)
class Watermark:
    """水位。`I-6`：设备侧 + 引擎侧**同时给**，单指标不得作为池内决策的唯一输入。

    设备侧三键来自后端；引擎侧四项来自池自己的账本。预分配会把设备侧读数压成恒平，
    块的真实涨落只在引擎侧可见 —— 这就是为什么两侧缺一不可。
    """

    # —— 设备侧（后端三键查询）——
    device_total: int
    device_used: int
    device_free: int
    # —— 引擎侧（池账本）——
    pool_in_use: int
    pool_reusable: int
    pool_pending: int
    pool_quarantined: int

    @property
    def pool_total(self) -> int:
        """池持有的全部字节（在用 + 可复用 + 待回收 + 隔离）。"""
        return self.pool_in_use + self.pool_reusable + self.pool_pending + self.pool_quarantined


@dataclass(frozen=True)
class TierFree:
    """一个档位的空闲分布。"""

    tier_bytes: int
    reusable_blocks: int
    reusable_bytes: int


@dataclass(frozen=True)
class FragmentationView:
    """碎片读数。双指标同时给（`I-6`），另给各档空闲分布与最大可立刻满足的单块。"""

    watermark: Watermark
    tiers: tuple[TierFree, ...]
    #: 本池**可立刻满足**的最大单块字节数（取自空闲表命中，不是地址空间连续量 ——
    #: 池不持有地址空间，见 `I-9`）
    largest_servable_bytes: int
    #: 引擎侧碎片率：可复用字节 ÷ 池持有字节。池持有为 0 时是 0.0
    engine_fragmentation: float
    #: 设备侧碎片率：空闲字节 ÷ 总字节（预分配下通常恒平，单看会误判）
    device_free_ratio: float


@dataclass(frozen=True)
class CompactResult:
    """碎片整理结果。`I-10`：无改善要如实说「无改善」，不把空操作报成成功。"""

    before: FragmentationView
    after: FragmentationView
    returned_bytes: int
    returned_blocks: int
    improved: bool
    note: str


@dataclass(frozen=True)
class ReclaimResult:
    """压力回收结果。达不到目标量要如实返回实际量并标「未达目标」，不谎报。"""

    requested_bytes: int
    released_bytes: int
    released_blocks: int
    met_target: bool
    #: 按用途标签聚合的释放量
    released_by_purpose: dict[Purpose, int]
    #: 分层换出回调回报的字节数（未允许分层换出、未登记回调或回报 None 时为 0）
    tier_offloaded_bytes: int
    note: str


@dataclass(frozen=True)
class DrainResult:
    """推进待回收队列的结果。超时不等于失败（`I-11`），按可恢复态上报。

    `promoted_*` = 这一轮**屏障确认后离开待回收**的块数与字节数。绝大多数进了对应档位的
    空闲表；按 `I-12` 不许进复用池的那几类标签（权重 / 通信缓冲 / 未声明）被调用方显式归还后
    直接交回底座，也计在这里 —— 它们同样是「离开了待回收」。
    """

    promoted_blocks: int
    promoted_bytes: int
    still_pending: int
    timed_out_blocks: tuple[int, ...]
    quarantined_blocks: tuple[int, ...]
    #: 超时是可恢复态，不是失败
    recoverable_timeout: bool


@dataclass(frozen=True)
class OwnershipVerdict:
    """归属校验结论：设备 / 上下文 / 账本三项分别给，不合并成一个布尔。"""

    block_id: int
    device_ok: bool
    context_ok: bool | None  # None = 该后端无上下文只读观测能力，已降级
    ledger_ok: bool
    degradations: tuple[DegradationNotice, ...]

    @property
    def ok(self) -> bool:
        """设备与账本两项必须过；上下文项只在**明确为 `False`** 时算不通过。

        上下文项为 `None` 表示该后端没有上下文只读观测能力（寒武纪 MLU590 那一档），
        这叫**未校验**，不叫通过 —— 所以 `degradations` 里必然有一条说明它为什么没校验，
        调用方据此知道这次放行的校验强度低于别家。这不是静默当作通过：
        若把 `None` 直接判成不通过，那一家上任何块都归还不了，池等于不可用。
        """
        return self.device_ok and self.ledger_ok and self.context_ok is not False


@dataclass(frozen=True)
class BudgetView:
    """预算视图。`I-7`：可压余量的判读随负载类型变，不是一个纯数字。"""

    workload: WorkloadType
    allocated_by_purpose: dict[Purpose, int]
    watermark: Watermark
    #: 可压余量判读。向量化 = 纯余量（可压）；生成式 = 真实并发容量（压了掉并发）
    compressible_bytes: int
    compressible_reading: str
    #: 聚合量从哪来（设备方向提供聚合口则复用；此处是池自统计）
    source: str


@dataclass(frozen=True)
class CapabilityView:
    """能力与降级态。这个口恒可用，是降级可观测性的基础。"""

    backend_name: str
    form: PoolForm
    native_stream_ordered: bool
    #: 建池换出回调是否已登记（接线可观测，I-10；未登记时允许分层换出仍报 NotWired）
    tier_offload_wired: bool
    core_path_ready: bool
    missing_core: frozenset[Any]
    branches_enabled: frozenset[Any]
    degradations: tuple[DegradationNotice, ...]
    tier_source: str


@dataclass(frozen=True)
class AuditView:
    """审计视图：在世块数、待回收深度、隔离量与原因分布、复用命中率。"""

    live_blocks: int
    pending_depth: int
    quarantined_blocks: int
    quarantined_bytes: int
    quarantine_reasons: dict[str, int]
    alloc_requests: int
    reuse_hits: int
    base_fetches: int

    @property
    def reuse_hit_rate(self) -> float:
        """复用命中率 = 复用命中次数 ÷ 分配请求数。无请求时是 0.0。"""
        return self.reuse_hits / self.alloc_requests if self.alloc_requests else 0.0


@dataclass(frozen=True)
class DestroyReport:
    """销毁报告：交回了多少、谁还在隔离、谁没归还（泄漏判据）。"""

    returned_bytes: int
    returned_blocks: int
    quarantined: tuple[BlockView, ...]
    leaked: tuple[BlockView, ...]
    drain_timed_out: bool
    pending: tuple[BlockView, ...]
    closing: bool
    completed: bool
    custody_id: int | None


@dataclass(frozen=True)
class CustodyView:
    """进程级安全托管清单，不暴露底座对象。"""

    custody_id: int
    closing: bool
    in_use_blocks: int
    pending_blocks: int
    quarantined_blocks: int
    held_bytes: int
    in_use_bytes: int
    pending_bytes: int
    quarantined_bytes: int


# ======================================================================
# 异常
# ======================================================================


class PoolError(RuntimeError):
    """池的异常基类。总纪律：失败就报错，不静默退化（`I-10`）。"""


class InvalidArgument(PoolError):
    """参数非法（字节数非正、标签不在枚举内、流未登记等）。"""


class CapacityExhausted(PoolError):
    """容量不足。

    `I-10`：不自动缩小请求、不自动换档、不自动触发阻塞等待 —— 直接报错，
    并把当前双指标水位与该档空闲量带在错误里，便于上层归因。
    """

    def __init__(self, nbytes: int, tier_bytes: int, watermark: Watermark, tier_free: TierFree) -> None:
        self.nbytes = nbytes
        self.tier_bytes = tier_bytes
        self.watermark = watermark
        self.tier_free = tier_free
        super().__init__(
            f"容量不足：请求 {nbytes} 字节（定档 {tier_bytes}）；"
            f"该档空闲 {tier_free.reusable_blocks} 块 / {tier_free.reusable_bytes} 字节；"
            f"设备侧 空闲 {watermark.device_free}/{watermark.device_total}，"
            f"引擎侧 在用 {watermark.pool_in_use} / 可复用 {watermark.pool_reusable} / "
            f"待回收 {watermark.pool_pending}。"
            f"按 I-10 不自动缩小请求、不自动换档、不自动阻塞等待。"
        )


class TierResolutionError(PoolError):
    """档位表推导不出来，或请求定不进任何档。

    档位粒度**不写死常数**（设计文档 3.1）：要么由后端的分配粒度查询推导，
    要么由调用方显式给配置项。两者都没有就报这个，绝不编一个默认值。
    """


class OwnershipViolation(PoolError):
    """非本池块 / 重复归还 / 归属不符。

    `I-1`：**绝不依赖厂商报错**来发现这类错误 —— 昇腾的裸释放对二次释放静默返回成功、
    昆仑芯的缓存分配器删除如实报错，两家行为相反，所以拦截必须由池自己的账本完成。
    """


class LedgerInconsistent(PoolError):
    """账本与底座不符。相关块转隔离并上报，不继续动块。"""


class BlockQuarantined(PoolError):
    """这块已在隔离区。隔离是终态：不复用、不释放、只上报。"""


class FormUnavailable(PoolError):
    """强制了某个实现形态，而该后端未声明对应能力。

    `I-10`：**不静默退化**为另一种形态 —— 形态是建池时的决定，报错让调用方改声明。
    """


class NotWired(PoolError):
    """这条路在本实现里没有接线。

    存在的意义是**不给空壳**：与其返回一个假的成功，不如明确说「这里还没接」，
    并在交付说明里列进未实现清单。
    """


class LeakDetected(PoolError):
    """销毁时还有 `在用` 块没归还。

    池**不静默释放**它们（下面可能还在跑），而是把泄漏清单挂在异常上交出去。
    报告在 `.report`（一份 `DestroyReport`）。
    """

    def __init__(self, report: DestroyReport) -> None:
        self.report = report
        leaked = ", ".join(str(v.block_id) for v in report.leaked)
        super().__init__(
            f"销毁时仍有 {len(report.leaked)} 块处在「在用」（块标识：{leaked}）。"
            f"池不静默释放在用块；已交回 {report.returned_bytes} 字节，"
            f"隔离 {len(report.quarantined)} 块。泄漏清单见异常的 .report。"
        )


@dataclass
class QuarantineRecord:
    """一条隔离记录（内部用，不进对外视图的 `__all__`）。"""

    block_id: int
    reason: str
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
