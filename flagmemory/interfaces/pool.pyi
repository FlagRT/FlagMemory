# 显存池管理器 · 对外接口的类型存根
#
# 这是什么：`docs/design/显存池概念与设计基础.md` 第四节那 15 条对外操作的**机器可读接口
# 声明**。按同一份设计文档第五节的现行方案，类型存根（`.pyi`）+ 接口文档共同充当招标口径里
# 的「接口头文件与接口文档」。该方案状态 = **总组倾向已定、待最终拍板**（草案 ADR-0015,
# 登记处 = 指挥台仓 `decisions/`；一句话：接口语言面定为类型存根 + 接口文档，C 头留触发式）。
#
# 这不是什么：不是实现，也不是另一套签名。存根与实现**同一套形参**，由
# `tests/interfaces/test_stub_matches_impl.py` 逐个核对（形参名、形参种类、有无缺省值、
# 有无返回标注），不一致就红 —— 这条进 CI。
#
# C 头什么时候出：出现**非 Python 消费方**时再出，届时以设计文档第四节的接口契约为准生成。
# 契约本身不随语言面改变 —— 这也是第四节把「参数语义 / 返回 / 错误」写成契约而不是形参签名的
# 原因。当前已知消费方（推理引擎侧连接件、分层存储适配层、画像探针）全在 Python 侧。
#
# 哪些契约尚未实现、以及未实现的理由，见 `docs/impl/11-28-分级块池与流有序分配.md`。

from typing import Any, Iterable

from flagmemory.pool.base import BaseAllocator
from flagmemory.pool.tiers import TierSpec, TierTable
from flagmemory.pool.types import (
    AuditView,
    BlockView,
    BudgetView,
    CapabilityView,
    CompactResult,
    DestroyReport,
    CustodyView,
    DrainResult,
    FragmentationView,
    OwnershipVerdict,
    PoolForm,
    PressureLevel,
    Purpose,
    ReclaimResult,
    Watermark,
    WorkloadType,
)

DEFAULT_TIER_COUNT: int
DEFAULT_TIMEOUT_S: float

class ReusePolicy:
    """自管路的回收策略：三个旋钮与三家原生流有序池的属性同形。"""

    follow_event_dependencies: bool
    opportunistic: bool
    release_threshold_bytes: int | None
    def __init__(
        self,
        follow_event_dependencies: bool = ...,
        opportunistic: bool = ...,
        release_threshold_bytes: int | None = ...,
    ) -> None: ...

class BlockPool:
    """池本体。15 条对外操作里的 2..15 是这个类的方法，第 1 条是 `create_pool`。"""

    # —— 只读属性 ——
    @property
    def form(self) -> PoolForm: ...
    @property
    def workload(self) -> WorkloadType: ...
    @property
    def tier_table(self) -> TierTable: ...
    @property
    def native_stream_ordered(self) -> bool:
        """原生优化是否启用；当前为假，使用自管回收屏障。"""
        ...
    @property
    def destroyed(self) -> bool: ...

    # —— 2：销毁 ——
    def destroy_pool(
        self, *, wait_for_drain: bool = ..., timeout_s: float | None = ...
    ) -> DestroyReport:
        """进入关闭中；安全排空后才释放流引用。未完成资源托管；在用块抛 LeakDetected。"""
        ...

    # —— 3：按流分配 ——
    def alloc_on_stream(
        self, nbytes: int, *, stream_id: int | None = ..., purpose: Purpose = ...
    ) -> BlockView:
        """定档 → 复用或向底座取块 → 绑定所属流与用途标签。容量不足即报错。"""
        ...

    # —— 4：归还 ——
    def free_block(self, block_id: int) -> BlockView:
        """进待回收并取屏障凭据。归属不符 / 重复归还由池账本拦截（I-1）。"""
        ...

    # —— 5：登记流 ——
    def register_stream(self, stream: Any) -> int:
        """经后端核验设备后登记，强持有至安全关闭或隔离托管终结。"""
        ...

    # —— 6：声明跨流使用 ——
    def declare_cross_stream_use(self, block_id: int, stream_id: int) -> frozenset[int]:
        """把该流并入块的跨流借用集；缺跨流保护能力时只记池内集并上报降级态。"""
        ...

    # —— 7：推进待回收 ——
    def drain_stream(
        self, *, stream_id: int | None = ..., timeout_s: float | None = ...
    ) -> DrainResult:
        """把已确认完成的块转可复用。超时不等于失败（I-11）。"""
        ...

    # —— 8：水位 ——
    def query_watermark(self) -> Watermark:
        """设备侧三键 + 引擎侧四项，两侧都给（I-6）。"""
        ...

    # —— 9：碎片 ——
    def query_fragmentation(self) -> FragmentationView:
        """双指标碎片读数 + 各档空闲分布 + 可立刻满足的最大单块。"""
        ...

    # —— 10：碎片整理 ——
    def compact(
        self, *, tier_bytes: int | None = ..., allow_return_to_base: bool = ...
    ) -> CompactResult:
        """交回可复用块；无改善如实报无改善。本层做不了「合并」（I-9）。"""
        ...

    # —— 11：压力回收 ——
    def reclaim(
        self,
        *,
        target_bytes: int | None = ...,
        pressure: PressureLevel | None = ...,
        allow_tier_offload: bool = ...,
    ) -> ReclaimResult:
        """进程内回收，如实返回实际释放量；不做设备级恢复（I-5）。"""
        ...

    # —— 12：归属校验 ——
    def verify_block_ownership(self, block_id: int) -> OwnershipVerdict:
        """设备 / 上下文 / 账本三项分别给；缺只读观测则该项为未校验并上报。"""
        ...

    # —— 13：预算视图 ——
    def query_budget(self, *, purposes: Iterable[Purpose] | None = ...) -> BudgetView:
        """按用途标签聚合的已分配量 + 按负载类型的可压余量判读（I-7）。"""
        ...

    # —— 14：能力与降级态 ——
    def query_capability(self) -> CapabilityView:
        """实现形态、能力分支、生效的降级路径与代价。此口恒可用。"""
        ...

    # —— 15：审计视图 ——
    def query_audit(self) -> AuditView:
        """在世块数、待回收深度、隔离量与原因分布、复用命中率。"""
        ...

    # —— 辅助（不在那 15 条里）——
    def block_view(self, block_id: int) -> BlockView: ...
    def access_block(self, block_id: int) -> Any: ...

# —— 1：建池 ——
def create_pool(
    backend: Any,
    *,
    device_index: int = ...,
    form: PoolForm = ...,
    workload: WorkloadType = ...,
    base_allocator: BaseAllocator | None = ...,
    tier_spec: TierSpec | None = ...,
    tier_count: int = ...,
    reuse: ReusePolicy | None = ...,
    barrier_timeout_s: float = ...,
) -> BlockPool:
    """在调用方既有设备上下文内建池（I-4：池不建上下文），定下形态与档位表。"""
    ...


def list_custodied_pools() -> tuple[CustodyView, ...]:
    """只读托管元数据清单。"""
    ...


def recover_custodied_pool(custody_id: int) -> BlockPool:
    """找回托管池以查询、归还、排空或重试关闭。"""
    ...
