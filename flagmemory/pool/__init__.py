"""池本体：显存池管理器的核心路径 —— 分级块池 + 流有序分配。

硬约束（依据 ADR-0014 §3.5 第 4 条与四芯片显存能力调研第五节）：
- 不得 import 推理框架（vllm / torch_npu / 各厂商 plugin）；框架绑定下沉到
  flagmemory.tiering.offload 的可选适配层；
- 核心路径只许用四家齐备的 L0 能力，L1 能力一律走 flagmemory.caps 的能力分支；
- 池持有流必须强持有，禁止用弱引用做自动清理。

### 文件分工

| 模块 | 放什么 |
|---|---|
| `types` | 六态状态机、用途标签六类与策略、负载类型、实现形态、各结果视图、异常 |
| `base` | 池向下要的五个**可选**协议口：底座 / 分配粒度 / 原生流有序 / 跨流保护 / 上下文只读观测 |
| `tiers` | 档位表：按运行时查到的分配粒度切档，池本体不内嵌粒度常数 |
| `streams` | 流登记表：强持有到池销毁（`I-3`），刻意不 import weakref |
| `barrier` | 回收屏障：只轮询事件 + 单调钟截止，不阻塞等待（`I-2` / `I-11`） |
| `ledger` | 块账本：稳定底座与活租约映射、状态索引、分级空闲表、隔离区（`I-1`） |
| `pool` | `BlockPool` 与 `create_pool`：设计文档第四节那 15 条对外操作 |

对外接口的类型存根在 `flagmemory/interfaces/pool.pyi`，接口文档在
`flagmemory/interfaces/接口文档-显存池对外接口.md`；当前访问、关闭托管、检查入口及未实现项见
`docs/impl/11-28-分级块池与流有序分配.md`。
"""

from .barrier import DEFAULT_TIMEOUT_S, BarrierOutcome, BarrierTicket, RecycleBarrier
from .base import (
    BaseAllocator,
    ContextObserver,
    CrossStreamProtection,
    GranularitySource,
    StreamOrderedAllocator,
)
from .ledger import Block, Ledger
from .pool import BlockPool, ReusePolicy, create_pool, list_custodied_pools, recover_custodied_pool
from .streams import DEFAULT_STREAM_ID, StreamRegistry
from .tiers import DEFAULT_TIER_COUNT, TierSpec, TierTable, resolve_tier_table
from .types import (
    AuditView,
    BlockQuarantined,
    BlockState,
    BlockView,
    BudgetView,
    CapabilityView,
    CapacityExhausted,
    CompactResult,
    DegradationNotice,
    DestroyReport,
    CustodyView,
    DrainResult,
    FormUnavailable,
    FragmentationView,
    InvalidArgument,
    LeakDetected,
    LedgerInconsistent,
    NotWired,
    OwnershipVerdict,
    OwnershipViolation,
    PoolError,
    PoolForm,
    PressureLevel,
    PURPOSE_POLICY,
    Purpose,
    PurposePolicy,
    RECLAIM_ORDER,
    ReclaimResult,
    TierFree,
    Watermark,
    WorkloadType,
)

__all__ = [
    # 入口
    "create_pool",
    "list_custodied_pools",
    "recover_custodied_pool",
    "CustodyView",
    "BlockPool",
    "ReusePolicy",
    # 协议口（可选，缺则走已登记的降级路径）
    "BaseAllocator",
    "GranularitySource",
    "StreamOrderedAllocator",
    "CrossStreamProtection",
    "ContextObserver",
    # 档位
    "TierSpec",
    "TierTable",
    "resolve_tier_table",
    "DEFAULT_TIER_COUNT",
    # 流与屏障
    "StreamRegistry",
    "DEFAULT_STREAM_ID",
    "RecycleBarrier",
    "BarrierTicket",
    "BarrierOutcome",
    "DEFAULT_TIMEOUT_S",
    # 账本
    "Block",
    "Ledger",
    # 枚举与策略
    "BlockState",
    "Purpose",
    "PurposePolicy",
    "PURPOSE_POLICY",
    "RECLAIM_ORDER",
    "WorkloadType",
    "PoolForm",
    "PressureLevel",
    # 视图
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
    "DegradationNotice",
    # 异常
    "PoolError",
    "InvalidArgument",
    "CapacityExhausted",
    "OwnershipViolation",
    "LedgerInconsistent",
    "BlockQuarantined",
    "FormUnavailable",
    "NotWired",
    "LeakDetected",
]
