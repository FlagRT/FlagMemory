"""分层存储：设备显存 → 主机内存 → 更慢介质的放置与换入换出。

三套 KV→Host 卸载栈在 flagmemory.tiering.offload 下。它们是**框架绑定的可选适配层**：
import 时需要宿主栈的 torch / vLLM，且各绑一个宿主栈版本，同一环境里只有一套能导入成功。
池本体不经过它们，也不反向依赖它们。

分层存储管理器本体在 `flagmemory.tiering.manager`（第一增量，2026-11）：放置决策
R1..R14、双指标触发判读 E1..E4 / I1..I3、抖动抑制四件、级二账本与搬运审计，
不变量 T-1..T-11 的登记处是 `docs/design/分层存储概念与设计基础.md` 第五节。
管理器本体不 import 推理框架、不 import offload 的任何实现（T-5）——
搬运后端只经 `manager.TierBackend` 协议接入。
"""

from .manager import (
    CooldownRejected,
    CrossStreamUndeclared,
    EvictResult,
    PlacementReading,
    PlacementVerdict,
    PrefetchNotTriggered,
    PromoteCapacityExhausted,
    PromoteResult,
    Tier2Entry,
    Tier2Watermark,
    TierAuditView,
    TierBackend,
    TierBackendCapability,
    TierCapabilityView,
    TierConfigError,
    TierCopyFailed,
    TierDestroyReport,
    TierEntryStats,
    TierError,
    TierLevel,
    TierManager,
    TierPolicy,
    TierRegistration,
    TierUnavailable,
    TierWatermarkView,
    create_tier_manager,
    destroy_tier_manager,
)

__all__ = [
    "CooldownRejected",
    "CrossStreamUndeclared",
    "EvictResult",
    "PlacementReading",
    "PlacementVerdict",
    "PrefetchNotTriggered",
    "PromoteCapacityExhausted",
    "PromoteResult",
    "Tier2Entry",
    "Tier2Watermark",
    "TierAuditView",
    "TierBackend",
    "TierBackendCapability",
    "TierCapabilityView",
    "TierConfigError",
    "TierCopyFailed",
    "TierDestroyReport",
    "TierEntryStats",
    "TierError",
    "TierLevel",
    "TierManager",
    "TierPolicy",
    "TierRegistration",
    "TierUnavailable",
    "TierWatermarkView",
    "create_tier_manager",
    "destroy_tier_manager",
]
