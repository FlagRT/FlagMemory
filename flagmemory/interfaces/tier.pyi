# 分层存储管理器 · 对外接口的类型存根
#
# 这是什么：`docs/design/分层存储概念与设计基础.md` 第六节 6.1 那 10 条分层存储操作的
# **机器可读接口声明**。接口语言面沿用《显存池概念与设计基础》第五节的现行口径
# （类型存根 + 接口文档，ADR-0015，登记处 = 指挥台仓 decisions/）。
#
# 这不是什么：不是实现，也不是另一套签名。存根与实现**同一套形参**，由
# `tests/interfaces/test_tier_stub_matches_impl.py` 逐个核对（形参名、形参种类、
# 有无缺省值、有无返回标注），不一致就红 —— 这条进 CI。
#
# 池的 15 条对外接口不在这里（见 pool.pyi）；分层存储只调用、不改它们（设计 6.2）。
# 管理器本体不 import 推理框架与 offload 实现（T-5）；芯片差异只许出现在
# TierBackendCapability 与后端实现里。
#
# 异步在途轨增量（2026-11）：evict_async / promote_async / poll_transfers /
# query_transfers 与级三留位处置口 apply_r13_disposition 加进存根——
# 同步轨（evict / promote）签名一字不动，增量演化不破坏既有签名。

from typing import Any, Iterable, Protocol, runtime_checkable

from flagmemory.pool import BlockPool, PressureLevel, Purpose, WorkloadType
from flagmemory.pool.types import DegradationNotice, Watermark

class TierError(RuntimeError): ...
class TierUnavailable(TierError): ...
class TierConfigError(TierError): ...
class CrossStreamUndeclared(TierError): ...
class TierCopyFailed(TierError): ...
class PromoteCapacityExhausted(TierError):
    nbytes: int
    watermark: Watermark

class TierLevel:
    TIER1: str
    TIER2: str
    TIER3: str

class TransferDirection:
    EVICT: str
    PROMOTE: str

class TransferState:
    PENDING: str
    TIMED_OUT: str
    CONFIRMED: str
    FAILED: str

class TierBackendCapability:
    transfer_granularity_bytes: int
    supports_pinned: bool
    has_batch_copy: bool

class TierEntryStats:
    hits: int
    misses: int
    last_access_seq: int
    current_seq: int
    reuse_distance: int

@runtime_checkable
class TierBackend(Protocol):
    backend_id: str
    def declare_capabilities(self) -> TierBackendCapability: ...
    def probe_capabilities(self) -> TierBackendCapability: ...
    def copy_out(self, entry_id: str, block_id: int, nbytes: int, *, stream_id: int) -> Any: ...
    def copy_in(self, entry_id: str, block_id: int, nbytes: int, *, stream_id: int) -> Any: ...
    def confirm(self, ticket: Any) -> bool: ...
    def block_activity(self, block_id: int) -> TierEntryStats | None: ...
    def drop_entry(self, entry_id: str) -> None: ...

class TierPolicy:
    h_dev: float
    h_eng: float
    l_dev: float
    l_eng: float
    hysteresis_n: int
    cooldown_s: float
    min_residency_s: float
    budget_bytes: int
    budget_window_s: float
    backoff_trigger_max: int
    level2_high_watermark: float

class PlacementReading: ...
class PlacementVerdict: ...
class EvictResult: ...
class PromoteResult: ...
class CooldownRejected: ...
class PrefetchNotTriggered: ...
class Tier2Entry: ...
class Tier2Watermark: ...
class TierWatermarkView: ...
class TierRegistration: ...
class TierCapabilityView: ...
class TierAuditView: ...
class TierDestroyReport: ...
class TransferTicket: ...
class TransferOutcome: ...
class TransferPollReport: ...
class AsyncEvictReceipt: ...
class AsyncPromoteReceipt: ...
class Tier3DispositionReport: ...

class TierManager:
    """分层存储管理器本体。10 条操作里的 2..10 是这个类的方法，第 1 条是 create_tier_manager；
    异步在途轨（evict_async / promote_async / poll_transfers / query_transfers）与
    级三留位处置口（apply_r13_disposition）是同步轨之外的增量操作面。"""

    @property
    def destroyed(self) -> bool: ...

    # —— 2：销毁 ——
    def destroy(self, *, wait_for_drain: bool = ..., timeout_s: float | None = ...) -> TierDestroyReport:
        """在途搬运不静默丢弃；返回级二残留条目与未回级一的块清单。"""
        ...

    # —— 3：登记搬运后端 ——
    def register_tier_backend(self, backend_id: str, adapter: TierBackend) -> TierRegistration:
        """声明探测不到 ⇒ 按降级登记并上报，不静默（T-4）。"""
        ...

    # —— 4：放置判定（只判不动）——
    def classify_placement(
        self, *, block_id: int | None = ..., entry_id: str | None = ...
    ) -> PlacementVerdict:
        """目标级别 + 命中规则号 R1..R14 + 五条判据读数（含双指标两侧）。"""
        ...

    # —— 5：换出 ——
    def evict(
        self,
        candidates: Iterable[int],
        *,
        target_bytes: int | None = ...,
        pressure: PressureLevel | None = ...,
        stream_id: int | None = ...,
        allow_bypass_min_residency: bool = ...,
    ) -> EvictResult:
        """换出六步：账本先落级二后 free_block（T-7）；未达目标如实标注（T-4）。"""
        ...

    # —— 5a：异步换出（发起即返回，完成确认后才变更归属）——
    def evict_async(
        self,
        candidates: Iterable[int],
        *,
        target_bytes: int | None = ...,
        pressure: PressureLevel | None = ...,
        stream_id: int | None = ...,
        allow_bypass_min_residency: bool = ...,
    ) -> AsyncEvictReceipt:
        """发起即在途账本立账；条目落账/还块推迟到 poll_transfers 的完成确认（T-7）。"""
        ...

    # —— 6：换入 ——
    def promote(
        self,
        entry_id: str,
        *,
        stream_id: int | None = ...,
        keep_tier2_copy: bool = ...,
        prefetch: bool = ...,
    ) -> PromoteResult | CooldownRejected | PrefetchNotTriggered:
        """换入五步：完成确认后才交块（T-7）；池容量不足报错不许当未命中（T-4）。"""
        ...

    # —— 6a：异步换入（发起即返回，完成确认后才交块）——
    def promote_async(
        self,
        entry_id: str,
        *,
        stream_id: int | None = ...,
        keep_tier2_copy: bool = ...,
        prefetch: bool = ...,
    ) -> AsyncPromoteReceipt | CooldownRejected | PrefetchNotTriggered:
        """块向池预留但不交块；块标识与底座引用在完成确认时交付（T-7 第 5 步）。"""
        ...

    # —— 异步在途轨：轮询与在途账本查询 ——
    def poll_transfers(self) -> TransferPollReport:
        """一次非阻塞轮询：确认 ⇒ 变更归属（换出落账还块 / 换入交块）；
        未确认仍在途；confirm 抛错 ⇒ 按 3.5 回收。超时是可恢复态（T-7）。"""
        ...

    def query_transfers(self, *, entry_id: str | None = ...) -> tuple[TransferTicket, ...]:
        """在途账本的可查视图（「在途」状态与跟踪），可按条目过滤。"""
        ...

    # —— 级三留位：R13 处置口（设计第八节第 2 条，只定位置不承诺实现）——
    def apply_r13_disposition(self) -> Tier3DispositionReport:
        """级二高水位 ⇒ 本应换出到级三；级三未实现 ⇒ 按级二淘汰策略丢弃并计审计。"""
        ...

    # —— 7：三级水位 ——
    def query_tier_watermark(self) -> TierWatermarkView:
        """级一双指标（两侧都给）+ 级二账本 + 级三「未实现」。"""
        ...

    # —— 8：能力与降级态 ——
    def query_tier_capability(self) -> TierCapabilityView:
        """恒可用；降级形态与各阈值生效值可查。"""
        ...

    # —— 9：搬运审计 ——
    def query_tier_audit(self) -> TierAuditView:
        """换入换出计数、抖动四件触发数、越窗换出、级三缺失丢弃、在途重复调度拒绝。"""
        ...

    # —— 10：改阈值与窗口 ——
    def set_tier_policy(self, policy: TierPolicy) -> TierPolicy:
        """滞回带约束 L ≤ H − 粒度×N 违反即报错不夹取；禁整项关闭抖动抑制（T-2）。"""
        ...

    # —— 辅助（不在那 10 条里）——
    def attach_stream(self, stream: Any) -> int: ...

# —— 1：创建 ——
def create_tier_manager(
    pool: BlockPool,
    backend: TierBackend | None,
    *,
    level2_capacity_bytes: int,
    policy: TierPolicy | None = ...,
    transfer_stream: Any | None = ...,
) -> TierManager:
    """级二容量 ≤ 级一池容量 ⇒ 报错不启（T-8）；无已注册适配层 ⇒ 报错（T-11）；
    池的设备侧三键不可用 ⇒ 报错（T-1）。"""
    ...

def destroy_tier_manager(
    manager: TierManager,
    *,
    wait_for_drain: bool = ...,
    timeout_s: float | None = ...,
) -> TierDestroyReport:
    """销毁：排空在途搬运、给出级二残留条目与未回级一的块清单。"""
    ...
