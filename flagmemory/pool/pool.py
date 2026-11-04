"""分级块池 + 流有序分配：`docs/design/显存池概念与设计基础.md` 第四节那 15 条接口的实现。

分层位置（设计文档 1.1）：这是**第三层**，站在厂商缓存分配器与推理框架预分配池**之上**。
池把底座当作「一次拿到的大块来源」，自己只管块的分级、归属、流序与账本 —— 不替代、不旁路，
也**不假设自己拿到的是裸内存**（`I-9`）。

向下只有一条依赖线：`flagmemory.caps` 的后端协议。本文件不 import 任何推理框架与厂商运行时
（`tools/check_layering.py` 规则 1 + 规则 2 静态把关）。

### 当前执行路径

池按流登记、按档复用，并用事件适配器确认回收寿命。原生流有序优化未启用，
`query_capability().native_stream_ordered` 为假。首次真实内存验证限定自管张量路径。
池接口由单线程调用；跨流数据依赖由调用方建立，回收登记不替代数据同步。
"""

from __future__ import annotations

import time
from itertools import count
from dataclasses import dataclass
from typing import Any, Iterable

from ..caps.backend import CapabilityNotDeclared
from ..caps.capabilities import Capability
from ..caps.degradation import degradation_for
from ..caps.query import report as caps_report
from .barrier import DEFAULT_TIMEOUT_S, BarrierOutcome, BarrierTicket, RecycleBarrier
from .base import (
    BaseAllocator,
    ContextObserver,
    CrossStreamProtection,
    StreamOrderedAllocator,
)
from .ledger import Block, Ledger
from .streams import DEFAULT_STREAM_ID, StreamRegistry
from .tiers import DEFAULT_TIER_COUNT, TierSpec, TierTable, resolve_tier_table
from .types import (
    AuditView,
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
    Purpose,
    PURPOSE_POLICY,
    RECLAIM_ORDER,
    ReclaimResult,
    Watermark,
    WorkloadType,
)

__all__ = ["ReusePolicy", "BlockPool", "create_pool", "list_custodied_pools", "recover_custodied_pool"]

# Explicit process-lifetime custody, without background work or finalizer device calls.
_CUSTODY_IDS = count(1)
_CUSTODIED_POOLS: dict[int, BlockPool] = {}


def list_custodied_pools() -> tuple[CustodyView, ...]:
    """Read-only snapshots of closing or quarantined resources held by this process."""
    return tuple(CustodyView(
        custody_id=key, closing=pool._closing,
        in_use_blocks=len(pool._ledger.by_state(BlockState.IN_USE)),
        pending_blocks=len(pool._ledger.by_state(BlockState.PENDING_RECYCLE)),
        quarantined_blocks=len(pool._ledger.by_state(BlockState.QUARANTINED)),
        held_bytes=sum(block.tier_bytes for block in pool._ledger.iter_blocks()),
        in_use_bytes=pool._ledger.bytes_in_state(BlockState.IN_USE),
        pending_bytes=pool._ledger.bytes_in_state(BlockState.PENDING_RECYCLE),
        quarantined_bytes=pool._ledger.bytes_in_state(BlockState.QUARANTINED),
    ) for key, pool in _CUSTODIED_POOLS.items())


def recover_custodied_pool(custody_id: int) -> BlockPool:
    """Recover a retained pool for queries, returns, draining and closing retries."""
    try:
        return _CUSTODIED_POOLS[custody_id]
    except KeyError:
        raise InvalidArgument(f"托管记录 {custody_id} 不存在或已安全结束") from None



@dataclass(frozen=True)
class ReusePolicy:
    """自管路的回收策略：三个旋钮与三家原生池属性同形。

    三家原生流有序池（寒武纪 MLU590 / 平头哥 PPU / 较新 CANN 的昇腾 910C）的属性语义高度
    同形：按事件依赖复用 / 机会性复用 / 释放阈值。自管路照这三个旋钮设计，于是两条路的
    行为可以对齐着调。

    | 旋钮 | 自管路里的含义 |
    |---|---|
    | `follow_event_dependencies` | 块的放行以事件依赖为准：屏障确认了所属流与跨流借用集上的操作才转可复用。**固定开启，不提供关闭** —— 关掉就违反状态机（设计文档 2.2：待回收的块不可被复用） |
    | `opportunistic` | 该档空闲表为空时，先顺手推进一次待回收队列再决定是否向底座取块 |
    | `release_threshold_bytes` | 池保留的可复用字节上限；归还路径上超过阈值的部分直接交回底座，不留在空闲表。`None` = 不设阈值 |
    """

    follow_event_dependencies: bool = True
    opportunistic: bool = True
    release_threshold_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.follow_event_dependencies:
            raise InvalidArgument(
                "follow_event_dependencies 不提供关闭：块的放行必须以事件依赖为准，"
                "关掉等于把「待回收」的块当「可复用」交给下一个人（违反设计文档 2.2 与 I-2）。"
            )
        if self.release_threshold_bytes is not None:
            if (
                not isinstance(self.release_threshold_bytes, int)
                or isinstance(self.release_threshold_bytes, bool)
                or self.release_threshold_bytes < 0
            ):
                raise InvalidArgument(
                    f"release_threshold_bytes 必须是非负 int 或 None，"
                    f"收到 {self.release_threshold_bytes!r}"
                )


class BlockPool:
    """显存池管理器的池本体。15 条对外操作都是这个类的方法（`create_pool` 是第 1 条）。

    不要直接构造：走 `create_pool`，它负责能力核验、形态决定与档位表推导。
    """

    # ------------------------------------------------------------------
    # 建池（接口 1：create_pool）
    # ------------------------------------------------------------------

    def __init__(
        self,
        backend: Any,
        *,
        device_index: int,
        form: PoolForm,
        workload: WorkloadType,
        tier_table: TierTable,
        base_allocator: BaseAllocator | None,
        reuse: ReusePolicy,
        degradations: tuple[DegradationNotice, ...],
        barrier_timeout_s: float,
    ) -> None:
        self._backend = backend
        self._device_index = device_index
        self._form = form
        self._workload = workload
        self._tiers = tier_table
        self._base_allocator = base_allocator
        self._reuse = reuse
        self._degradations = degradations
        self._ledger = Ledger()
        self._streams = StreamRegistry(backend, device_index)
        self._barrier = RecycleBarrier(
            backend, device_index=device_index, timeout_s=barrier_timeout_s
        )
        self._destroyed = False
        self._closing = False
        self._custody_id = next(_CUSTODY_IDS)
        #: 建池时的上下文观测值（有这项能力才有值），归属校验拿它对照
        self._context_token: Any = None
        if isinstance(backend, ContextObserver):
            try:
                self._context_token = backend.observe_context(device_index)
            except Exception:
                # 观测失败不等于不具备：按未观测处理，归属校验那一项会降级并上报。
                self._context_token = None
        # 默认流：池不建流，只把调用方当前流登记成默认流并强持有（I-3）。
        self._streams.register_default(backend.current_stream(device_index))

    # —— 只读属性 ——

    @property
    def form(self) -> PoolForm:
        return self._form

    @property
    def workload(self) -> WorkloadType:
        return self._workload

    @property
    def tier_table(self) -> TierTable:
        return self._tiers

    @property
    def native_stream_ordered(self) -> bool:
        """原生流有序优化是否启用；当前为假，自管路承担完整回收屏障。"""
        return False  # Native allocation/free completion is not validated for this path.

    @property
    def destroyed(self) -> bool:
        return self._destroyed

    # ------------------------------------------------------------------
    # 接口 2：destroy_pool
    # ------------------------------------------------------------------

    def destroy_pool(self, *, wait_for_drain: bool = True, timeout_s: float | None = None) -> DestroyReport:
        """关闭池：排空并交回安全资源；全部完成才放开流强引用。

        存在 `在用` 块时**不静默释放**（下面可能还在跑），而是抛 `LeakDetected` 并把泄漏
        清单挂在异常的 `.report` 上。排空超时按可恢复态上报（`I-11`），不阻塞。
        """
        self._require_alive()
        budget = self._barrier.budget(timeout_s)
        self._closing = True
        self._retain()
        timed_out = False
        if wait_for_drain:
            timed_out = self.drain_stream(timeout_s=budget).recoverable_timeout
        returned_bytes = 0
        returned_blocks = 0
        for block in self._ledger.by_state(BlockState.REUSABLE):
            returned_bytes += self._return_to_base(block)
            returned_blocks += 1
        leaked = tuple(b.view() for b in self._ledger.by_state(BlockState.IN_USE))
        pending = tuple(b.view() for b in self._ledger.by_state(BlockState.PENDING_RECYCLE))
        completed = len(self._ledger) == 0
        if completed:
            self._streams.release_all()
            self._destroyed = True
            _CUSTODIED_POOLS.pop(self._custody_id, None)
        report = DestroyReport(
            returned_bytes=returned_bytes, returned_blocks=returned_blocks,
            quarantined=self._ledger.quarantined_views(), leaked=leaked,
            drain_timed_out=timed_out, pending=pending, closing=not completed,
            completed=completed, custody_id=None if completed else self._custody_id,
        )
        if leaked:
            raise LeakDetected(report)
        return report

    # ------------------------------------------------------------------
    # 接口 3：alloc_on_stream
    # ------------------------------------------------------------------

    def alloc_on_stream(
        self,
        nbytes: int,
        *,
        stream_id: int | None = None,
        purpose: Purpose = Purpose.UNDECLARED,
    ) -> BlockView:
        """按流分配一块：定档 → 复用或向底座取块 → 绑定所属流与用途标签。

        失败一律报错（`I-10`）：容量不足不自动缩小请求、不自动换档、不自动阻塞等待；
        定档失败即报错，不自动扩档。
        """
        self._require_open()
        if not isinstance(purpose, Purpose):
            raise InvalidArgument(
                f"用途标签必须取自 Purpose 枚举（六类，见设计文档 4.3），收到 {purpose!r}"
            )
        owner = DEFAULT_STREAM_ID if stream_id is None else stream_id
        if not self._streams.has(owner):
            raise InvalidArgument(
                f"流标识 {owner} 未登记。请先 register_stream —— 池强持有流（I-3），"
                f"不替调用方猜流。"
            )

        tier = self._tiers.tier_for(nbytes)  # 参数非法与定档失败都在这里报
        self._ledger.alloc_requests += 1

        block = self._ledger.pop_free(tier)
        if block is None and self._reuse.opportunistic:
            # 机会性复用：空闲表空了，先顺手推进一次待回收队列再决定要不要去底座拿。
            self.drain_stream(timeout_s=0.0)
            block = self._ledger.pop_free(tier)

        if block is not None:
            return self._reuse_block(block, nbytes, owner, purpose)
        return self._fetch_block(tier, nbytes, owner, purpose)

    def _reuse_block(self, block: Block, nbytes: int, owner: int, purpose: Purpose) -> BlockView:
        """复用一块（设计文档 3.2）：改绑流、清空跨流借用集、改写用途标签。"""
        if block.tier_bytes != self._tiers.tier_for(nbytes):  # pragma: no cover - 防御
            self._quarantine(block, "档位不匹配", f"空闲表给出 {block.tier_bytes}")
            raise LedgerInconsistent(f"块 {block.block_id} 档位与请求不匹配，已转隔离。")
        self._ledger.renew(block)
        block.state = BlockState.ALLOCATING
        block.owner_stream = owner
        block.borrowers.clear()
        block.purpose = purpose
        block.nbytes = nbytes
        block.ticket = None
        block.state = BlockState.IN_USE
        self._ledger.reuse_hits += 1
        return block.view()

    def _fetch_block(self, tier: int, nbytes: int, owner: int, purpose: Purpose) -> BlockView:
        """向底座取一块新的。两条路在这里分叉，对外语义相同。"""
        try:
            if self.native_stream_ordered:
                # 原生路：包厂商的流有序分配原语。
                stream = self._streams.get(owner)
                base_ref = self._backend.alloc_from_pool_async(tier, stream, self._device_index)
            elif self._form is PoolForm.HANDLE:
                base_ref = self._backend.alloc_handle(
                    tier,
                    self._device_index,
                    "device",
                    purpose.value,
                    owner,
                )
            else:
                assert self._base_allocator is not None  # 建池时已核，见 create_pool
                base_ref = self._base_allocator.allocate(tier, self._device_index, stream=self._streams.get(owner))
        except MemoryError as exc:
            raise CapacityExhausted(
                nbytes, tier, self.query_watermark(), self._ledger.tier_free(tier)
            ) from exc
        except CapabilityNotDeclared as exc:
            # 形态是建池时的决定，不在单次分配里偷换（I-10 / 设计文档 3.1）。
            raise FormUnavailable(
                f"形态 {self._form.value} 要用的能力在运行期不可用：{exc}。"
                f"池不在单次分配里换形态 —— 请在建池时改形态或改后端声明。"
            ) from exc

        lease_id = self._ledger.next_block_id()
        block = Block(
            block_id=lease_id,
            storage_id=lease_id,
            tier_bytes=tier,
            nbytes=nbytes,
            base_ref=base_ref,
            owner_stream=owner,
            purpose=purpose,
            device_index=self._device_index,
            state=BlockState.ALLOCATING,
        )
        self._ledger.add(block)
        self._ledger.base_fetches += 1
        block.state = BlockState.IN_USE
        return block.view()

    # ------------------------------------------------------------------
    # 接口 4：free_block
    # ------------------------------------------------------------------

    def free_block(self, block_id: int) -> BlockView:
        """归还一块：进 `待回收` 并取屏障凭据。

        **归还不等于可复用**（设计文档 2.2）：调用方看到的「还回去了」只到 `待回收`；
        流序未确认时池绝不把块交给下一个人。

        错误、重复或过期租约由账本拒绝，不损伤当前块。当前租约的设备/上下文
        明确不符才隔离；不依赖厂商报错。
        """
        self._require_alive()
        block = self._ledger.require_live(block_id)

        verdict = self.verify_block_ownership(block_id)
        if not verdict.ok:
            self._quarantine(block, "归属校验失败", str(verdict))
            raise OwnershipViolation(
                f"块 {block_id} 归属校验不通过（设备 {verdict.device_ok} / "
                f"上下文 {verdict.context_ok} / 账本 {verdict.ledger_ok}），已转隔离。"
            )

        streams = {sid: self._streams.get(sid) for sid in block.related_streams}
        self._ledger.revoke(block)
        block.ticket = BarrierTicket()
        try:
            self._barrier.take_ticket(streams, block.ticket)
        except Exception as exc:
            # 取不到凭据就不能放行：宁可归还失败并上报，也不给一张空凭据。
            self._quarantine(block, "取屏障凭据失败", repr(exc))
            raise PoolError(
                f"块 {block_id} 取屏障凭据失败，已转隔离：{exc!r}。"
                f"池不接受空凭据 —— 那会让还在用的显存立刻变成「可复用」。"
            ) from exc

        block.state = BlockState.PENDING_RECYCLE
        return block.view()

    # ------------------------------------------------------------------
    # 接口 5：register_stream
    # ------------------------------------------------------------------

    def register_stream(self, stream: Any) -> int:
        """经后端核验设备后登记流，强持有至安全关闭；隔离由进程托管继续持有。"""
        self._require_open()
        return self._streams.register(stream)

    # ------------------------------------------------------------------
    # 接口 6：declare_cross_stream_use
    # ------------------------------------------------------------------

    def declare_cross_stream_use(self, block_id: int, stream_id: int) -> frozenset[int]:
        """声明一块将被另一条流读写，把该流并入块的跨流借用集。

        两条路径（设计文档 2.4），**对外语义完全相同**：声明了跨流保护能力的后端经统一层
        登记；未声明的（寒武纪 MLU590）只记池内借用集并上报降级态。差别只在池内部的屏障成本。

        边界照抄设计文档：**池不承诺在未声明跨流使用的情况下保护跨流读写。**
        """
        self._require_open()
        block = self._ledger.require_live(block_id)
        if block.state is not BlockState.IN_USE:
            raise InvalidArgument(
                f"块 {block_id} 状态是 {block.state.value}，只有「在用」的块能声明跨流使用。"
            )
        if not self._streams.has(stream_id):
            raise InvalidArgument(f"流标识 {stream_id} 未登记，请先 register_stream。")

        if self._has_cap(Capability.CROSS_STREAM_PROTECTION) and isinstance(
            self._backend, CrossStreamProtection
        ):
            self._backend.record_stream(block.base_ref, self._streams.get(stream_id))

        block.borrowers.add(stream_id)
        return frozenset(block.borrowers)

    # ------------------------------------------------------------------
    # 接口 7：drain_stream
    # ------------------------------------------------------------------

    def drain_stream(self, *, stream_id: int | None = None, timeout_s: float | None = None) -> DrainResult:
        """推进待回收队列，把已确认完成的块转可复用。

        零等待只观察；正等待到期仍保持可恢复，不按次数隔离。所有块共享
        单调时钟截止时间，参数必须有限非负；后端单次 query 必须不阻塞。
        """
        self._require_alive()
        budget = self._barrier.budget(timeout_s)
        deadline = time.monotonic() + budget
        if stream_id is not None and not self._streams.has(stream_id):
            raise InvalidArgument(f"流标识 {stream_id} 未登记。")

        promoted_blocks = 0
        promoted_bytes = 0
        timed_out: list[int] = []
        quarantined: list[int] = []

        for block in self._ledger.by_state(BlockState.PENDING_RECYCLE):
            if stream_id is not None and stream_id not in block.related_streams:
                continue
            ticket = block.ticket
            if ticket is None:  # pragma: no cover - 防御：待回收必有凭据
                self._quarantine(block, "待回收缺屏障凭据")
                quarantined.append(block.block_id)
                continue

            outcome = self._barrier.advance(ticket, budget, deadline=deadline)
            if outcome == BarrierOutcome.DONE:
                promoted_bytes += self._promote(block)
                promoted_blocks += 1
            elif outcome == BarrierOutcome.FAULT:
                self._quarantine(block, "事件适配器报错")
                quarantined.append(block.block_id)
            elif budget > 0:
                timed_out.append(block.block_id)

        return DrainResult(
            promoted_blocks=promoted_blocks,
            promoted_bytes=promoted_bytes,
            still_pending=len(self._ledger.by_state(BlockState.PENDING_RECYCLE)),
            timed_out_blocks=tuple(timed_out),
            quarantined_blocks=tuple(quarantined),
            recoverable_timeout=bool(timed_out),
        )

    def _promote(self, block: Block) -> int:
        """屏障确认后处置一块：进空闲表，或按 `I-12` / 释放阈值直接交回底座。"""
        policy = PURPOSE_POLICY[block.purpose]
        if not policy.reusable:
            # I-12：权重与通信缓冲永不进复用池；「未声明」按最保守处理。
            # 调用方显式归还了，就交回底座，而不是留着给别人复用。
            block.state = BlockState.REUSABLE  # 过渡：交回底座的前置状态
            return self._return_to_base(block)

        threshold = self._reuse.release_threshold_bytes
        if threshold is not None:
            held = self._ledger.bytes_in_state(BlockState.REUSABLE)
            if held + block.tier_bytes > threshold:
                # 释放阈值旋钮：超出部分不留在空闲表，直接交回底座。
                block.state = BlockState.REUSABLE
                return self._return_to_base(block)

        block.state = BlockState.REUSABLE
        self._ledger.push_free(block)
        return block.tier_bytes

    # ------------------------------------------------------------------
    # 接口 8：query_watermark
    # ------------------------------------------------------------------

    def query_watermark(self) -> Watermark:
        """水位：设备侧三键 + 引擎侧四项，**两侧都给**（`I-6`）。

        设备侧查不到就报错，**不用引擎侧单边数充抵** —— 预分配会把设备侧读数压成恒平，
        只看一侧会误判成「池没有动态」或「存在泄漏」。
        """
        self._require_alive()
        try:
            info = self._backend.device_memory_info(self._device_index)
        except CapabilityNotDeclared as exc:
            raise PoolError(
                f"设备侧显存三键查询不可用：{exc}。按 I-6 双指标缺一不可，"
                f"池不用引擎侧单边数充抵。"
            ) from exc
        return Watermark(
            device_total=info.total,
            device_used=info.used,
            device_free=info.free,
            pool_in_use=self._ledger.bytes_in_state(BlockState.IN_USE),
            pool_reusable=self._ledger.bytes_in_state(BlockState.REUSABLE),
            pool_pending=self._ledger.bytes_in_state(BlockState.PENDING_RECYCLE),
            pool_quarantined=self._ledger.bytes_in_state(BlockState.QUARANTINED),
        )

    # ------------------------------------------------------------------
    # 接口 9：query_fragmentation
    # ------------------------------------------------------------------

    def query_fragmentation(self) -> FragmentationView:
        """碎片：双指标 + 各档空闲分布 + 本池可立刻满足的最大单块。

        注意 `largest_servable_bytes` 是「空闲表能立刻给出的最大单块」，**不是地址空间
        连续量** —— 池不持有地址空间（`I-9`），给不出连续量这个数，就不假装给。
        """
        self._require_alive()
        watermark = self.query_watermark()
        tiers = tuple(self._ledger.tier_free(tier) for tier in self._tiers.tiers)
        servable = max((t.tier_bytes for t in tiers if t.reusable_blocks), default=0)
        pool_total = watermark.pool_total
        return FragmentationView(
            watermark=watermark,
            tiers=tiers,
            largest_servable_bytes=servable,
            engine_fragmentation=(watermark.pool_reusable / pool_total) if pool_total else 0.0,
            device_free_ratio=(
                watermark.device_free / watermark.device_total if watermark.device_total else 0.0
            ),
        )

    # ------------------------------------------------------------------
    # 接口 10：compact
    # ------------------------------------------------------------------

    def compact(self, *, tier_bytes: int | None = None, allow_return_to_base: bool = True) -> CompactResult:
        """碎片整理：把 `可复用` 的块交回底座，返回整理前后的双指标读数。

        `在用` 与 `待回收` 一概不动（设计文档 3.4 前置条件）。无改善就**如实返回「无改善」**，
        不把一次空操作报成成功（`I-10`）。

        这一层**做不了「合并」**：合并要求池能动地址空间，而池不持有地址空间、也不假设拿到
        裸内存（`I-9`）。所以本实现的整理动作只有「交回底座」一种，这是现状而非待优化项。
        """
        self._require_alive()
        before = self.query_fragmentation()
        targets = self._tiers.tiers if tier_bytes is None else (tier_bytes,)
        if tier_bytes is not None and tier_bytes not in self._tiers.tiers:
            raise InvalidArgument(
                f"档位 {tier_bytes} 不在档位表里（共 {len(self._tiers.tiers)} 档）。"
            )

        returned_bytes = 0
        returned_blocks = 0
        if allow_return_to_base:
            for tier in targets:
                for block in self._ledger.free_blocks(tier):
                    returned_bytes += self._return_to_base(block)
                    returned_blocks += 1

        after = self.query_fragmentation()
        if not allow_return_to_base:
            note = (
                "不允许交回底座时本层无可做的整理动作：合并要求动地址空间，"
                "而池不持有地址空间（I-9）。如实报无改善。"
            )
        elif returned_blocks:
            note = f"交回底座 {returned_blocks} 块 / {returned_bytes} 字节（本层的整理动作只有交回）。"
        else:
            note = "没有可交回的可复用块，无改善。"
        return CompactResult(
            before=before,
            after=after,
            returned_bytes=returned_bytes,
            returned_blocks=returned_blocks,
            improved=returned_blocks > 0,
            note=note,
        )

    # ------------------------------------------------------------------
    # 接口 11：reclaim
    # ------------------------------------------------------------------

    def reclaim(
        self,
        *,
        target_bytes: int | None = None,
        pressure: PressureLevel | None = None,
        allow_tier_offload: bool = False,
    ) -> ReclaimResult:
        """压力回收：进程内回收，如实返回实际释放量。

        只动 `可复用` 的块，且按用途标签的次序走（临时工作区 → 激活 → KV 缓存）；
        **权重与通信缓冲、以及「未声明」的块一概不动**（`I-12`）。

        达不到目标量就返回实际量并标「未达目标」，不谎报（`I-10`）。
        **不做任何设备级恢复**（`I-5`）：四家里只有昇腾有设备级真重建，另三家分别是无原语、
        确认不具备、以及原语返回成功但设备随后全废 —— 任何自救路径在三家上跑不通。
        """
        self._require_alive()
        if (target_bytes is None) == (pressure is None):
            raise InvalidArgument("target_bytes 与 pressure 必须给且只给一个。")
        if allow_tier_offload:
            raise NotWired(
                "分层换出的触发口在本实现里没有接线：池与 flagmemory.tiering 的联动属分层存储"
                "那条出口，不在分级块池这条里。按「不给空壳」的纪律这里报错，而不是返回一个"
                "假的成功 —— 未实现项已列进交付说明。"
            )
        if target_bytes is not None:
            if not isinstance(target_bytes, int) or isinstance(target_bytes, bool) or target_bytes <= 0:
                raise InvalidArgument(f"target_bytes 必须是正 int，收到 {target_bytes!r}")
            requested = target_bytes
        else:
            assert pressure is not None
            if not isinstance(pressure, PressureLevel):
                raise InvalidArgument(f"pressure 必须取自 PressureLevel，收到 {pressure!r}")
            reusable = self._ledger.bytes_in_state(BlockState.REUSABLE)
            requested = int(reusable * pressure.fraction)

        released_bytes = 0
        released_blocks = 0
        by_purpose: dict[Purpose, int] = {}
        for purpose in RECLAIM_ORDER:
            if released_bytes >= requested:
                break
            for tier in self._tiers.tiers:
                if released_bytes >= requested:
                    break
                for block in self._ledger.free_blocks(tier):
                    if released_bytes >= requested:
                        break
                    if block.purpose is not purpose:
                        continue
                    freed = self._return_to_base(block)
                    released_bytes += freed
                    released_blocks += 1
                    by_purpose[purpose] = by_purpose.get(purpose, 0) + freed

        met = released_bytes >= requested
        return ReclaimResult(
            requested_bytes=requested,
            released_bytes=released_bytes,
            released_blocks=released_blocks,
            met_target=met,
            released_by_purpose=by_purpose,
            note=(
                "达到目标量。"
                if met
                else "未达目标：可复用块里能动的已全部交回（权重/通信缓冲/未声明按 I-12 不动）。"
            ),
        )

    # ------------------------------------------------------------------
    # 接口 12：verify_block_ownership
    # ------------------------------------------------------------------

    def verify_block_ownership(self, block_id: int) -> OwnershipVerdict:
        """块归属校验：设备 / 上下文 / 账本**三项分别给**，主动查，不等厂商报（`I-1`）。

        上下文只读观测能力缺失（寒武纪 MLU590）时，上下文那一项退为 `None`（未校验）并
        上报降级态 —— **不可静默当作校验通过**。
        """
        self._require_alive()
        block = self._ledger.require_live(block_id)
        degradations: list[DegradationNotice] = []

        device_ok = block.device_index == self._device_index

        context_ok: bool | None
        if self._has_cap(Capability.CONTEXT_OBSERVE) and isinstance(self._backend, ContextObserver):
            try:
                context_ok = self._backend.observe_context(self._device_index) == self._context_token
            except Exception as exc:
                context_ok = None
                degradations.append(
                    DegradationNotice(
                        key="context_observe_failed",
                        action=f"上下文只读观测调用失败（{exc!r}），该项退为未校验",
                        cost="归属校验强度下降，须靠账本审计补",
                    )
                )
        else:
            context_ok = None
            degradations.append(
                DegradationNotice(
                    key="no_context_observe",
                    action="后端没有上下文只读观测，归属校验只靠池自己的登记表",
                    cost="该家上归属校验强度下降，须靠账本审计补（调研 §5.2 第 2 条）",
                )
            )

        ledger_ok = block_id in self._ledger and block.state is not BlockState.QUARANTINED

        return OwnershipVerdict(
            block_id=block_id,
            device_ok=device_ok,
            context_ok=context_ok,
            ledger_ok=ledger_ok,
            degradations=tuple(degradations),
        )

    # ------------------------------------------------------------------
    # 接口 13：query_budget
    # ------------------------------------------------------------------

    def query_budget(self, *, purposes: Iterable[Purpose] | None = None) -> BudgetView:
        """预算视图：按用途标签聚合的已分配量 + **按负载类型**的可压余量判读。

        `I-7`：向量化负载的池是**纯余量**（可压），生成式负载的池是**真实并发容量**
        （压了就掉并发）。只看水位数字不看负载类型的仲裁一律视为缺陷 —— 所以这个口的
        返回里，判读文本与负载类型绑在一起给，不给一个裸数字。
        """
        self._require_alive()
        totals = self._ledger.allocated_by_purpose()
        if purposes is not None:
            wanted = set(purposes)
            for p in wanted:
                if not isinstance(p, Purpose):
                    raise InvalidArgument(f"用途标签必须取自 Purpose 枚举，收到 {p!r}")
            totals = {p: v for p, v in totals.items() if p in wanted}

        watermark = self.query_watermark()
        compressible = watermark.pool_reusable
        if self._workload is WorkloadType.EMBEDDING:
            reading = (
                "向量化负载：池是纯余量，可复用量可放心压（画像结论 3；"
                "该类负载池为整块分配、本来无碎片）。"
            )
        elif self._workload is WorkloadType.GENERATIVE:
            reading = (
                "生成式负载：池是真实并发容量，压了就掉并发（画像结论 3 实例：显存利用率参数"
                "由高压到低，并发能力相差数倍）。这个数字不可当作「可放心压的余量」读。"
            )
        else:
            reading = (
                "混合负载：可压部分按生成式口径保守读 —— 判读需要上层按实际请求构成拆分，"
                "池不替上层猜。"
            )
        return BudgetView(
            workload=self._workload,
            allocated_by_purpose=totals,
            watermark=watermark,
            compressible_bytes=compressible,
            compressible_reading=reading,
            source="池自统计（设备方向提供按标签聚合口时改为复用那一侧）",
        )

    # ------------------------------------------------------------------
    # 接口 14：query_capability
    # ------------------------------------------------------------------

    def query_capability(self) -> CapabilityView:
        """能力与降级态：实现形态、启用的能力分支、生效的降级路径与各自代价。

        这个口**恒可用**，是降级可观测性的基础（`I-10`：静默降级算缺陷）。
        """
        rep = caps_report(self._backend)
        return CapabilityView(
            backend_name=rep.backend_name,
            form=self._form,
            native_stream_ordered=self.native_stream_ordered,
            core_path_ready=rep.core_path_ready,
            missing_core=rep.missing_core,
            branches_enabled=frozenset(
                cap for cap in (
                    Capability.MEMORY_HANDLE,
                    Capability.CROSS_STREAM_PROTECTION,
                    Capability.CONTEXT_OBSERVE,
                    Capability.TRUE_BOUNDED_SYNC,
                ) if cap in rep.declared
            ),
            degradations=self._degradations,
            tier_source=self._tiers.source,
        )

    # ------------------------------------------------------------------
    # 接口 15：query_audit
    # ------------------------------------------------------------------

    def query_audit(self) -> AuditView:
        """审计视图：在世块数、待回收深度、隔离量与原因分布、复用命中率。"""
        self._require_alive()
        quarantined = self._ledger.by_state(BlockState.QUARANTINED)
        return AuditView(
            live_blocks=len(self._ledger),
            pending_depth=len(self._ledger.by_state(BlockState.PENDING_RECYCLE)),
            quarantined_blocks=len(quarantined),
            quarantined_bytes=sum(b.tier_bytes for b in quarantined),
            quarantine_reasons=self._ledger.quarantine_reasons(),
            alloc_requests=self._ledger.alloc_requests,
            reuse_hits=self._ledger.reuse_hits,
            base_fetches=self._ledger.base_fetches,
        )

    # ------------------------------------------------------------------
    # 辅助（非对外接口）
    # ------------------------------------------------------------------

    def block_view(self, block_id: int) -> BlockView:
        """取得当前在用租约的元数据快照；已归还租约不再可查。"""
        self._require_alive()
        return self._ledger.require_live(block_id).view()

    def access_block(self, block_id: int) -> Any:
        """取得当前活租约的可消费底座对象；归还后本入口失效。

        调用方必须在归还前完成工作提交与相关流登记。Python 别名不能撤销，
        归还后不得再用保留的对象读写或排入工作。跨流数据依赖由调用方建立。
        """
        self._require_open()
        block = self._ledger.require_live(block_id)
        if not self.verify_block_ownership(block_id).ok:
            raise OwnershipViolation("当前设备/上下文与租约不符，拒绝访问")
        return block.base_ref

    def _retain(self) -> None:
        _CUSTODIED_POOLS[self._custody_id] = self

    def _quarantine(self, block: Block, reason: str, detail: str = "") -> None:
        self._ledger.quarantine(block, reason, detail)
        self._retain()

    def _require_open(self) -> None:
        self._require_alive()
        if self._closing:
            raise PoolError("池关闭中，不接受新增工作或底座访问。")

    def _has_cap(self, cap: Capability) -> bool:
        try:
            return cap in self._backend.declared_capabilities()
        except Exception:  # pragma: no cover - 后端连声明都给不出
            return False

    def _require_alive(self) -> None:
        if self._destroyed:
            raise PoolError("池已销毁，不接受任何操作。")

    def _return_to_base(self, block: Block) -> int:
        """把一块交回底座。底座释放报错 → 转隔离并上报（设计文档 2.2）。"""
        nbytes = block.tier_bytes
        try:
            if self.native_stream_ordered and block.owner_stream is not None:
                self._backend.free_async(block.base_ref, self._streams.get(block.owner_stream))
            elif self._form is PoolForm.HANDLE:
                self._backend.free_handle(block.base_ref)
            else:
                assert self._base_allocator is not None
                self._base_allocator.release(block.base_ref)
        except Exception as exc:
            self._quarantine(block, "底座释放报错", repr(exc))
            raise PoolError(
                f"块 {block.block_id} 交回底座时报错，已转隔离：{exc!r}"
            ) from exc
        self._ledger.retire(block)
        return nbytes


# ======================================================================
# 接口 1：create_pool
# ======================================================================


def create_pool(
    backend: Any,
    *,
    device_index: int = 0,
    form: PoolForm = PoolForm.AUTO,
    workload: WorkloadType = WorkloadType.MIXED,
    base_allocator: BaseAllocator | None = None,
    tier_spec: TierSpec | None = None,
    tier_count: int = DEFAULT_TIER_COUNT,
    reuse: ReusePolicy | None = None,
    barrier_timeout_s: float = DEFAULT_TIMEOUT_S,
) -> BlockPool:
    """在调用方**既有**设备上下文内创建一个池，定下实现形态与档位表。

    `I-4`：池绝不自建、切换或销毁设备上下文 —— 昆仑芯平台只允许一个上下文，第二次创建
    直接失败，抢建会让框架起不来。所以这里只观测上下文，不建。

    形态怎么定（设计文档 1.3 + 3.1）：

    | `form` | 行为 |
    |---|---|
    | `TENSOR` | 张量级（四家通用的默认路径）。必须给 `base_allocator` —— 取空张量要 import 框架，而池本体不许 |
    | `HANDLE` | 句柄形态（优化路径）。后端未声明 `MEMORY_HANDLE` 则报 `FormUnavailable`，**不静默退化**（`I-10`） |
    | `AUTO` | 给了 `base_allocator` 就用张量级；没给而后端声明了句柄能力就用句柄形态（这条选择会进 `query_capability` 的降级清单，不静默）；两者都没有就报错 |

    `I-8`：核心路径只许用四家齐备的 L0 能力。L0 不齐 ⇒ 池在这款芯片上**不该起**，
    而不是降级 —— 所以这里直接报错。
    """
    rep = caps_report(backend)
    if not rep.core_path_ready:
        missing = "、".join(sorted(cap.value for cap in rep.missing_core))
        raise PoolError(
            f"后端 {rep.backend_name} 缺 L0 核心能力：{missing}。"
            f"按 I-8，核心路径只许用四家齐备的 L0 能力 —— 缺了就是这款芯片不支持池，"
            f"不是降级问题，池不起。"
        )

    degradations: list[DegradationNotice] = []
    has_handle = Capability.MEMORY_HANDLE in rep.declared

    if form is PoolForm.HANDLE and not has_handle:
        raise FormUnavailable(
            f"强制句柄形态，但后端 {rep.backend_name} 未声明统一内存句柄能力（L1，"
            f"寒武纪 MLU590 那一档未声明）。按 I-10 这里报错，不静默退化为张量级 —— "
            f"形态是建池时的决定。"
        )
    if form is PoolForm.TENSOR and base_allocator is None:
        raise InvalidArgument(
            "张量级形态必须给 base_allocator：取空张量作底座要 import 框架，"
            "而池本体不许（分层门禁规则 1），所以底座由外面注入。"
        )

    resolved = form
    if form is PoolForm.AUTO:
        if base_allocator is not None:
            resolved = PoolForm.TENSOR
        elif has_handle:
            resolved = PoolForm.HANDLE
            degradations.append(
                DegradationNotice(
                    key="auto_form_picked_handle",
                    action="自动形态：未注入张量级底座，改用句柄形态（默认路径是张量级）",
                    cost="该后端上的行为与四家通用默认路径不同，对账要按句柄形态读",
                )
            )
        else:
            raise InvalidArgument(
                "自动形态定不下来：没注入 base_allocator（张量级底座），后端也没声明统一内存"
                "句柄能力。请给 base_allocator，或换一个声明了句柄能力的后端。"
            )

    if resolved is PoolForm.TENSOR and not isinstance(base_allocator, BaseAllocator):
        raise InvalidArgument(
            "base_allocator 不满足 flagmemory.pool.base.BaseAllocator 协议"
            "（要有 allocate / release 两个方法）。"
        )

    # 缺句柄能力 ⇒ 张量级池，这条是降级表第 1 条（也是四家通用的默认路径）。
    if not has_handle:
        path = degradation_for(Capability.MEMORY_HANDLE)
        if path is not None:
            degradations.append(
                DegradationNotice(key=path.key, action=path.action, cost="失去句柄形态的记账校验基准")
            )
    # 其余已登记的降级路径照抄能力层的计划，池不自建第二套说法。
    for path in rep.plan:
        if path.missing is Capability.MEMORY_HANDLE:
            continue
        degradations.append(
            DegradationNotice(key=path.key, action=path.action, cost=f"依据 {path.basis}")
        )

    tier_table = resolve_tier_table(
        backend, tier_spec=tier_spec, device_index=device_index, tier_count=tier_count
    )

    return BlockPool(
        backend,
        device_index=device_index,
        form=resolved,
        workload=workload,
        tier_table=tier_table,
        base_allocator=base_allocator,
        reuse=reuse or ReusePolicy(),
        degradations=tuple(degradations),
        barrier_timeout_s=barrier_timeout_s,
    )
