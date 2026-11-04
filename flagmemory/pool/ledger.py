"""池的账本：块的身份、状态索引、分级空闲表、隔离区与计数器。

这个文件是不变量 `I-1` 的落点：**回收不依赖厂商报错。** 四家行为相反 —— 昇腾 910C 的裸
释放原语对二次释放与非法指针**静默返回成功**，昆仑芯 P800 的缓存分配器删除**如实报错** ——
依赖厂商报错等于在一家上完全失效。所以重复归还、非本池块、非法引用一律由这里的账本拦截。

两条配套纪律：

- **块标识生命周期内不复用**（设计文档 2.1）：已归还的块标识不再分配给新块，于是审计对账
  时「这个标识指的是哪一块」永不含糊。
- **隔离是终态**（2.2）：池宁可泄漏一块、把问题暴露出来，也不把一块状态不明的显存再交给
  别人。隔离量与隔离原因分布是对外可查的审计指标。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from .types import (
    BlockQuarantined,
    BlockState,
    BlockView,
    OwnershipViolation,
    Purpose,
    QuarantineRecord,
    TierFree,
)

__all__ = ["Block", "Ledger"]


@dataclass
class Block:
    """一块的全部账目（设计文档 2.1 的字段表）。

    `base_ref` 是**底座引用**：张量级形态下是底座张量对象，句柄形态下是统一层的内存句柄。
    **池不持有厂商指针**，也不假设底座底下是裸内存（`I-9`）。
    """

    block_id: int
    tier_bytes: int
    nbytes: int
    base_ref: Any
    owner_stream: int | None
    purpose: Purpose
    device_index: int
    state: BlockState
    #: 跨流借用集：曾被哪些流读写过（池内流标识）
    borrowers: set[int] = field(default_factory=set)
    #: 屏障凭据（仅待回收期间有值）
    ticket: Any = None

    def view(self) -> BlockView:
        """对外快照。刻意不含 `base_ref` —— 不让调用方绕过池直接动底座。"""
        return BlockView(
            block_id=self.block_id,
            state=self.state,
            tier_bytes=self.tier_bytes,
            nbytes=self.nbytes,
            purpose=self.purpose,
            owner_stream=self.owner_stream,
            borrowers=frozenset(self.borrowers),
            device_index=self.device_index,
        )

    @property
    def related_streams(self) -> tuple[int, ...]:
        """屏障要等的全部流：所属流 + 跨流借用集（设计文档 2.4）。"""
        streams = set(self.borrowers)
        if self.owner_stream is not None:
            streams.add(self.owner_stream)
        return tuple(sorted(streams))


class Ledger:
    """块账本。所有状态迁移都经这里，于是「账本与实际不符」只有一个地方要查。"""

    def __init__(self) -> None:
        self._blocks: dict[int, Block] = {}
        #: 已归还的块标识（不复用，用来区分「非本池块」与「重复归还」）
        self._retired: set[int] = set()
        self._next_block_id = 1
        #: 档位 → 可复用块标识（先进先出，便于复现问题）
        self._free: dict[int, list[int]] = {}
        self._quarantine: list[QuarantineRecord] = []
        # —— 计数器（进 query_audit）——
        self.alloc_requests = 0
        self.reuse_hits = 0
        self.base_fetches = 0

    # —— 身份 ——

    def next_block_id(self) -> int:
        """发一个新块标识。**不复用已归还的标识**（设计文档 2.1）。"""
        block_id = self._next_block_id
        self._next_block_id += 1
        return block_id

    def add(self, block: Block) -> None:
        if block.block_id in self._blocks:
            raise OwnershipViolation(f"块标识 {block.block_id} 已在账本里，不得重复登记。")
        self._blocks[block.block_id] = block

    # —— 查询 ——

    def get(self, block_id: int) -> Block:
        """按标识取块。

        `I-1`：这里是重复归还与非本池块的拦截点，**不等厂商报错**。两种情形分开报，
        因为对调用方是两个不同的 bug：
        - 标识在已归还集合里 ⇒ 重复归还（或拿着失效引用继续用）；
        - 标识从没发过 ⇒ 非本池块。
        """
        block = self._blocks.get(block_id)
        if block is not None:
            return block
        if block_id in self._retired:
            raise OwnershipViolation(
                f"块 {block_id} 已归还给底座（重复归还，或拿着失效引用继续用）。"
                f"按 I-1，这是池自己的账本拦下的，没有依赖厂商报错 —— "
                f"昇腾的裸释放对二次释放静默返回成功，指望厂商报在那一家上会完全失效。"
            )
        raise OwnershipViolation(
            f"块 {block_id} 不是本池发出的（账本里没有，也不在已归还集合里）。"
        )

    def require_live(self, block_id: int) -> Block:
        """取块并确认它不在隔离区。隔离是终态，不复用不释放。"""
        block = self.get(block_id)
        if block.state is BlockState.QUARANTINED:
            raise BlockQuarantined(
                f"块 {block_id} 在隔离区（原因：{self.reason_of(block_id) or '未记'}）。"
                f"隔离是终态：不复用、不释放、只上报。"
            )
        return block

    def __contains__(self, block_id: object) -> bool:
        return block_id in self._blocks

    def __len__(self) -> int:
        return len(self._blocks)

    def iter_blocks(self) -> Iterator[Block]:
        return iter(list(self._blocks.values()))

    def by_state(self, state: BlockState) -> tuple[Block, ...]:
        return tuple(b for b in self._blocks.values() if b.state is state)

    def bytes_in_state(self, state: BlockState) -> int:
        return sum(b.tier_bytes for b in self._blocks.values() if b.state is state)

    def allocated_by_purpose(self) -> dict[Purpose, int]:
        """按用途标签聚合的**在用**字节数（`query_budget` 的取数）。"""
        totals = {purpose: 0 for purpose in Purpose}
        for block in self._blocks.values():
            if block.state is BlockState.IN_USE:
                totals[block.purpose] += block.tier_bytes
        return totals

    # —— 分级空闲表 ——

    def push_free(self, block: Block) -> None:
        """把一块放进它所属档位的空闲表。只有 `可复用` 的块能进。"""
        if block.state is not BlockState.REUSABLE:
            raise OwnershipViolation(
                f"块 {block.block_id} 状态是 {block.state.value}，不是「可复用」，不得进空闲表。"
            )
        self._free.setdefault(block.tier_bytes, []).append(block.block_id)

    def pop_free(self, tier_bytes: int) -> Block | None:
        """从某档空闲表取一块可复用的。表空返回 `None`（不是报错 —— 取不到就去底座拿）。"""
        queue = self._free.get(tier_bytes)
        while queue:
            block_id = queue.pop(0)
            block = self._blocks.get(block_id)
            if block is None or block.state is not BlockState.REUSABLE:
                # 表里残留了不该在的标识：这本身是账本不一致的信号，跳过并继续找。
                continue
            return block
        return None

    def drop_free(self, block: Block) -> None:
        """把一块从空闲表里摘掉（压力回收 / 碎片整理要交回底座时用）。"""
        queue = self._free.get(block.tier_bytes)
        if queue and block.block_id in queue:
            queue.remove(block.block_id)

    def free_blocks(self, tier_bytes: int) -> tuple[Block, ...]:
        """某档空闲表里当前真正可复用的块。"""
        return tuple(
            b
            for b in (self._blocks.get(i) for i in self._free.get(tier_bytes, ()))
            if b is not None and b.state is BlockState.REUSABLE
        )

    def tier_free(self, tier_bytes: int) -> TierFree:
        blocks = self.free_blocks(tier_bytes)
        return TierFree(
            tier_bytes=tier_bytes,
            reusable_blocks=len(blocks),
            reusable_bytes=sum(b.tier_bytes for b in blocks),
        )

    # —— 终结与隔离 ——

    def retire(self, block: Block) -> None:
        """块已交回底座：从账本摘除，标识进已归还集合（不复用）。"""
        self._blocks.pop(block.block_id, None)
        self.drop_free(block)
        self._retired.add(block.block_id)
        block.state = BlockState.RETURNED
        block.ticket = None

    def quarantine(self, block: Block, reason: str, detail: str = "") -> None:
        """转隔离：从空闲表摘掉，留在账本里（隔离量要可查），状态改隔离。"""
        self.drop_free(block)
        block.state = BlockState.QUARANTINED
        block.ticket = None
        self._quarantine.append(
            QuarantineRecord(block_id=block.block_id, reason=reason, detail=detail)
        )

    def reason_of(self, block_id: int) -> str | None:
        for record in reversed(self._quarantine):
            if record.block_id == block_id:
                return record.reason
        return None

    def quarantine_reasons(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self._quarantine:
            counts[record.reason] = counts.get(record.reason, 0) + 1
        return counts

    def quarantined_views(self) -> tuple[BlockView, ...]:
        return tuple(b.view() for b in self.by_state(BlockState.QUARANTINED))
