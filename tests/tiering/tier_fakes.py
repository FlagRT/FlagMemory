"""分层存储用例的测试替身：主机内存字典模拟级二的假搬运后端。零真卡依赖。

刻意不 import torch / vLLM（T-5 的可测面）：管理器本体只认 `TierBackend` 协议，
这里给出一个协议的完整假实现，用来钉 T-1..T-11 的可主机验证部分。

可注入的故障：
- `fail_copy_out` / `fail_copy_in`：拷贝中途失败（可先把半个条目写进 storage 再抛，
  用来验「条目不落账本 ⇒ 级二查不到」的 T-7 时序）；
- `never_confirm`：完成确认永远不通过（轮询耗尽 ⇒ 按搬运失败处置）；
- `stats`：判据 4/5 的条目级统计（不给 ⇒ None，退用其余判据并标降级）。
"""

from __future__ import annotations

from typing import Any

from flagmemory.tiering.manager import TierBackendCapability, TierEntryStats

__all__ = [
    "FakeTierBackend",
    "GRANULARITY_BYTES",
]

#: 用例统一的搬运粒度（= 一次搬运粒度声明）。是用例配置项，不是管理器常数（T-9）。
GRANULARITY_BYTES = 64 << 10  # 64 KiB


class FakeTierBackend:
    """级二 = 一个主机内存字典。backend_id 与池侧假后端的 name 对齐。"""

    def __init__(
        self,
        *,
        backend_id: str = "kunlunxin-like",
        granularity_bytes: int = GRANULARITY_BYTES,
        declared_pinned: bool = True,
        declared_batch: bool = True,
        probe_pinned: bool | None = None,
        probe_batch: bool | None = None,
        fail_copy_out: bool = False,
        fail_copy_in: bool = False,
        never_confirm: bool = False,
    ) -> None:
        self.backend_id = backend_id
        self._granularity = granularity_bytes
        self._declared = TierBackendCapability(
            transfer_granularity_bytes=granularity_bytes,
            supports_pinned=declared_pinned,
            has_batch_copy=declared_batch,
        )
        self._probe = TierBackendCapability(
            transfer_granularity_bytes=granularity_bytes,
            supports_pinned=declared_pinned if probe_pinned is None else probe_pinned,
            has_batch_copy=declared_batch if probe_batch is None else probe_batch,
        )
        self.fail_copy_out = fail_copy_out
        self.fail_copy_in = fail_copy_in
        self.never_confirm = never_confirm
        #: 级二主机侧存储（entry_id -> 内容字节）
        self.storage: dict[str, bytes] = {}
        #: 级一内容模拟（block_id -> 内容字节），由用例预先放好，copy_out 从这取
        self.device_blocks: dict[int, bytes] = {}
        #: 判据 4/5 的条目级统计（block_id -> stats）；不在这个字典里 = 取不到
        self.stats: dict[int, TierEntryStats] = {}
        #: 逐块拷贝调用记录（(方向, entry_id, block_id, nbytes, stream_id)）
        self.copy_log: list[tuple[str, str, int, int, int]] = []
        self.dropped: list[str] = []

    # —— 协议面 ——

    def declare_capabilities(self) -> TierBackendCapability:
        return self._declared

    def probe_capabilities(self) -> TierBackendCapability:
        return self._probe

    def copy_out(
        self, entry_id: str, block_id: int, nbytes: int, *, stream_id: int
    ) -> Any:
        self.copy_log.append(("out", entry_id, block_id, nbytes, stream_id))
        if block_id not in self.device_blocks:
            raise RuntimeError(f"级一没有块 {block_id} 的内容（用例没放）")
        if self.fail_copy_out:
            # 「拷贝中途失败」：半个条目已经写进主机内存，然后抛——
            # 用来验管理器不落账本 ⇒ 级二查不到（T-7 账本先行）
            self.storage[entry_id] = self.device_blocks[block_id][: nbytes // 2]
            raise RuntimeError(f"注入：copy_out 在块 {block_id} 中途失败")
        self.storage[entry_id] = self.device_blocks[block_id][:nbytes]
        return object()

    def copy_in(
        self, entry_id: str, block_id: int, nbytes: int, *, stream_id: int
    ) -> Any:
        self.copy_log.append(("in", entry_id, block_id, nbytes, stream_id))
        if entry_id not in self.storage:
            raise RuntimeError(f"级二没有条目 {entry_id}")
        if self.fail_copy_in:
            raise RuntimeError(f"注入：copy_in 到块 {block_id} 中途失败")
        self.device_blocks[block_id] = self.storage[entry_id][:nbytes]
        return object()

    def confirm(self, ticket: Any) -> bool:
        return not self.never_confirm

    def block_activity(self, block_id: int) -> TierEntryStats | None:
        return self.stats.get(block_id)

    def drop_entry(self, entry_id: str) -> None:
        self.dropped.append(entry_id)
        self.storage.pop(entry_id, None)
