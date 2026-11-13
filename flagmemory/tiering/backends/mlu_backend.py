# SPDX-License-Identifier: Apache-2.0
"""寒武纪 MLU 栈（MLU590 / NEUWARE 4.4.3 + torch_mlu）卸载适配层 → TierBackend 协议面接线层。

四栈第四栈（昇腾栈见 `npu_backend.py`、昆仑芯栈见 `xpu_backend.py`、平头哥栈见
`ppu_backend.py`，同构模板）：**协议面包装 + 注册路径打通**。与前三栈的一点差异：
本仓尚无 MLU 卸载 connector（`flagmemory/tiering/offload/` 下无 mlu 子包），故
`bind` 的默认真路径 = 本模块内嵌的最小 torch.mlu 拷贝通路 `MluTorchCopyBackend`
（守护线程 + 逐块 `copy_(non_blocking=True)` + 事件 record 后才入表，与
`offload/npu/npu_copy_backend.py` 的 `NpuDmaCopyBackend` 同构，仅命名空间换
`torch.mlu.*`、不进 vllm 进程故无 vllm logger）；connector 落地后 bind 默认
路径切换到 connector 侧拷贝后端，协议面与本模块其余部分不变。接线方向同 T-5：
外部把本适配层包成 TierBackend 注册给 manager，manager 本体反向不 import 本模
块；本模块对 torch / torch_mlu 的 import **延迟到运行注册 / bind / 探测时**
（主机单测环境无 torch_mlu，模块顶层必须零框架依赖）。

能力声明按栈实测事实填（登记处 = 分层存储设计 §十补记二（2026-11-10 MLU590
真卡取证）+ 2026-11-13 接线前事实钉死探针）：
- 批量搬运原语：**无** —— torch_mlu 的 Python 面（`torch.mlu.*`，与 torch.cuda
  同族）不暴露批量 memcpy 驱动 API，逐块 `copy_(non_blocking=True)` 即实现。
  声明与实测都登记 `has_batch_copy=False`，即已处于并如实登记降级态（3.5），
  不静默。
- 锁页：声明支持（torch_mlu 张量 `pin_memory()` 走 cnrt 主机锁页路线，
  2026-11-13 真卡实测 PASS）；实测 = 注册时真实试锁一次小张量，试不出按降级
  登记上报（T-4）。
- 一次搬运粒度：以 bind 时按 KV spec 推导的逐块字节数为准；未 bind 时用
  MLU590 实测分配粒度实例值 8 KiB（2026-11-10 地址间距法实测）。实例值非通用
  常数（T-9）：bind 后由推导值取代。

⚠️ 本栈事件面铁则（与平头哥 PPU 同列，2026-11-10 取证 + 2026-11-13 复核）：
**没 record 过的事件查询原生返回「已完成」**。本模块 `confirm` 即协议要求的
事件适配器：完成事件由拷贝通路的后台线程 record 后**才入事件表**，凭据不在
表内 ⇒ 一律 False（未完成），绝不把未 record 事件的原生 True 递出去。

⚠️ MLU 三条硬约束（2026-11-10 真卡取证，全部落为本模块设计约束，绝不触碰）：
1. `cnrtMemAllocFromPoolAsync` 在本档位（NEUWARE 4.4.3 + 驱动 6.2.29）疑似
   静默崩溃 —— 本模块一切设备内存走 torch 缓存分配器路径（torch_mlu 张量），
   任何路径不得调用该驱动 API；
2. fork 不可用（fork 后子进程驱动上下文死 cnCtxSetCurrent 100002）—— 本模块
   单进程内工作，拷贝通路是后台线程不是子进程；
3. IPC handle 只认裸 cnrtMalloc 内存（torch 缓存分配器内存导出报 632015）——
   本接线层不涉 IPC。

条目级统计：无引擎侧条目访问统计，`block_activity` 恒 None（协议允许，管理器
退用判据 1+2+3 并标降级，不静默当 0）。

真卡语义（T-6 / T-7）属后续增量：搬运流登记与账本时序由 manager 本体驱动，本
接线层只提供逐块拷贝与完成确认（事件适配器形态，不裸调厂商事件查询）。
"""

from __future__ import annotations

import queue
import threading
from typing import Any

from flagmemory.tiering.manager import (
    TierBackend,
    TierBackendCapability,
    TierError,
    TierUnavailable,
)

__all__ = [
    "MluTierBackend",
    "MluTorchCopyBackend",
    "MLU590_INSTANCE_BLOCK_BYTES",
    "register_mlu_tier_backend",
]

#: 未 bind 时的搬运粒度登记值 = MLU590 实测分配粒度实例值（2026-11-10 地址间距
#: 法实测 8 KiB）。实例值非通用常数（T-9）：bind 后由按 KV spec 推导的逐块字节
#: 数取代。
MLU590_INSTANCE_BLOCK_BYTES = 8 * 1024


class MluTorchCopyBackend:
    """MLU↔主机逐块拷贝的最小真通路（torch.mlu 命名空间；connector 前的过渡件）。

    与 `offload/npu/npu_copy_backend.py` 的 `NpuDmaCopyBackend` 同构：守护线程
    消费工作队列（主线程发起搬运不阻塞），完成事件 record 后才入表（该栈未
    record 事件原生误报已完成，见模块头铁则）。差异：命名空间 `torch.mlu.*`
    （torch_mlu，PrivateUse1）；无 vllm 依赖（本通路不进 vllm 进程，诊断走
    调用方）。

    刻意不在模块顶层 import torch（主机测试环境无 torch_mlu）：torch 一律在
    方法体内延迟 import。
    """

    def __init__(self) -> None:
        self._store_layers: list = []  # (gpu_tensor, cpu_tensor) pairs
        self._load_layers: list = []  # (cpu_tensor, gpu_tensor) pairs
        self._queue: "queue.SimpleQueue | None" = None
        self._thread: threading.Thread | None = None
        self._shutdown: bool = False

    def init(
        self,
        gpu_caches: dict,
        cpu_caches: dict,
        device: Any,
        load_stream: Any,
        store_stream: Any,
    ) -> None:
        assert list(gpu_caches.keys()) == list(cpu_caches.keys())
        for k in gpu_caches:
            g, c = gpu_caches[k], cpu_caches[k]
            assert g.shape[1:] == c.shape[1:], f"shape mismatch for {k}"
            self._store_layers.append((g, c))
            self._load_layers.append((c, g))

        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._copy_loop,
            args=(self._queue, device, load_stream, store_stream),
            daemon=True,
            name="mlu-kv-offload-copy",
        )
        self._thread.start()

    def launch_copy(
        self,
        src_blocks: list,
        dst_blocks: list,
        is_store: bool,
        event_idx: int,
        events_list: list,
    ) -> None:
        layers = self._store_layers if is_store else self._load_layers
        assert self._queue is not None
        self._queue.put(
            (src_blocks, dst_blocks, layers, is_store, event_idx, events_list)
        )

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self._queue is not None:
            self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    @staticmethod
    def _copy_loop(q, device, load_stream, store_stream) -> None:
        import torch  # 延迟：主机测试环境无 torch_mlu

        torch.mlu.set_device(device)
        while True:
            item = q.get()
            if item is None:
                return
            src_blocks, dst_blocks, layers, is_store, event_idx, events_list = item
            stream = store_stream if is_store else load_stream
            with torch.mlu.stream(stream):
                for src_t, dst_t in layers:
                    for b_src, b_dst in zip(src_blocks, dst_blocks):
                        dst_t[b_dst].copy_(src_t[b_src], non_blocking=True)
                event = torch.mlu.Event()
                event.record(stream)
            # record 之后才入表：凭据在表内的才是真发起过的事件（该栈未 record
            # 事件原生误报已完成，confirm 适配器只认表内凭据）
            events_list.append((event_idx, event))


class MluTierBackend:
    """把 MLU 逐块拷贝 / 完成确认包成 TierBackend 协议实现。

    - `copy_backend`：可注入。真卡运行时默认懒加载本模块内嵌的
      `MluTorchCopyBackend`（init / launch_copy / 事件上报形状）；主机测试注入
      同形状假件，不碰真拷贝通路。
    - `bind(...)`：把 gpu/cpu 侧块表交给拷贝通路初始化（真卡路径在此才
      import torch / torch_mlu）。
    - 完成凭据 ticket = 事件表的下标；`confirm` 即事件适配器：凭据尚未入表
      （= 事件未 record）⇒ 未完成（False），入表后按其状态判定 —— 该栈未
      record 事件原生误报已完成，绝不信表外查询。
    """

    backend_id: str

    def __init__(
        self,
        *,
        backend_id: str = "cambricon",
        granularity_bytes: int | None = None,
        copy_backend: Any | None = None,
    ) -> None:
        self.backend_id = backend_id
        self._granularity = (
            MLU590_INSTANCE_BLOCK_BYTES if granularity_bytes is None
            else granularity_bytes
        )
        self._copy_backend: Any | None = copy_backend
        self._bound = False
        #: bind 后按 KV spec 推导的逐块字节数（取代实例登记值）
        self._bound_block_bytes: int | None = None
        #: 完成事件表：[(event_idx, event)]，由拷贝通路的后台线程在 record 之后
        #: 追加（未 record 的事件绝不入表 —— 该栈原生误报已完成）
        self._events: list[tuple[int, Any]] = []
        self._event_idx = 0
        #: 级二侧已登记条目（entry_id -> block_id），drop_entry 的账面
        self._tier2_entries: dict[str, int] = {}
        #: 注册时实测锁页结果（None = 尚未 probe）
        self._probed_pinned: bool | None = None

    # ------------------------------------------------------------------
    # TierBackend 协议面
    # ------------------------------------------------------------------

    def declare_capabilities(self) -> TierBackendCapability:
        return TierBackendCapability(
            transfer_granularity_bytes=(
                self._bound_block_bytes or self._granularity
            ),
            supports_pinned=True,
            has_batch_copy=False,
        )

    def probe_capabilities(self) -> TierBackendCapability:
        if self._probed_pinned is None:
            self._probed_pinned = self._probe_pinned()
        return TierBackendCapability(
            transfer_granularity_bytes=(
                self._bound_block_bytes or self._granularity
            ),
            supports_pinned=self._probed_pinned,
            # MLU 栈 Python 面无批量 memcpy 驱动 API 是登记事实，不是探测失败
            # 的降级
            has_batch_copy=False,
        )

    def copy_out(
        self, entry_id: str, block_id: int, nbytes: int, *, stream_id: int
    ) -> Any:
        """级一 → 级二逐块拷贝（store 方向），返回完成凭据。失败抛异常。

        发起即把该条目的级二落点行（= 本次 gpu→cpu 的行号）登记进条目账 ——
        它是 `copy_in` 定位源行的唯一依据（2026-11-10 三栈同构核查钉死的口径，
        四栈同款同修）。"""
        self._require_bound()
        ticket = self._launch(block_id, block_id, is_store=True)
        self._tier2_entries[entry_id] = block_id
        return ticket

    def copy_in(
        self, entry_id: str, block_id: int, nbytes: int, *, stream_id: int
    ) -> Any:
        """级二 → 级一逐块拷贝（load 方向），返回完成凭据。失败抛异常。

        契约（T-5/T-7 编排语义，对齐昇腾/昆仑芯/平头哥已钉死口径）：**源行按
        条目的级二记录定位**（`copy_out` 发起时登记的落点行，或
        `register_entry` 预登记的行），**目标行 = 本次传入的 block_id** ——
        manager 换入时它是新分配块的行号，与源行是两套编号，不得拿 block_id
        同时当 src/dst。未知条目按 T-4 报错不静默，绝不退化为「block_id 当源
        行」。"""
        self._require_bound()
        src_row = self._tier2_entries.get(entry_id)
        if src_row is None:
            raise TierError(
                f"换入条目 {entry_id} 无级二落点记录：copy_in 源行必须按条目定位"
                "（copy_out 发起登记 / register_entry 预登记），未知条目不静默退化"
                "为 block_id 当源行（T-4）；请先对该条目发起换出或预登记。"
            )
        return self._launch(src_row, block_id, is_store=False)

    def confirm(self, ticket: Any) -> bool:
        """事件适配器（该栈事件面铁则）：只信 record 后入表的事件。

        凭据不在表内 ⇒ False（该栈未 record 事件原生查询返回 True，是实测
        误报，绝不外传）；入表 ⇒ 按事件状态判定，未完成 False 可轮询再问。
        """
        for idx, event in self._events:
            if idx == ticket:
                return bool(event.query())
        return False

    def block_activity(self, block_id: int):  # -> TierEntryStats | None
        """无引擎侧条目访问统计：恒 None，退用其余判据并标降级。"""
        return None

    def drop_entry(self, entry_id: str) -> None:
        """释放级二主机侧副本的账面（真释放在拷贝通路的宿主张量生命周期）。"""
        self._tier2_entries.pop(entry_id, None)

    # ------------------------------------------------------------------
    # 运行注册路径（真卡侧调用；此处才允许碰框架绑定模块）
    # ------------------------------------------------------------------

    def bind(
        self,
        gpu_caches: dict,
        cpu_caches: dict,
        device: Any,
        load_stream: Any,
        store_stream: Any,
        *,
        block_bytes: int | None = None,
    ) -> None:
        """初始化拷贝通路。真卡路径在调用点（不在模块顶层）import torch_mlu。"""
        if self._copy_backend is None:
            self._copy_backend = MluTorchCopyBackend()
        self._copy_backend.init(
            gpu_caches, cpu_caches, device, load_stream, store_stream
        )
        if block_bytes is not None:
            self._bound_block_bytes = block_bytes
        self._bound = True

    def register_entry(self, entry_id: str, block_id: int) -> None:
        """级二账本账面登记（真账本在 manager；此处供 drop_entry 对账）。"""
        self._tier2_entries[entry_id] = block_id

    def shutdown(self) -> None:
        if self._copy_backend is not None and self._bound:
            self._copy_backend.shutdown()
        self._bound = False

    # —— 内部 ——

    def _require_bound(self) -> None:
        if not self._bound:
            raise TierUnavailable(
                "寒武纪搬运适配层未 bind（无已初始化拷贝通路），按 T-4 报错"
                "不静默：先在真卡环境调 bind(gpu_caches, cpu_caches, ...) "
                "再发起搬运。"
            )

    def _launch(self, src_block: int, dst_block: int, *, is_store: bool) -> int:
        self._event_idx += 1
        ticket = self._event_idx
        self._copy_backend.launch_copy(
            [src_block], [dst_block], is_store, ticket, self._events
        )
        return ticket

    def _probe_pinned(self) -> bool:
        """实测锁页：真环境试锁一张小 CPU 张量（与 torch_mlu pin_memory 同路线）。

        探测不到（主机无 torch / torch_mlu、或 pin_memory 抛错）⇒ 返回 False，
        由注册流程按「声明锁页但探测不到」登记降级并上报，不静默（T-4）。
        """
        try:
            import torch  # noqa: PLC0415 — 刻意延迟：主机测试环境可无 torch
        except Exception:
            return False
        try:
            torch.zeros(16, dtype=torch.uint8).pin_memory()
            return True
        except Exception:
            return False


def register_mlu_tier_backend(
    manager: Any,
    pool: Any,
    *,
    backend_id: str | None = None,
    copy_backend: Any | None = None,
) -> Any:
    """把 MLU 适配层包成 TierBackend 注册给 manager（接线方向见模块头，T-5）。

    `pool` = manager 所在的池（manager 不公开池引用，接线层由调用方带一份）；
    `backend_id` 默认取池 `query_capability().backend_name`（注册口要求一致，
    不一致报错）。返回 `TierRegistration`（含声明/实测比对与降级项）。
    """
    if backend_id is None:
        backend_id = pool.query_capability().backend_name
    adapter = MluTierBackend(
        backend_id=backend_id, copy_backend=copy_backend
    )
    return manager.register_tier_backend(backend_id, adapter)


# 协议形状自检（runtime_checkable）：import 即可验，供测试与真卡路径共用。
assert isinstance(MluTierBackend(), TierBackend)
