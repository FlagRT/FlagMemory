# SPDX-License-Identifier: Apache-2.0
"""平头哥 PPU 栈（ZW810E / vendor vllm 0.24 四补丁）卸载适配层 → TierBackend 协议面接线层。

三栈第三栈（收官；昇腾栈见 `npu_backend.py`、昆仑芯栈见 `xpu_backend.py`，同构
模板）：**协议面包装 + 注册路径打通**，不重构 connector 内部；connector 经推理引擎
KV 外挂点（`kv_connector_module_path`）的加载方式不变。接线方向同 T-5：外部把本
适配层包成 TierBackend 注册给 manager，manager 本体反向不 import 本模块；本模块对
connector 的 import **延迟到运行注册 / bind 时**（connector 顶层绑 vllm / torch，
主机单测环境两者皆无，模块顶层必须零框架依赖）。

能力声明按栈实测事实填（登记处 = `flagmemory/tiering/offload/README.md` PPU 节
（2026-10-26 冒烟）+ `docs/design/分层存储概念与设计基础.md` §4.1/§4.4）：
- 批量搬运原语：**无** —— PPU HGGC 驱动不实现 `cuMemcpyBatchAsync`
  （err=998，补丁 P-2），connector 即按逐块 `copy_(non_blocking=True)` 实现
  （`ppu_mem_ops.py` 模块头）。声明与实测都登记 `has_batch_copy=False`，即已处于
  并如实登记降级态（3.5），不静默。
- 锁页：声明支持；该栈锁页是**三级降级链**（补丁 P-3：
  `cudaHostRegister` → `Tensor.pin_memory()` → 不锁页），声明按「支持」登记，
  注册时真实试锁一次小张量，**探测到哪一级按哪级登记**（探测到一二级 =
  `supports_pinned=True`；探到不锁页级 = False，按降级上报），不静默（T-4）。
  探测到的级别另存 `probed_pinned_level` 供诊断（不影响协议 bool 面）。
- 一次搬运粒度：以 bind 时按 KV spec 推导的逐块字节数为准；未 bind 时用 PPU 栈
  实测实例值 512 KiB/块（FL 版默认 block_size=8 tok → 512 KiB/块，README
  「PPU 栈已知边界」2026-10-26 取证）。实例值非通用常数（T-9）：bind 后由推导值取代。

⚠️ 本栈事件面铁则（设计 §4.1 第 4 条 / §4.4 第 6 条）：**没 record 过的事件查询
原生返回「已完成」**（该栈与寒武纪同列，2026-11-06 PPU 真卡实测复现）。本模块
`confirm` 即协议要求的**事件适配器**：完成事件由 copy backend 的后台线程 record
后**才入事件表**，凭据不在表内 ⇒ 一律 False（未完成），绝不把未 record 事件的
原生 True 递出去。

条目级统计：connector 无引擎侧条目访问统计，`block_activity` 恒 None（协议允许，
管理器退用判据 1+2+3 并标降级，不静默当 0）。

真卡语义（T-6 / T-7）属后续增量：搬运流登记与账本时序由 manager 本体驱动，本
接线层只提供逐块拷贝与完成确认（事件适配器形态，不裸调厂商事件查询）。
"""

from __future__ import annotations

from typing import Any

from flagmemory.tiering.manager import (
    TierBackend,
    TierBackendCapability,
    TierError,
    TierUnavailable,
)

__all__ = [
    "PpuTierBackend",
    "PPU_ZW810E_INSTANCE_BLOCK_BYTES",
    "register_ppu_tier_backend",
]

#: 未 bind 时的搬运粒度登记值 = PPU 栈实测实例值（FL 版默认 block_size=8 tok →
#: 512 KiB/块，README「PPU 栈已知边界」）。实例值非通用常数（T-9）：bind 后由按
#: KV spec 推导的逐块字节数取代。
PPU_ZW810E_INSTANCE_BLOCK_BYTES = 512 * 1024

#: 锁页探测级别（P-3 三级降级链的登记口径）
PINNED_LEVEL_HOST_REGISTER = "cudaHostRegister"
PINNED_LEVEL_PIN_MEMORY = "pin_memory"
PINNED_LEVEL_UNPINNED = "unpinned"


class PpuTierBackend:
    """把 PPU connector 的逐块拷贝 / 完成确认包成 TierBackend 协议实现。

    - `copy_backend`：可注入。真卡运行时默认懒加载 connector 的 `DmaCopyBackend`
      （init / launch_copy / 事件上报形状）；主机测试注入同形状假件，不碰真
      connector 内部。
    - `bind(...)`：把 gpu/cpu 侧块表交给 copy backend 初始化（真卡路径在此才
      import torch / vllm 相关模块）。
    - 完成凭据 ticket = connector 事件表的下标；`confirm` 即事件适配器：凭据
      尚未入表（= 事件未 record）⇒ 未完成（False），入表后按其状态判定 ——
      该栈未 record 事件原生误报已完成，绝不信表外查询。
    """

    backend_id: str

    def __init__(
        self,
        *,
        backend_id: str = "ppu",
        granularity_bytes: int | None = None,
        copy_backend: Any | None = None,
    ) -> None:
        self.backend_id = backend_id
        self._granularity = (
            PPU_ZW810E_INSTANCE_BLOCK_BYTES if granularity_bytes is None
            else granularity_bytes
        )
        self._copy_backend: Any | None = copy_backend
        self._bound = False
        #: bind 后按 KV spec 推导的逐块字节数（取代实例登记值）
        self._bound_block_bytes: int | None = None
        #: 完成事件表：[(event_idx, event)]，由 copy backend 的后台线程在
        #: record 之后追加（未 record 的事件绝不入表 —— 该栈原生误报已完成）
        self._events: list[tuple[int, Any]] = []
        self._event_idx = 0
        #: 级二侧已登记条目（entry_id -> block_id），drop_entry 的账面
        self._tier2_entries: dict[str, int] = {}
        #: 注册时实测锁页结果（None = 尚未 probe）
        self._probed_pinned: bool | None = None
        #: 探测到的锁页级别（P-3 链的哪一级；诊断用，不影响协议 bool 面）
        self.probed_pinned_level: str | None = None

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
            # PPU 栈无批量原语（cuMemcpyBatchAsync err=998，P-2）是登记事实，
            # 不是探测失败的降级
            has_batch_copy=False,
        )

    def copy_out(
        self, entry_id: str, block_id: int, nbytes: int, *, stream_id: int
    ) -> Any:
        """级一 → 级二逐块拷贝（store 方向），返回完成凭据。失败抛异常。

        发起即把该条目的级二落点行（= 本次 gpu→cpu 的行号）登记进条目账 ——
        它是 `copy_in` 定位源行的唯一依据（2026-11-10 修复：与昇腾 b3fa22b /
        昆仑芯 f793a56 同款缺陷在 PPU 栈同构中招，同款修法）。"""
        self._require_bound()
        ticket = self._launch(block_id, block_id, is_store=True)
        self._tier2_entries[entry_id] = block_id
        return ticket

    def copy_in(
        self, entry_id: str, block_id: int, nbytes: int, *, stream_id: int
    ) -> Any:
        """级二 → 级一逐块拷贝（load 方向），返回完成凭据。失败抛异常。

        契约（T-5/T-7 编排语义，对齐昇腾/昆仑芯已钉死口径）：**源行按条目的
        级二记录定位**（`copy_out` 发起时登记的落点行，或 `register_entry`
        预登记的行），**目标行 = 本次传入的 block_id** —— manager 换入时它是
        新分配块的行号，与源行是两套编号，不得拿 block_id 同时当 src/dst。
        未知条目按 T-4 报错不静默，绝不退化为「block_id 当源行」。"""
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
        """事件适配器（设计 §4.1 第 4 条）：只信 record 后入表的事件。

        凭据不在表内 ⇒ False（该栈未 record 事件原生查询返回 True，是实测
        误报，绝不外传）；入表 ⇒ 按事件状态判定，未完成 False 可轮询再问。
        """
        for idx, event in self._events:
            if idx == ticket:
                return bool(event.query())
        return False

    def block_activity(self, block_id: int):  # -> TierEntryStats | None
        """connector 无引擎侧条目访问统计：恒 None，退用其余判据并标降级。"""
        return None

    def drop_entry(self, entry_id: str) -> None:
        """释放级二主机侧副本的账面（真释放在 connector 侧，由宿主生命周期管理）。"""
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
        """初始化 connector 的拷贝通路。真卡路径在调用点（不在模块顶层）import。"""
        if self._copy_backend is None:
            from flagmemory.tiering.offload.ppu.ppu_copy_backend import (
                DmaCopyBackend,
            )

            self._copy_backend = DmaCopyBackend()
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
                "PPU 搬运适配层未 bind（无已初始化拷贝通路），按 T-4 报错"
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
        """实测锁页（P-3 三级降级链逐级探测，探到哪级按哪级登记）。

        ① `cudaHostRegister`（connector `pin_tensor` 同路线，能真锁且保
        张量身份）→ ② 失败退 `Tensor.pin_memory()`（锁页副本）→ ③ 再失败
        不锁页。①② = `supports_pinned=True`，③ = False 按降级登记上报，
        级别名记入 `probed_pinned_level`，不静默（T-4）。
        """
        self.probed_pinned_level = PINNED_LEVEL_UNPINNED
        try:
            import torch  # noqa: PLC0415 — 刻意延迟：主机测试环境可无 torch
        except Exception:
            return False
        try:
            t = torch.zeros(16, dtype=torch.uint8)
            torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.nbytes, 0)
            self.probed_pinned_level = PINNED_LEVEL_HOST_REGISTER
            return True
        except Exception:
            pass
        try:
            torch.zeros(16, dtype=torch.uint8).pin_memory()
            self.probed_pinned_level = PINNED_LEVEL_PIN_MEMORY
            return True
        except Exception:
            return False


def register_ppu_tier_backend(
    manager: Any,
    pool: Any,
    *,
    backend_id: str | None = None,
    copy_backend: Any | None = None,
) -> Any:
    """把 PPU 适配层包成 TierBackend 注册给 manager（接线方向见模块头，T-5）。

    `pool` = manager 所在的池（manager 不公开池引用，接线层由调用方带一份）；
    `backend_id` 默认取池 `query_capability().backend_name`（注册口要求一致，
    不一致报错）。返回 `TierRegistration`（含声明/实测比对与降级项）。
    """
    if backend_id is None:
        backend_id = pool.query_capability().backend_name
    adapter = PpuTierBackend(
        backend_id=backend_id, copy_backend=copy_backend
    )
    return manager.register_tier_backend(backend_id, adapter)


# 协议形状自检（runtime_checkable）：import 即可验，供测试与真卡路径共用。
assert isinstance(PpuTierBackend(), TierBackend)
