# SPDX-License-Identifier: Apache-2.0
"""昇腾栈（Ascend 910C / vllm-ascend）卸载适配层 → TierBackend 协议面接线层。

设计第四节第 5 条：现状三套卸载 adapter 都不经池（4.2「缺」第⑤条）；改造排期不属
设计冻结范围，冻结的是改造后语义（T-6 搬运流经池登记、T-7 账本先行/确认后交块）。
本模块 = 三栈中昇腾栈的第一步：**协议面包装 + 注册路径打通**，不重构 connector 内部；
connector 原有经推理引擎 KV 外挂点（`kv_connector_module_path`）的加载方式不变。

接线方向（T-5）：**外部把适配层包成 TierBackend 注册给 manager** —— 本模块可以
import `flagmemory.tiering.offload.npu` 的实现，manager 本体反向不 import 本模块。
且本模块对 connector 的 import **延迟到运行注册 / bind 时**：connector 顶层 import
vllm / torch.npu（框架绑定），主机单测环境两者皆无，模块顶层必须零框架依赖。

能力声明按栈实测事实填（登记处 = `flagmemory/tiering/offload/README.md` +
`offload/npu/RUNBOOK.md`，2026-10-14 W2/W3 闭环）：
- 批量搬运原语：**无** —— Ascend 未向 Python 暴露批量 memcpy 驱动 API，connector
  即按逐块 `copy_(non_blocking=True)` 实现（`npu_copy_backend.py` 模块头）。
  声明与实测都登记 `has_batch_copy=False`，即已处于并如实登记降级态（3.5），
  不静默。
- 锁页：声明支持（`npu_mem_ops.pin_or_fallback` 走 torch_npu `pin_memory()`，
  不可用自动退 pageable）；实测 = 注册时真实试锁一次小张量，试不出按降级登记
  上报（T-4）。
- 一次搬运粒度：以 bind 时 connector 按 KV spec 推导的逐块字节数为准；未 bind
  时用 W2 实测实例值（OneRec-8B @910C：8 GiB / ≈3700 块 ≈ 2.2 MiB/块，RUNBOOK §3
  检查点 1）。实例值会随模型变，bind 后即被实测值取代。

条目级统计：connector 无引擎侧条目访问统计，`block_activity` 恒 None（协议允许，
管理器退用判据 1+2+3 并标降级，不静默当 0）。

真卡语义（T-6 / T-7）属后续增量：搬运流登记与账本时序由 manager 本体驱动，
本接线层只提供逐块拷贝与完成确认（事件适配器形态，不裸调厂商事件查询）。
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
    "NpuTierBackend",
    "NPU_910C_INSTANCE_BLOCK_BYTES",
    "register_npu_tier_backend",
]

#: 未 bind 时的搬运粒度登记值 = W2 实测实例值（OneRec-8B @910C，≈2.2 MiB/块）。
#: 实例值非通用常数（T-9）：bind 后由 connector 推导的逐块字节数取代。
NPU_910C_INSTANCE_BLOCK_BYTES = 2_306_867  # ≈ 2.2 MiB（8 GiB / 3700 块）


class NpuTierBackend:
    """把昇腾 connector 的逐块拷贝 / 完成确认包成 TierBackend 协议实现。

    - `copy_backend`：可注入。真卡运行时默认懒加载 connector 的
      `NpuDmaCopyBackend`（init / launch_copy / 事件上报形状）；主机测试注入
      同形状假件，不碰真 connector 内部。
    - `bind(...)`：把 gpu/cpu 侧块表交给 copy backend 初始化（真卡路径在此才
      import torch.npu / vllm 相关模块）。
    - 完成凭据 ticket = connector 事件表的下标；`confirm` 即协议占位的事件
      适配器：事件尚未入表 ⇒ 未完成（False），入表后按其状态判定。
    """

    backend_id: str

    def __init__(
        self,
        *,
        backend_id: str = "ascend",
        granularity_bytes: int | None = None,
        copy_backend: Any | None = None,
    ) -> None:
        self.backend_id = backend_id
        self._granularity = (
            NPU_910C_INSTANCE_BLOCK_BYTES if granularity_bytes is None
            else granularity_bytes
        )
        self._copy_backend: Any | None = copy_backend
        self._bound = False
        #: bind 后由 connector 推导的逐块字节数（取代实例登记值）
        self._bound_block_bytes: int | None = None
        #: 完成事件表：[(event_idx, event)]，由 copy backend 的后台线程追加
        self._events: list[tuple[int, Any]] = []
        self._event_idx = 0
        #: 级二侧条目账（entry_id -> 级二落点行）：copy_out 发起时登记（外部
        #: register_entry 可预登记，copy_out 以实际发起覆盖）；copy_in 按它定位
        #: 源行，drop_entry 撤账
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
            # Ascend 无批量原语是登记事实，不是探测失败的降级
            has_batch_copy=False,
        )

    def copy_out(
        self, entry_id: str, block_id: int, nbytes: int, *, stream_id: int
    ) -> Any:
        """级一 → 级二逐块拷贝（store 方向），返回完成凭据。失败抛异常。

        发起即把该条目的级二落点行（= 本次 gpu→cpu 的行号）登记进条目账 ——
        它是 `copy_in` 定位源行的唯一依据。"""
        self._require_bound()
        ticket = self._launch(block_id, block_id, is_store=True)
        self._tier2_entries[entry_id] = block_id
        return ticket

    def copy_in(
        self, entry_id: str, block_id: int, nbytes: int, *, stream_id: int
    ) -> Any:
        """级二 → 级一逐块拷贝（load 方向），返回完成凭据。失败抛异常。

        契约（T-5/T-7 编排语义，2026-11-08 910C 真卡冒烟钉死）：**源行按条目
        的级二记录定位**（`copy_out` 发起时登记的落点行，或 `register_entry`
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
        """事件适配器（T-6 第 4 步口径）：不裸调厂商事件查询之外的东西。"""
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
            from flagmemory.tiering.offload.npu.npu_copy_backend import (
                NpuDmaCopyBackend,
            )

            self._copy_backend = NpuDmaCopyBackend()
        self._copy_backend.init(
            gpu_caches, cpu_caches, device, load_stream, store_stream
        )
        if block_bytes is not None:
            self._bound_block_bytes = block_bytes
        self._bound = True

    def register_entry(self, entry_id: str, block_id: int) -> None:
        """外部预登记条目的级二落点行（真账本在 manager）。

        供调用方在 copy_out 之外先行登记（如宿主接管既有级二副本）；随后同
        条目的 copy_out 会以实际发起落点覆盖本登记。"""
        self._tier2_entries[entry_id] = block_id

    def shutdown(self) -> None:
        if self._copy_backend is not None and self._bound:
            self._copy_backend.shutdown()
        self._bound = False

    # —— 内部 ——

    def _require_bound(self) -> None:
        if not self._bound:
            raise TierUnavailable(
                "昇腾搬运适配层未 bind（无已初始化拷贝通路），按 T-4 报错不静默："
                "先在真卡环境调 bind(gpu_caches, cpu_caches, ...) 再发起搬运。"
            )

    def _launch(self, src_block: int, dst_block: int, *, is_store: bool) -> int:
        self._event_idx += 1
        ticket = self._event_idx
        self._copy_backend.launch_copy(
            [src_block], [dst_block], is_store, ticket, self._events
        )
        return ticket

    def _probe_pinned(self) -> bool:
        """实测锁页：真环境试锁一张小 CPU 张量（与 pin_or_fallback 同路线）。

        探测不到（主机无 torch / torch_npu、或 pin_memory 抛错）⇒ 返回 False，
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


def register_npu_tier_backend(
    manager: Any,
    pool: Any,
    *,
    backend_id: str | None = None,
    copy_backend: Any | None = None,
) -> Any:
    """把昇腾适配层包成 TierBackend 注册给 manager（接线方向见模块头，T-5）。

    `pool` = manager 所在的池（manager 不公开池引用，接线层由调用方带一份）；
    `backend_id` 默认取池 `query_capability().backend_name`（注册口要求一致，
    不一致报错）。返回 `TierRegistration`（含声明/实测比对与降级项）。
    """
    if backend_id is None:
        backend_id = pool.query_capability().backend_name
    adapter = NpuTierBackend(
        backend_id=backend_id, copy_backend=copy_backend
    )
    return manager.register_tier_backend(backend_id, adapter)


# 协议形状自检（runtime_checkable）：import 即可验，供测试与真卡路径共用。
assert isinstance(NpuTierBackend(), TierBackend)
