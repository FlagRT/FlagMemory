"""池用例用的假后端与假底座。刻意不 import torch / vLLM：池本体的用例**零真卡依赖**。

放在独立模块而不是 `conftest.py` 里，是为了让用例文件能直接 `from fakes import ...` ——
`conftest.py` 由 pytest 以特殊方式装载，不该被用例当普通模块 import。

五个测试替身，各加一件池要分支的事实：

| 替身 | 加了什么 | 用来验什么 |
|---|---|---|
| `FakeBaseAllocator` | 张量级底座（取/还 + 容量上限 + 可造释放失败） | 张量级形态、容量不足报错、底座释放报错转隔离 |
| `CompletingBackend` | record 即完成 | 屏障放行、复用、回收、碎片整理 |
| `GranularityBackend` | `allocation_granularity` | 档位表由后端查询推导（昇腾 / 寒武纪 / 平头哥 那一档） |
| `NativeOrderedBackend` | 原生流有序分配原语 | 能力标记为真时走原生路，对外语义不变 |
| `ObservingBackend` | 上下文只读观测 | 归属校验三项齐全 vs 降级为未校验 |

没有 `allocation_granularity` 的后端（= 昆仑芯 P800 那一格，两边都查不到粒度）必须由用例
显式给 `tier_spec` —— 这正是「池不编粒度常数」那条约束在用例里的样子。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from flagmemory.caps.fake import FakeBackend, FakeEvent, FakeStream

__all__ = [
    "TEST_BASE_BYTES",
    "TEST_STEP_BYTES",
    "TEST_TIER_COUNT",
    "FakeBaseAllocator",
    "CompletingBackend",
    "GranularityBackend",
    "NativeOrderedBackend",
    "ObservingBackend",
]

#: 用例统一的档位基数与步长。是**用例的配置项**，不是池里的常数 ——
#: 池本体里没有任何粒度数字，所以这两个数只能从外面喂进来。
TEST_BASE_BYTES = 512 << 10   # 512 KiB（实例：平头哥那栈的默认块大小是这个量级）
TEST_STEP_BYTES = 4 << 10     # 4 KiB
TEST_TIER_COUNT = 6


@dataclass
class FakeBaseAllocator:
    """张量级形态的底座：满足 `flagmemory.pool.base.BaseAllocator`。

    真实现在框架绑定的适配层里（要 import 框架取空张量），池本体不许碰 —— 所以这里的
    「底座引用」就是个不透明元组，正好证明池不解释底座、不碰裸指针（`I-9`）。
    """

    capacity_bytes: int = 64 << 20
    used_bytes: int = 0
    live: dict[int, int] = field(default_factory=dict, repr=False)
    #: 置真则 release 一律报错，用来验「底座释放报错 → 转隔离并上报」
    fail_release: bool = False
    _next: int = field(default=1, repr=False)

    def allocate(self, nbytes: int, device_index: int = 0) -> Any:
        if nbytes > self.capacity_bytes - self.used_bytes:
            raise MemoryError(
                f"底座容量不足：请求 {nbytes}，剩余 {self.capacity_bytes - self.used_bytes}"
            )
        ref = ("base", self._next, nbytes)
        self.live[self._next] = nbytes
        self._next += 1
        self.used_bytes += nbytes
        return ref

    def release(self, base_ref: Any) -> None:
        if self.fail_release:
            raise RuntimeError("底座释放失败（用例刻意造的）")
        key = base_ref[1]
        if key not in self.live:
            raise RuntimeError(f"底座引用 {base_ref} 不在底座的在世表里")
        self.used_bytes -= self.live.pop(key)


class CompletingBackend(FakeBackend):
    """record 之后事件立刻算完成。屏障放行路径的用例用它。"""

    def record_event(self, event: FakeEvent, stream: FakeStream) -> None:
        super().record_event(event, stream)
        event.completed = True


class GranularityBackend(CompletingBackend):
    """带分配粒度查询的后端（昇腾 / 寒武纪 / 平头哥 三家各有「最小 / 推荐」两档）。"""

    min_granularity: int = TEST_STEP_BYTES
    recommended_granularity: int = TEST_BASE_BYTES

    def allocation_granularity(self, device_index: int = 0) -> tuple[int, int]:
        return (self.min_granularity, self.recommended_granularity)


class NativeOrderedBackend(CompletingBackend):
    """声明了原生流有序分配原语的后端（优化路径）。

    记下每次取块 / 还块所在的流，用来验「原生路也是按流分配，且对外语义与自管路相同」。
    """

    @property
    def native_log(self) -> list[tuple[str, int, int | None]]:
        log = getattr(self, "_native_log", None)
        if log is None:
            log = []
            setattr(self, "_native_log", log)
        return log

    def alloc_from_pool_async(self, nbytes: int, stream: Any, device_index: int = 0) -> Any:
        self.native_log.append(("alloc", nbytes, getattr(stream, "stream_id", None)))
        return ("native", len(self.native_log), nbytes)

    def free_async(self, base_ref: Any, stream: Any) -> None:
        self.native_log.append(("free", base_ref[2], getattr(stream, "stream_id", None)))


class ObservingBackend(CompletingBackend):
    """带设备上下文只读观测的后端。观测值固定，归属校验三项才能齐。"""

    def observe_context(self, device_index: int = 0) -> Any:
        return ("ctx", device_index)
