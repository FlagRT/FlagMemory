"""池向下要用的后端能力面（协议声明，纯标准库、零框架依赖）。

这一层**只写协议，不写实现**。真后端（对接设备方向统一后端接口的那一层）是后续任务；
本仓当前只随附一个纯内存假后端 `flagmemory.caps.fake`，供单测与无卡环境用。

为什么只写协议：池是第三层（在厂商缓存分配器与框架预分配池之上），依据
`inbox/调研-四芯片显存能力对照与池抽象分级-20261103.md` 第一节 —— 池应当站在统一层接口上，
不站在厂商原语上。所以 `flagmemory/caps/**` 与 `flagmemory/pool/**` 里
**不许出现 torch / vLLM / 厂商运行时的 import**，由 `tools/check_layering.py` 静态把关。

### 两条沿用统一层的纪律（池不自建第二套）

1. **能力如实声明**：后端用 `declared_capabilities()` 如实报它有什么；没实测到的一律不写。
2. **未声明的能力被调用时报错，不得静默退化** —— 实现方应抛 `CapabilityNotDeclared`。
   池侧看到这个异常就按 `flagmemory.caps.degradation` 的降级路径走，**而不是**靠
   「调用返回了个空值」去猜。

### 命名口径

内存句柄沿用统一后端接口的五个公共字段 —— 句柄号 / 种类 / 后端名 / 设备序号 / 字节数
（见 `inbox/接口面诉求-统一内存句柄与显存池界面-交刘鸿彬-20261103.md` 第一节）。
两个待加字段（**用途标签** `purpose`、**所属流** `stream_id`）是 11-25 与设备方向联合冻结
那一份的内容，本文件按「可选、缺则按最保守处理」先占位，冻结后对齐改名。
**厂商指针不进句柄**（统一层已钉死的纪律，本协议同样不暴露裸指针）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .capabilities import Capability

__all__ = [
    "CapsError",
    "CapabilityNotDeclared",
    "EventNotRecorded",
    "DoubleFree",
    "MemoryInfo",
    "MemoryHandle",
    "DeviceBackend",
]


class CapsError(RuntimeError):
    """能力层的异常基类。"""


class CapabilityNotDeclared(CapsError):
    """调用了后端未声明的能力。

    存在的意义就是**不许静默退化**：池必须看到异常，才知道该切降级路径。
    """

    def __init__(self, cap: Capability, backend_name: str = "<unknown>") -> None:
        self.capability = cap
        self.backend_name = backend_name
        super().__init__(
            f"后端 {backend_name} 未声明能力 {cap.value}：按统一层纪律必须报错，不得静默退化；"
            f"调用方请改走 flagmemory.caps.degradation 给出的降级路径。"
        )


class EventNotRecorded(CapsError):
    """对未 record 的事件做了完成性判断。

    不变量二：寒武纪与平头哥的原生实现会对这种事件误报「已完成」，适配层必须自己拦住，
    绝不把那个 True 递给池。
    """


class DoubleFree(CapsError):
    """重复释放，或释放了非本层的句柄。

    不变量一：二次释放的厂商行为四家相反（昇腾静默成功、昆仑芯如实报错），
    所以本层靠句柄登记表自己记账拦截，不依赖厂商报错。
    """


@dataclass(frozen=True)
class MemoryInfo:
    """设备级显存三键查询的结果（L0，四家齐备）。

    三个数都是字节。`used + free == total` 不作强制断言：各家口径有差，
    池只把它当水位输入，记账基准另有分配器视角那一项（L1）。
    """

    total: int
    used: int
    free: int

    def __post_init__(self) -> None:
        for name in ("total", "used", "free"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"MemoryInfo.{name} 必须是 int，收到 {type(value).__name__}")
            if value < 0:
                raise ValueError(f"MemoryInfo.{name} 不能为负：{value}")


@dataclass(frozen=True)
class MemoryHandle:
    """统一内存句柄（L1，寒武纪 MLU590 未声明）。

    前五个字段 = 统一后端接口的五个公共字段。后两个是 11-25 联合冻结那份要加的，
    这里先占位：`purpose` 缺省 `None` 表示「未声明用途」，池按最保守处理
    （不回收、不复用）；`stream_id` 缺省 `None` 表示默认流。
    """

    handle_id: int        # 句柄号
    kind: str             # 种类
    backend: str          # 后端名
    device_index: int     # 设备序号
    nbytes: int           # 字节数
    purpose: str | None = None    # 用途标签（透传，语义归显存方向解释）
    stream_id: int | None = None  # 所属流标识（不是流对象，避免生命周期纠缠）


@runtime_checkable
class DeviceBackend(Protocol):
    """池向下要用的后端能力面。

    实现方须：
    - 在 `declared_capabilities()` 里如实声明；
    - 对未声明能力对应的方法抛 `CapabilityNotDeclared`；
    - **不得**替池自建设备上下文（`CONTEXT_LIFECYCLE` 属 L2，池一律不用）。
    """

    # —— 标识与声明 ——

    @property
    def name(self) -> str:
        """后端名（进句柄的那个「后端名」字段，同一口径）。"""
        ...

    def declared_capabilities(self) -> frozenset[Capability]:
        """如实声明：这个后端实测具备哪些能力。"""
        ...

    # —— L0：设备显存三键查询 ——

    def device_memory_info(self, device_index: int = 0) -> MemoryInfo:
        """总量 / 已用 / 空闲（字节）。池的水位输入。"""
        ...

    # —— L0：流 ——

    def current_stream(self, device_index: int = 0) -> Any:
        """取当前流。

        返回的流对象由池**强持有**（不变量三：流 + 弱引用 + 被回收 = 退出期段错误）。
        """
        ...

    def streams_equal(self, left: Any, right: Any) -> bool:
        """两个流是否同一条。各家流对象的 `==` 语义不一致，统一走这个口。"""
        ...

    # —— L0：事件 ——

    def create_event(self, device_index: int = 0) -> Any:
        """新建一个未 record 的事件。"""
        ...

    def record_event(self, event: Any, stream: Any) -> None:
        """在流上 record。

        不变量二：**未 record 的事件不许裸调 `query_event`**（寒武纪与平头哥会误报
        「已完成」）；`query_event` 的实现须自己拦住这种调用。
        """
        ...

    def query_event(self, event: Any) -> bool:
        """事件是否已完成。对未 record 的事件须报错，不得返回 True。"""
        ...

    def synchronize_event(self, event: Any, timeout_s: float) -> bool:
        """有界同步：等到完成返回 True，到点未完成返回 False。

        **超时不等于失败**（降级路径三）：返回 False 时池按可恢复态上报并继续轮询，
        不得阻塞等待。真有界只有昇腾（`TRUE_BOUNDED_SYNC`），其余三家这里是
        超时上报 / 轮询实现。
        """
        ...

    # —— L1（可选）：统一内存句柄 ——

    def alloc_handle(
        self,
        nbytes: int,
        device_index: int = 0,
        kind: str = "device",
        purpose: str | None = None,
        stream_id: int | None = None,
    ) -> MemoryHandle:
        """申请一块显存，返回句柄。未声明 `MEMORY_HANDLE` 时抛 `CapabilityNotDeclared`。"""
        ...

    def free_handle(self, handle: MemoryHandle) -> None:
        """成对释放。重复释放与非本层句柄必须报错（统一层已钉死的纪律）。"""
        ...
