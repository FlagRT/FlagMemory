"""池向下要用的四个可选协议口（纯标准库，零框架依赖）。

`flagmemory.caps.backend.DeviceBackend` 是池唯一的下行依赖（设计文档 1.4）。但有四件事
那份协议里没有、而池确实需要，且**都不是四家齐备的 L0 能力**，所以一律按「可选协议 +
运行时检测 + 缺则走已登记的降级路径」处理，不往 `DeviceBackend` 里硬加：

| 协议 | 池拿它干什么 | 缺了怎么办 |
|---|---|---|
| `BaseAllocator` | 张量级形态的底座（取/还一块） | 建池时报错，不静默换形态（`I-10`） |
| `GranularitySource` | 推档位表用的「最小 / 推荐」两档粒度 | 要调用方给档位配置项，两者都无则报错（绝不编常数） |
| `StreamOrderedAllocator` | 原生流有序分配/释放原语 | 走自管路（池自己记流序 + 事件屏障），这是**基线** |
| `CrossStreamProtection` | 跨流内存保护的登记 | 池自己记跨流借用集并上报降级态（设计文档 2.4） |

为什么用 `Protocol` + `isinstance` 而不是 `hasattr`：协议是可被类型检查器核的声明，
`hasattr` 不是；且四个口都要进 `flagmemory/interfaces/` 的存根，必须有名字。

**池不碰裸指针**（`I-9`）：`BaseAllocator` 换手的是「底座引用」—— 张量级形态下是底座张量
对象，句柄形态下是统一层的内存句柄。池只转交，不解释，不假设它底下是裸内存。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = [
    "BaseAllocator",
    "GranularitySource",
    "StreamOrderedAllocator",
    "CrossStreamProtection",
    "ContextObserver",
]


@runtime_checkable
class BaseAllocator(Protocol):
    """张量级形态的底座：池向它一次取一块，不再往下看。

    实现方在**框架绑定的可选适配层**（`flagmemory.tiering.offload.*`）或调用方自己那边 ——
    绝不在池本体里，因为取空张量要 import 框架，而池本体不许（分层门禁规则 1）。
    """

    def allocate(self, nbytes: int, device_index: int = 0) -> Any:
        """取一块至少 `nbytes` 字节的底座引用。取不到应抛 `MemoryError`。"""
        ...

    def release(self, base_ref: Any) -> None:
        """把底座引用交回。重复交回应报错 —— 但池不依赖它报（`I-1`，池自己记账）。"""
        ...


@runtime_checkable
class GranularitySource(Protocol):
    """分配粒度查询：档位表的推导输入。

    昇腾 910C / 寒武纪 MLU590 / 平头哥 PPU 三家各有「最小 / 推荐」两档粒度查询接口；
    **昆仑芯 P800 两边都查不到**（原生设备属性里无粒度项、头文件里无对齐宏），
    所以那一家的粒度基数要么实测、要么由调用方给配置项 —— 池不编常数。
    """

    def allocation_granularity(self, device_index: int = 0) -> tuple[int, int]:
        """返回 `(最小粒度, 推荐粒度)`，单位字节，两者都必须为正。"""
        ...


@runtime_checkable
class StreamOrderedAllocator(Protocol):
    """原生流有序分配原语（优化路径，不是基线）。

    四家分四档（只读符号探查结论，登记处 = 指挥台仓
    `inbox/调研-四芯片SDK显存原语面补探-20261103.md`）：

    - **寒武纪 MLU590**：有，运行时层与驱动层双层符号齐备；
    - **平头哥 PPU**：有（整套是英伟达接口的改名克隆），但头文件把该能力挂在设备属性
      门禁上，属性实际值未取证；
    - **昇腾 910C**：要较新的 CANN 大版本才有（池符号 7 个），老版本完全没有（池符号 0），
      同一机房里两种版本并存，所以这是**按机器而不是按芯片**的事实；
    - **昆仑芯 P800**：没有。驱动层整族缺失，运行时层那套异步接口是**同步实现的空壳**，
      绝不能当原生实现接进来。

    因此池的对外语义是「按流分配 + 后端能力标记」：标记为假就走自管路（**无条件可用的
    基线**），为真才包这个协议做优化路径。两条路对上层**暴露同一套接口**。
    """

    def alloc_from_pool_async(self, nbytes: int, stream: Any, device_index: int = 0) -> Any:
        """在流上做流有序分配，返回底座引用。"""
        ...

    def free_async(self, base_ref: Any, stream: Any) -> None:
        """在流上做流有序释放。"""
        ...


@runtime_checkable
class CrossStreamProtection(Protocol):
    """跨流内存保护的登记口（L1 能力 `CROSS_STREAM_PROTECTION` 的方法面）。

    声明了该能力的三家（昇腾 910C / 昆仑芯 P800 / 平头哥 PPU）经这个口登记；
    寒武纪 MLU590 未声明，池改为自己记跨流借用集、对借用集里每条流各取一次事件凭据，
    并**显式上报降级态**（设计文档 2.4）。两条路径的对外语义完全相同。
    """

    def record_stream(self, base_ref: Any, stream: Any) -> None:
        """登记「这块将被这条流读写」。"""
        ...


@runtime_checkable
class ContextObserver(Protocol):
    """设备上下文**只读**观测（L1 能力 `CONTEXT_OBSERVE` 的方法面）。

    只读是硬的：上下文生命周期（建 / 切 / 销）属 L2，池一律不用（`I-4`）—— 昆仑芯平台
    只允许一个上下文，第二次创建直接失败，抢建会让框架起不来。所以这个协议里
    **只有一个取当前上下文标识的口**，没有任何改动口。

    方法面尚未与设备方向联合冻结（那是 11-25 那份接口文件的事），所以池这边按
    「有就用、没有就把归属校验的上下文那一项降级为未校验并上报」处理，
    绝不把未校验当成通过。
    """

    def observe_context(self, device_index: int = 0) -> Any:
        """返回当前设备上下文的标识（可比较即可，池不解释它的内容）。"""
        ...
