"""能力枚举 + 能力分级 + 池的用法策略（纯标准库，零框架依赖）。

分级事实照抄 `inbox/调研-四芯片显存能力对照与池抽象分级-20261103.md` 第五节 5.1，
**本文件不重新论证**，只把那张表变成池可以分支的数据：

    L0  四家齐备（昇腾 910C / 昆仑芯 P800 / 寒武纪 MLU590 / 平头哥 PPU）
        设备级显存三键查询、流、事件、有界同步（语义有差）、图捕获、探针级恢复、
        多卡、流优先级范围可读。**池的核心路径只许用 L0。**
    L1  三家有、缺寒武纪 MLU590
        统一内存句柄（申请/释放）、分配器视角已分配量、跨流内存保护（record_stream）、
        设备上下文只读观测。走能力分支，缺则降级。
    L2  两家（昇腾 / 平头哥）
        设备上下文生命周期 ⇒ 池一律不用（池绝不自建上下文）。
    L3  一家（仅昇腾）
        厂商数值错误码表、设备级真重建 ⇒ 池一律不用，只作诊断增强。

分级是按**声明该能力的芯片家数**分的，不等于池的用法。所以每项能力另带一个
`PoolUsage`：`TRUE_BOUNDED_SYNC`（真有界同步）按家数属 L3（调研 2.3 第 1 组事实：
只有昇腾是真有界，其余三家的「有界」= 超时上报/轮询），但它的用法是**降级分支**，
不是「只作诊断增强」—— 若只看级别就会把它误判成池不该用的东西。
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType

__all__ = [
    "Capability",
    "CapabilityLevel",
    "PoolUsage",
    "CAPABILITY_LEVEL",
    "CAPABILITY_USAGE",
    "LEVEL_CAPABILITIES",
    "L0_CAPABILITIES",
    "L1_CAPABILITIES",
    "L2_CAPABILITIES",
    "L3_CAPABILITIES",
    "POOL_INVARIANTS",
    "capability_level",
    "pool_usage",
]


class CapabilityLevel(Enum):
    """按「四款在研国产芯片里有几家声明了这项能力」分的级。"""

    L0 = 0  # 四家齐备
    L1 = 1  # 三家有，缺寒武纪 MLU590
    L2 = 2  # 两家（昇腾 / 平头哥）
    L3 = 3  # 一家（仅昇腾）

    @property
    def families(self) -> int:
        """声明该级能力的芯片家数。"""
        return 4 - self.value


class PoolUsage(Enum):
    """池对这项能力的用法（与级别分开写，级别只说家数）。"""

    CORE_PATH = "core_path"            # 核心路径可直接用（仅 L0）
    BRANCH = "branch"                  # 走能力分支，未声明则按降级路径
    FORBIDDEN = "forbidden"            # 池一律不用
    DIAGNOSTIC_ONLY = "diagnostic"     # 池一律不用，只作诊断增强


class Capability(Enum):
    """池向下要用到的后端能力项。

    命名对齐设备方向统一后端接口的既有口径（内存句柄的五个公共字段 =
    句柄号 / 种类 / 后端名 / 设备序号 / 字节数，见
    `inbox/接口面诉求-统一内存句柄与显存池界面-交刘鸿彬-20261103.md` 第一节）。
    """

    # —— L0：四家齐备 ——
    DEVICE_MEMORY_INFO = "device_memory_info"            # 设备级显存三键查询（总量/已用/空闲）
    STREAM = "stream"                                    # 流的获取与比较
    EVENT = "event"                                      # 事件 record / query
    BOUNDED_SYNC = "bounded_sync"                        # 有界同步（语义有差，见 TRUE_BOUNDED_SYNC）
    GRAPH_CAPTURE = "graph_capture"                       # 图捕获（入口名各家不同）
    PROBE_LEVEL_RECOVERY = "probe_level_recovery"         # 探针级恢复
    MULTI_DEVICE = "multi_device"                         # 多卡
    STREAM_PRIORITY_RANGE = "stream_priority_range"       # 流优先级范围可读（只读，不含能设置）

    # —— L1：三家有，缺寒武纪 MLU590 ——
    MEMORY_HANDLE = "memory_handle"                       # 统一内存句柄（申请/释放）
    ALLOCATOR_ALLOCATED_BYTES = "allocator_allocated_bytes"  # 分配器视角已分配量
    CROSS_STREAM_PROTECTION = "cross_stream_protection"   # 跨流内存保护（record_stream）
    CONTEXT_OBSERVE = "context_observe"                   # 设备上下文只读观测

    # —— L2：两家（昇腾 / 平头哥）——
    CONTEXT_LIFECYCLE = "context_lifecycle"               # 设备上下文生命周期（建/切/销）

    # —— L3：一家（仅昇腾）——
    TRUE_BOUNDED_SYNC = "true_bounded_sync"               # 真有界同步（其余三家是超时上报/轮询）
    VENDOR_ERROR_CODE_TABLE = "vendor_error_code_table"   # 厂商数值错误码表
    DEVICE_LEVEL_REBUILD = "device_level_rebuild"         # 设备级真重建


_LEVELS: dict[Capability, CapabilityLevel] = {
    Capability.DEVICE_MEMORY_INFO: CapabilityLevel.L0,
    Capability.STREAM: CapabilityLevel.L0,
    Capability.EVENT: CapabilityLevel.L0,
    Capability.BOUNDED_SYNC: CapabilityLevel.L0,
    Capability.GRAPH_CAPTURE: CapabilityLevel.L0,
    Capability.PROBE_LEVEL_RECOVERY: CapabilityLevel.L0,
    Capability.MULTI_DEVICE: CapabilityLevel.L0,
    Capability.STREAM_PRIORITY_RANGE: CapabilityLevel.L0,
    Capability.MEMORY_HANDLE: CapabilityLevel.L1,
    Capability.ALLOCATOR_ALLOCATED_BYTES: CapabilityLevel.L1,
    Capability.CROSS_STREAM_PROTECTION: CapabilityLevel.L1,
    Capability.CONTEXT_OBSERVE: CapabilityLevel.L1,
    Capability.CONTEXT_LIFECYCLE: CapabilityLevel.L2,
    Capability.TRUE_BOUNDED_SYNC: CapabilityLevel.L3,
    Capability.VENDOR_ERROR_CODE_TABLE: CapabilityLevel.L3,
    Capability.DEVICE_LEVEL_REBUILD: CapabilityLevel.L3,
}

_USAGE: dict[Capability, PoolUsage] = {
    # L0 八项都在核心路径里
    **{cap: PoolUsage.CORE_PATH for cap, lvl in _LEVELS.items() if lvl is CapabilityLevel.L0},
    # L1 四项走能力分支
    Capability.MEMORY_HANDLE: PoolUsage.BRANCH,
    Capability.ALLOCATOR_ALLOCATED_BYTES: PoolUsage.BRANCH,
    Capability.CROSS_STREAM_PROTECTION: PoolUsage.BRANCH,
    Capability.CONTEXT_OBSERVE: PoolUsage.BRANCH,
    # L2：池绝不自建上下文
    Capability.CONTEXT_LIFECYCLE: PoolUsage.FORBIDDEN,
    # L3：真有界同步是分支（缺了换回收屏障的实现），另两项只作诊断增强
    Capability.TRUE_BOUNDED_SYNC: PoolUsage.BRANCH,
    Capability.VENDOR_ERROR_CODE_TABLE: PoolUsage.DIAGNOSTIC_ONLY,
    Capability.DEVICE_LEVEL_REBUILD: PoolUsage.DIAGNOSTIC_ONLY,
}

CAPABILITY_LEVEL = MappingProxyType(_LEVELS)
CAPABILITY_USAGE = MappingProxyType(_USAGE)

LEVEL_CAPABILITIES = MappingProxyType(
    {
        level: frozenset(cap for cap, lvl in _LEVELS.items() if lvl is level)
        for level in CapabilityLevel
    }
)

L0_CAPABILITIES = LEVEL_CAPABILITIES[CapabilityLevel.L0]
L1_CAPABILITIES = LEVEL_CAPABILITIES[CapabilityLevel.L1]
L2_CAPABILITIES = LEVEL_CAPABILITIES[CapabilityLevel.L2]
L3_CAPABILITIES = LEVEL_CAPABILITIES[CapabilityLevel.L3]


#: 三条写死的不变量（调研第五节 5.3 与 2.3；设计文档与实现都不得绕过）。
POOL_INVARIANTS = (
    "回收不依赖厂商报错：二次释放的行为四家相反（昇腾静默成功、昆仑芯如实报错），"
    "池必须自己记账拦截。",
    "事件判完成不许裸调查询：寒武纪与平头哥对未 record 的事件误报「已完成」，"
    "必须走统一层已修正的事件适配器。",
    "池持有流必须强持有、禁用弱引用：流 + 弱引用 + 被回收在平头哥与昇腾上是退出期段错误。",
)


def capability_level(cap: Capability) -> CapabilityLevel:
    """这项能力属哪一级（按声明家数）。"""
    return _LEVELS[cap]


def pool_usage(cap: Capability) -> PoolUsage:
    """池对这项能力的用法。级别只说家数，用法看这里。"""
    return _USAGE[cap]
