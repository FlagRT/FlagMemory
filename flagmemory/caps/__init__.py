"""芯片能力抽象层：把「这款芯片有没有这项能力」变成池可以分支的事实。

能力分级（四芯片显存能力调研第五节 5.1）：

    L0  四家齐备 —— 设备级显存三键查询 / 流 / 事件 / 有界同步（语义有差）/ 图捕获 /
        探针级恢复 / 多卡 / 流优先级范围可读。**池的核心路径只许用这一级。**
    L1  三家有，缺寒武纪 —— 统一内存句柄、分配器视角已分配量、跨流内存保护、
        上下文只读观测。走能力分支，缺则降级。
    L2  两家（设备上下文生命周期）、L3  一家（昇腾独有的数值错误码表、设备级真重建）——
        池一律不用，只作诊断增强。

能力如实声明：没实测到的一律不写，未声明的能力被调用时报错，不得静默退化。
这条纪律沿用设备方向统一后端接口，池不自建第二套声明。

本层四块（全部纯标准库，**不许 import torch / vLLM / 厂商运行时**，由
`tools/check_layering.py` 静态把关）：

    capabilities.py  能力枚举 + 分级 + 池的用法策略 + 三条写死的不变量
    backend.py       DeviceBackend 协议（池向下要用的后端能力面）与异常
    degradation.py   五条降级路径与查表函数
    query.py         能力查询入口（has / require / report）
    fake.py          纯内存假后端（单测与无卡环境用；真后端是后续任务）
"""

from .backend import (
    CapabilityNotDeclared,
    CapsError,
    DeviceBackend,
    DoubleFree,
    EventNotRecorded,
    MemoryHandle,
    MemoryInfo,
)
from .capabilities import (
    CAPABILITY_LEVEL,
    CAPABILITY_USAGE,
    Capability,
    CapabilityLevel,
    L0_CAPABILITIES,
    L1_CAPABILITIES,
    L2_CAPABILITIES,
    L3_CAPABILITIES,
    LEVEL_CAPABILITIES,
    POOL_INVARIANTS,
    PoolUsage,
    capability_level,
    pool_usage,
)
from .degradation import (
    DEFAULT_IMPLEMENTATION_PATH,
    DEGRADATION_TABLE,
    DegradationPath,
    degradation_for,
    degradation_plan,
    missing_capabilities,
)
from .query import CapabilityReport, has, report, require

__all__ = [
    # backend
    "CapsError",
    "CapabilityNotDeclared",
    "EventNotRecorded",
    "DoubleFree",
    "MemoryInfo",
    "MemoryHandle",
    "DeviceBackend",
    # capabilities
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
    # degradation
    "DegradationPath",
    "DEGRADATION_TABLE",
    "DEFAULT_IMPLEMENTATION_PATH",
    "degradation_for",
    "degradation_plan",
    "missing_capabilities",
    # query
    "CapabilityReport",
    "has",
    "require",
    "report",
]

# 刻意不在这里 import fake：假后端是测试与无卡环境的工具，不是能力层的对外面。
# 要用就显式 `from flagmemory.caps.fake import FakeBackend`。
