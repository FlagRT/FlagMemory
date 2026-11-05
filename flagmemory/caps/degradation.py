"""降级路径查表：缺这项能力时池走哪条路（纯标准库，零框架依赖）。

五条路径照抄 `inbox/调研-四芯片显存能力对照与池抽象分级-20261103.md` 第五节 5.2，
**不重新论证**。表的用法：池发现某项能力未声明（或调用抛了 `CapabilityNotDeclared`），
就查这张表拿到路径，按路径走；查不到路径的能力属 L0 核心能力，缺了就是「这款芯片不支持」，
不是降级问题。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from .capabilities import Capability

__all__ = [
    "DegradationPath",
    "DEGRADATION_TABLE",
    "DEFAULT_IMPLEMENTATION_PATH",
    "degradation_for",
    "missing_capabilities",
    "degradation_plan",
]


@dataclass(frozen=True)
class DegradationPath:
    """一条降级路径：缺哪项能力、池改怎么做、依据在哪。"""

    key: str
    missing: Capability
    #: 一句人话：池改走什么
    action: str
    #: 依据（调研稿的条目号）
    basis: str

    def __str__(self) -> str:  # pragma: no cover - 便于报错时直读
        return f"{self.key}: 缺 {self.missing.value} ⇒ {self.action}"


#: ① 缺统一内存句柄（寒武纪 MLU590）→ 张量级池。
#:   这条**同时是四家通用的默认实现路径**；句柄形态只是声明了该能力的三家上的优化路径。
TENSOR_LEVEL_POOL = DegradationPath(
    key="tensor_level_pool",
    missing=Capability.MEMORY_HANDLE,
    action=(
        "池退到张量级池：用框架侧的空张量申请作底座，池只管块的分级、归属与归还账本，"
        "不碰裸指针。这条同时是四家通用的默认实现路径，句柄形态是三家上的优化路径。"
    ),
    basis="调研 §5.2 第 1 条",
)

#: ② 缺上下文生命周期（昆仑芯 P800 / 寒武纪 MLU590）→ 绝不自建上下文。
NO_SELF_CONTEXT = DegradationPath(
    key="no_self_context",
    missing=Capability.CONTEXT_LIFECYCLE,
    action=(
        "池绝不自建上下文，只在框架既有上下文内工作；归属校验走只读观测，"
        "寒武纪上连观测都没有 ⇒ 那一家只靠池自己的登记表。"
    ),
    basis="调研 §5.2 第 2 条",
)

#: ③ 缺真有界同步（除昇腾外三家）→ 超时上报 + 轮询，超时不等于失败。
TIMEOUT_REPORT_AND_POLL = DegradationPath(
    key="timeout_report_and_poll",
    missing=Capability.TRUE_BOUNDED_SYNC,
    action=(
        "回收屏障用「超时上报 + 轮询」，不得阻塞等待；超时不等于失败，"
        "作可恢复态上报。"
    ),
    basis="调研 §5.2 第 3 条",
)

#: ④ 缺数值错误码（除昇腾外三家）→ 错误文本规则，沿用统一层那套，不自建。
ERROR_TEXT_RULES = DegradationPath(
    key="error_text_rules",
    missing=Capability.VENDOR_ERROR_CODE_TABLE,
    action=(
        "异常分级靠错误文本规则：显存不足归最高级、形状/参数类归次级；"
        "沿用统一层已验证的那套分级，不自建。"
    ),
    basis="调研 §5.2 第 4 条",
)

#: ⑤ 缺设备级真重建（除昇腾外三家）→ 只做进程内回收，设备级故障只上报不自救。
IN_PROCESS_RECLAIM_ONLY = DegradationPath(
    key="in_process_reclaim_only",
    missing=Capability.DEVICE_LEVEL_REBUILD,
    action=(
        "压力回收只做进程内回收（释放可回收块、触发卸载）；设备级故障只上报不自救。"
    ),
    basis="调研 §5.2 第 5 条",
)

DEGRADATION_TABLE = MappingProxyType(
    {
        Capability.MEMORY_HANDLE: TENSOR_LEVEL_POOL,
        Capability.CONTEXT_LIFECYCLE: NO_SELF_CONTEXT,
        Capability.TRUE_BOUNDED_SYNC: TIMEOUT_REPORT_AND_POLL,
        Capability.VENDOR_ERROR_CODE_TABLE: ERROR_TEXT_RULES,
        Capability.DEVICE_LEVEL_REBUILD: IN_PROCESS_RECLAIM_ONLY,
    }
)

#: 池的默认实现路径（不是「出了问题才走」的那种降级）——调研 §5.2 第 1 条后半句。
DEFAULT_IMPLEMENTATION_PATH = TENSOR_LEVEL_POOL


def degradation_for(cap: Capability) -> DegradationPath | None:
    """缺这项能力时走哪条降级路径。

    返回 `None` 表示**没有降级路径**：这类是 L0 核心能力（四家齐备），缺了就是
    「这款芯片不支持池」，该报错退出而不是降级。
    """
    return DEGRADATION_TABLE.get(cap)


def missing_capabilities(declared, wanted=None) -> frozenset[Capability]:
    """`wanted` 里有哪些没被 `declared` 声明。`wanted` 缺省 = 降级表里的全部能力。"""
    if wanted is None:
        wanted = DEGRADATION_TABLE.keys()
    declared = frozenset(declared)
    return frozenset(cap for cap in wanted if cap not in declared)


def degradation_plan(declared) -> tuple[DegradationPath, ...]:
    """给一份能力声明，算出池在这个后端上要走的全部降级路径（按表的顺序）。"""
    declared = frozenset(declared)
    return tuple(
        path for cap, path in DEGRADATION_TABLE.items() if cap not in declared
    )
