"""能力查询入口（纯标准库，零框架依赖）。

池只经这里问「这个后端有没有这项能力」，不直接读后端的内部字段。
三个口：

    has(backend, cap)        有没有
    require(backend, cap)    没有就抛 CapabilityNotDeclared（不许静默退化）
    report(backend)          一次性拿全：声明/缺失/分级/降级计划
"""

from __future__ import annotations

from dataclasses import dataclass

from .backend import CapabilityNotDeclared, DeviceBackend
from .capabilities import (
    CAPABILITY_LEVEL,
    Capability,
    CapabilityLevel,
    L0_CAPABILITIES,
    PoolUsage,
    pool_usage,
)
from .degradation import DegradationPath, degradation_plan, missing_capabilities

__all__ = ["CapabilityReport", "has", "require", "report"]


@dataclass(frozen=True)
class CapabilityReport:
    """一个后端的能力画像（纯数据，可直接打进日志）。"""

    backend_name: str
    declared: frozenset[Capability]
    missing: frozenset[Capability]
    plan: tuple[DegradationPath, ...]

    @property
    def core_path_ready(self) -> bool:
        """L0 八项是否齐备。不齐 ⇒ 池在这款芯片上不该起，而不是降级。"""
        return L0_CAPABILITIES <= self.declared

    @property
    def missing_core(self) -> frozenset[Capability]:
        """缺的 L0 能力（无降级路径的那些）。"""
        return frozenset(L0_CAPABILITIES - self.declared)

    def by_level(self, level: CapabilityLevel) -> frozenset[Capability]:
        """已声明的能力里属这一级的那些。"""
        return frozenset(
            cap for cap in self.declared if CAPABILITY_LEVEL[cap] is level
        )

    def usable_by_pool(self) -> frozenset[Capability]:
        """已声明且池真会用的能力（核心路径 + 能力分支；L2/L3 的禁用与诊断项不算）。"""
        return frozenset(
            cap
            for cap in self.declared
            if pool_usage(cap) in (PoolUsage.CORE_PATH, PoolUsage.BRANCH)
        )


def _declared(backend: DeviceBackend) -> frozenset[Capability]:
    return frozenset(backend.declared_capabilities())


def _name(backend: DeviceBackend) -> str:
    return getattr(backend, "name", backend.__class__.__name__)


def has(backend: DeviceBackend, cap: Capability) -> bool:
    """这个后端声明了这项能力吗。"""
    return cap in _declared(backend)


def require(backend: DeviceBackend, cap: Capability) -> None:
    """要求这项能力；未声明则抛 `CapabilityNotDeclared`。

    池的调用约定：核心路径直接 `require` L0 能力；L1/L3 的分支能力先 `has` 再决定
    走不走，不要拿 `require` 当分支判断（那是用异常做控制流）。
    """
    if cap not in _declared(backend):
        raise CapabilityNotDeclared(cap, _name(backend))


def report(backend: DeviceBackend) -> CapabilityReport:
    """一次性拿这个后端的能力画像与降级计划。"""
    declared = _declared(backend)
    return CapabilityReport(
        backend_name=_name(backend),
        declared=declared,
        missing=missing_capabilities(declared),
        plan=degradation_plan(declared),
    )
