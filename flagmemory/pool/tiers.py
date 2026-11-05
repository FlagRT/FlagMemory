"""档位表：按运行时查到的分配粒度切档，**池本体不内嵌任何粒度常数**。

依据 `docs/design/显存池概念与设计基础.md` 第三节 3.1 档位设计的两条约束：

1. **档位粒度不写死常数。** 底座的块粒度在四家上不同（实例：平头哥那栈的默认块大小是
   512 KB），档位表必须由能力层按后端给出的分配粒度推导。
2. 档位数量与跨档策略属实现可调参数，不属冻结范围；**冻结的是「定档失败即报错」**。

推导规则（与三家原生粒度查询接口的两档语义对齐）：

    档位基数 base = 推荐粒度
    步长     step = 最小粒度
    第 i 档        = align_up(base × 2ⁱ, step)

即基数取推荐值、每档向上对齐到最小值的整数倍 —— 所以每一档都是「最小粒度的整数倍」，
这正是三家接口里 `最小粒度` 那一档的用途。

粒度从哪来，只有两条合法来源，缺两条就报 `TierResolutionError`：

| 来源 | 适用 | 标记 |
|---|---|---|
| 后端的 `GranularitySource` 查询 | 昇腾 910C / 寒武纪 MLU590 / 平头哥 PPU | `backend_query` |
| 调用方显式给的配置项 | **昆仑芯 P800**（两边都查不到粒度，无厂商声明可依据） | `caller_config` |
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import GranularitySource
from .types import InvalidArgument, TierResolutionError

__all__ = ["TierSpec", "TierTable", "resolve_tier_table"]

#: 默认档位数。属「实现可调参数」，不是粒度常数 —— 它不决定任何字节数，
#: 只决定从基数往上翻几档。
DEFAULT_TIER_COUNT = 12


def _align_up(value: int, step: int) -> int:
    return -(-value // step) * step


@dataclass(frozen=True)
class TierSpec:
    """档位表的推导输入。调用方给这个，就是上面表里的 `caller_config` 那一行。"""

    #: 档位基数（对应三家接口里的「推荐粒度」）
    base_bytes: int
    #: 步长（对应「最小粒度」），每一档都对齐到它的整数倍
    step_bytes: int
    #: 从基数往上翻几档
    tier_count: int = DEFAULT_TIER_COUNT

    def __post_init__(self) -> None:
        for name in ("base_bytes", "step_bytes", "tier_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise InvalidArgument(f"TierSpec.{name} 必须是 int，收到 {type(value).__name__}")
            if value <= 0:
                raise InvalidArgument(f"TierSpec.{name} 必须为正：{value}")


@dataclass(frozen=True)
class TierTable:
    """推导出来的档位表。`source` 说明粒度是查来的还是调用方给的，进 `query_capability`。"""

    tiers: tuple[int, ...]
    spec: TierSpec
    source: str

    def __post_init__(self) -> None:
        if not self.tiers:
            raise TierResolutionError("档位表为空。")
        if list(self.tiers) != sorted(set(self.tiers)):
            raise TierResolutionError(f"档位表必须严格递增且不重复：{self.tiers}")

    @property
    def largest(self) -> int:
        return self.tiers[-1]

    @property
    def smallest(self) -> int:
        return self.tiers[0]

    def tier_for(self, nbytes: int) -> int:
        """定档：取第一个 ≥ 请求字节数的档位。

        **定档失败即报错**（设计文档 3.1 冻结的那条语义）：请求超过最大档就报
        `TierResolutionError`，不自动扩档、不把它当最大档凑合过去。
        """
        if not isinstance(nbytes, int) or isinstance(nbytes, bool):
            raise InvalidArgument(f"请求字节数必须是 int，收到 {type(nbytes).__name__}")
        if nbytes <= 0:
            raise InvalidArgument(f"请求字节数必须为正：{nbytes}")
        for tier in self.tiers:
            if tier >= nbytes:
                return tier
        raise TierResolutionError(
            f"定档失败：请求 {nbytes} 字节超过最大档 {self.largest}（共 {len(self.tiers)} 档，"
            f"基数 {self.spec.base_bytes} / 步长 {self.spec.step_bytes}，粒度来源 {self.source}）。"
            f"按设计文档 3.1「定档失败即报错」，池不自动扩档。"
        )


def _tiers_from_spec(spec: TierSpec) -> tuple[int, ...]:
    base = _align_up(spec.base_bytes, spec.step_bytes)
    seen: list[int] = []
    for i in range(spec.tier_count):
        tier = _align_up(base * (2 ** i), spec.step_bytes)
        if tier not in seen:
            seen.append(tier)
    return tuple(seen)


def resolve_tier_table(
    backend: Any,
    *,
    tier_spec: TierSpec | None = None,
    device_index: int = 0,
    tier_count: int = DEFAULT_TIER_COUNT,
) -> TierTable:
    """定出档位表。调用方给的配置项优先于后端查询；两者都没有就报错。

    调用方给的优先，是因为昆仑芯 P800 那一格**没有厂商声明可依据** —— 实测得到的基数
    必须能从外面喂进来，而且要盖过任何猜测。
    """
    if tier_spec is not None:
        return TierTable(tiers=_tiers_from_spec(tier_spec), spec=tier_spec, source="caller_config")

    if isinstance(backend, GranularitySource):
        try:
            minimum, recommended = backend.allocation_granularity(device_index)
        except Exception as exc:  # 查询存在但这台机器上失败：如实报，不退回猜测
            raise TierResolutionError(
                f"后端声明了分配粒度查询但调用失败：{exc!r}。"
                f"请改用 tier_spec 显式给档位基数与步长；池不编默认粒度。"
            ) from exc
        for name, value in (("最小粒度", minimum), ("推荐粒度", recommended)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise TierResolutionError(f"后端返回的{name}不是正整数：{value!r}")
        spec = TierSpec(base_bytes=recommended, step_bytes=minimum, tier_count=tier_count)
        return TierTable(tiers=_tiers_from_spec(spec), spec=spec, source="backend_query")

    raise TierResolutionError(
        "推不出档位表：后端没有分配粒度查询（例如昆仑芯 P800 那一档 —— 原生设备属性里无粒度项、"
        "头文件里无对齐宏），调用方也没给 tier_spec。"
        "按设计文档 3.1「档位粒度不写死常数」，池在这里报错而不是编一个默认值 —— "
        "请实测该后端的分配粒度后用 tier_spec 传进来。"
    )
