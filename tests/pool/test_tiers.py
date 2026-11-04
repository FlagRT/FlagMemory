"""档位表：按运行时查到的分配粒度切档，池本体不内嵌粒度常数。全部不依赖真卡。

判据来源：`docs/design/显存池概念与设计基础.md` 第三节 3.1 档位设计的两条约束 ——
「档位粒度不写死常数」与「定档失败即报错」。
"""

from __future__ import annotations

import pytest

from flagmemory.caps.capabilities import L0_CAPABILITIES, L1_CAPABILITIES
from flagmemory.pool import (
    InvalidArgument,
    PoolForm,
    TierSpec,
    TierTable,
    create_pool,
    resolve_tier_table,
)
from flagmemory.pool.types import TierResolutionError

from fakes import TEST_BASE_BYTES, TEST_STEP_BYTES, CompletingBackend, GranularityBackend


class TestNormal:
    """正常：两条合法粒度来源各走一遍，档位表递增且每档对齐到步长。"""

    def test_后端有粒度查询时按查询推导(self):
        backend = GranularityBackend(
            name="granularity", declared=frozenset(L0_CAPABILITIES | L1_CAPABILITIES)
        )
        table = resolve_tier_table(backend, tier_count=5)
        assert table.source == "backend_query"
        assert table.spec.base_bytes == TEST_BASE_BYTES   # 基数取推荐粒度
        assert table.spec.step_bytes == TEST_STEP_BYTES   # 步长取最小粒度
        assert table.tiers[0] == TEST_BASE_BYTES
        assert len(table.tiers) == 5

    def test_调用方给配置项时按配置项推导(self, tier_spec: TierSpec):
        backend = CompletingBackend(name="no-granularity", declared=frozenset(L0_CAPABILITIES))
        table = resolve_tier_table(backend, tier_spec=tier_spec)
        assert table.source == "caller_config"
        assert table.tiers == (512 << 10, 1 << 20, 2 << 20, 4 << 20, 8 << 20, 16 << 20)

    def test_每一档都是步长的整数倍(self, tier_spec: TierSpec):
        table = TierTable(
            tiers=tuple(
                t for t in resolve_tier_table(
                    CompletingBackend(name="x", declared=frozenset(L0_CAPABILITIES)),
                    tier_spec=tier_spec,
                ).tiers
            ),
            spec=tier_spec,
            source="caller_config",
        )
        for tier in table.tiers:
            assert tier % tier_spec.step_bytes == 0

    def test_定档取第一个不小于请求的档(self, tier_spec: TierSpec):
        table = resolve_tier_table(
            CompletingBackend(name="x", declared=frozenset(L0_CAPABILITIES)), tier_spec=tier_spec
        )
        assert table.tier_for(1) == 512 << 10
        assert table.tier_for(512 << 10) == 512 << 10
        assert table.tier_for((512 << 10) + 1) == 1 << 20

    def test_调用方配置项盖过后端查询(self):
        """昆仑芯那一格的粒度只能实测得到，所以外面喂进来的必须盖过一切。"""
        backend = GranularityBackend(
            name="granularity", declared=frozenset(L0_CAPABILITIES | L1_CAPABILITIES)
        )
        spec = TierSpec(base_bytes=1 << 20, step_bytes=1 << 10, tier_count=3)
        table = resolve_tier_table(backend, tier_spec=spec)
        assert table.source == "caller_config"
        assert table.tiers[0] == 1 << 20


class TestBoundary:
    """边界：最小档、最大档、基数不是步长整数倍、单档表。"""

    def test_基数不是步长整数倍时向上对齐(self):
        spec = TierSpec(base_bytes=1000, step_bytes=256, tier_count=3)
        table = resolve_tier_table(
            CompletingBackend(name="x", declared=frozenset(L0_CAPABILITIES)), tier_spec=spec
        )
        assert table.tiers[0] == 1024          # 1000 向上对齐到 256 的整数倍
        assert all(t % 256 == 0 for t in table.tiers)

    def test_请求正好等于最大档可以定档(self, tier_spec: TierSpec):
        table = resolve_tier_table(
            CompletingBackend(name="x", declared=frozenset(L0_CAPABILITIES)), tier_spec=tier_spec
        )
        assert table.tier_for(table.largest) == table.largest

    def test_单档表(self):
        spec = TierSpec(base_bytes=4096, step_bytes=4096, tier_count=1)
        table = resolve_tier_table(
            CompletingBackend(name="x", declared=frozenset(L0_CAPABILITIES)), tier_spec=spec
        )
        assert table.tiers == (4096,)
        assert table.smallest == table.largest == 4096


class TestError:
    """异常：两条来源都没有就报错（绝不编常数）；定档失败即报错；参数非法即报错。"""

    def test_两条粒度来源都没有时报错而不是编常数(self):
        """= 昆仑芯 P800 那一格：原生设备属性无粒度项、头文件无对齐宏，又没给配置项。"""
        backend = CompletingBackend(name="kunlunxin-like", declared=frozenset(L0_CAPABILITIES))
        with pytest.raises(TierResolutionError) as exc:
            resolve_tier_table(backend)
        assert "不写死常数" in str(exc.value)

    def test_请求超过最大档时定档失败即报错(self, tier_spec: TierSpec):
        table = resolve_tier_table(
            CompletingBackend(name="x", declared=frozenset(L0_CAPABILITIES)), tier_spec=tier_spec
        )
        with pytest.raises(TierResolutionError) as exc:
            table.tier_for(table.largest + 1)
        assert "定档失败" in str(exc.value)

    def test_后端粒度查询返回非正数时报错(self):
        class BadGranularity(GranularityBackend):
            def allocation_granularity(self, device_index: int = 0) -> tuple[int, int]:
                return (0, 512 << 10)

        with pytest.raises(TierResolutionError):
            resolve_tier_table(BadGranularity(name="bad", declared=frozenset(L0_CAPABILITIES)))

    def test_后端粒度查询抛异常时如实报错不退回猜测(self):
        class RaisingGranularity(GranularityBackend):
            def allocation_granularity(self, device_index: int = 0) -> tuple[int, int]:
                raise RuntimeError("这台机器上查不到")

        with pytest.raises(TierResolutionError) as exc:
            resolve_tier_table(RaisingGranularity(name="raise", declared=frozenset(L0_CAPABILITIES)))
        assert "不编默认粒度" in str(exc.value)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_档位配置项必须为正(self, bad: int):
        with pytest.raises(InvalidArgument):
            TierSpec(base_bytes=bad, step_bytes=4096)

    @pytest.mark.parametrize("nbytes", [0, -1])
    def test_请求字节数必须为正(self, tier_spec: TierSpec, nbytes: int):
        table = resolve_tier_table(
            CompletingBackend(name="x", declared=frozenset(L0_CAPABILITIES)), tier_spec=tier_spec
        )
        with pytest.raises(InvalidArgument):
            table.tier_for(nbytes)

    def test_布尔不算整数(self, tier_spec: TierSpec):
        table = resolve_tier_table(
            CompletingBackend(name="x", declared=frozenset(L0_CAPABILITIES)), tier_spec=tier_spec
        )
        with pytest.raises(InvalidArgument):
            table.tier_for(True)  # type: ignore[arg-type]

    def test_建池时粒度推不出来也一样报错(self):
        """建池路径与直接调 resolve 的行为必须一致，不许建池时偷偷给个默认值。"""
        backend = CompletingBackend(
            name="kunlunxin-like", declared=frozenset(L0_CAPABILITIES | L1_CAPABILITIES)
        )
        with pytest.raises(TierResolutionError):
            create_pool(backend, form=PoolForm.HANDLE)
