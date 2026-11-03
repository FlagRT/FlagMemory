"""能力枚举与四级分法。全部不依赖真卡。"""

from __future__ import annotations

import pytest

from flagmemory.caps.capabilities import (
    CAPABILITY_LEVEL,
    CAPABILITY_USAGE,
    Capability,
    CapabilityLevel,
    L0_CAPABILITIES,
    L1_CAPABILITIES,
    L2_CAPABILITIES,
    L3_CAPABILITIES,
    POOL_INVARIANTS,
    PoolUsage,
    capability_level,
    pool_usage,
)


class TestNormal:
    """正常：分级表与调研稿第五节 5.1 对得上。"""

    def test_每项能力都有级别和用法(self):
        for cap in Capability:
            assert cap in CAPABILITY_LEVEL, cap
            assert cap in CAPABILITY_USAGE, cap

    def test_L0_是四家齐备的八项(self):
        assert {c.value for c in L0_CAPABILITIES} == {
            "device_memory_info",
            "stream",
            "event",
            "bounded_sync",
            "graph_capture",
            "probe_level_recovery",
            "multi_device",
            "stream_priority_range",
        }

    def test_L1_是三家有缺寒武纪的四项(self):
        assert {c.value for c in L1_CAPABILITIES} == {
            "memory_handle",
            "allocator_allocated_bytes",
            "cross_stream_protection",
            "context_observe",
        }

    def test_L2_只有上下文生命周期_L3_含昇腾独有三项(self):
        assert L2_CAPABILITIES == frozenset({Capability.CONTEXT_LIFECYCLE})
        assert {c.value for c in L3_CAPABILITIES} == {
            "true_bounded_sync",
            "vendor_error_code_table",
            "device_level_rebuild",
        }

    def test_核心路径只许用L0(self):
        core = {c for c in Capability if pool_usage(c) is PoolUsage.CORE_PATH}
        assert core == set(L0_CAPABILITIES)

    def test_上下文生命周期是池一律不用(self):
        assert pool_usage(Capability.CONTEXT_LIFECYCLE) is PoolUsage.FORBIDDEN

    def test_三条写死的不变量在位(self):
        assert len(POOL_INVARIANTS) == 3
        joined = "".join(POOL_INVARIANTS)
        assert "不依赖厂商报错" in joined
        assert "裸调查询" in joined
        assert "强持有" in joined


class TestBoundary:
    """边界：分级的边界 —— 家数、级别与用法三者不是一回事。"""

    def test_级别对应的家数(self):
        assert CapabilityLevel.L0.families == 4
        assert CapabilityLevel.L1.families == 3
        assert CapabilityLevel.L2.families == 2
        assert CapabilityLevel.L3.families == 1

    def test_真有界同步按家数属L3但用法是分支(self):
        # 这是「级别只说家数、用法另算」的边界：只看级别会把它误判成池不该用的东西。
        assert capability_level(Capability.TRUE_BOUNDED_SYNC) is CapabilityLevel.L3
        assert pool_usage(Capability.TRUE_BOUNDED_SYNC) is PoolUsage.BRANCH

    def test_另两项L3只作诊断增强(self):
        for cap in (Capability.VENDOR_ERROR_CODE_TABLE, Capability.DEVICE_LEVEL_REBUILD):
            assert pool_usage(cap) is PoolUsage.DIAGNOSTIC_ONLY

    def test_有界同步与真有界同步是两项不同能力(self):
        # L0 的「有界同步」四家齐备但语义有差；「真有界」只有昇腾。混成一项就会在三家上写出
        # 阻塞等待。
        assert Capability.BOUNDED_SYNC is not Capability.TRUE_BOUNDED_SYNC
        assert capability_level(Capability.BOUNDED_SYNC) is CapabilityLevel.L0

    def test_四级互不重叠且覆盖全部(self):
        levels = [L0_CAPABILITIES, L1_CAPABILITIES, L2_CAPABILITIES, L3_CAPABILITIES]
        union = frozenset().union(*levels)
        assert union == frozenset(Capability)
        assert sum(len(s) for s in levels) == len(union)


class TestError:
    """异常：表是只读的，不许运行期被改。"""

    def test_分级表不可写(self):
        with pytest.raises(TypeError):
            CAPABILITY_LEVEL[Capability.STREAM] = CapabilityLevel.L3  # type: ignore[index]

    def test_用法表不可写(self):
        with pytest.raises(TypeError):
            CAPABILITY_USAGE[Capability.STREAM] = PoolUsage.FORBIDDEN  # type: ignore[index]

    def test_查不在枚举里的东西报KeyError(self):
        with pytest.raises(KeyError):
            capability_level("memory_handle")  # type: ignore[arg-type]
