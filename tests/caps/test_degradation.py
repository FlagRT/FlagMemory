"""五条降级路径的查表。全部不依赖真卡。"""

from __future__ import annotations

import pytest

from flagmemory.caps.capabilities import Capability, L0_CAPABILITIES, L1_CAPABILITIES
from flagmemory.caps.degradation import (
    DEFAULT_IMPLEMENTATION_PATH,
    DEGRADATION_TABLE,
    degradation_for,
    degradation_plan,
    missing_capabilities,
)
from flagmemory.caps.fake import ASCEND_LIKE, CAMBRICON_LIKE, KUNLUNXIN_LIKE, PPU_LIKE


class TestNormal:
    """正常：五条路径查得到，内容对得上调研稿 §5.2。"""

    def test_表里正好五条(self):
        assert len(DEGRADATION_TABLE) == 5

    def test_缺内存句柄给出张量级池路径(self):
        path = degradation_for(Capability.MEMORY_HANDLE)
        assert path is not None
        assert path.key == "tensor_level_pool"
        assert "张量级池" in path.action
        assert "不碰裸指针" in path.action

    def test_张量级池同时是四家通用的默认实现路径(self):
        assert DEFAULT_IMPLEMENTATION_PATH is degradation_for(Capability.MEMORY_HANDLE)
        assert "默认实现路径" in DEFAULT_IMPLEMENTATION_PATH.action

    def test_缺上下文生命周期则池绝不自建上下文(self):
        path = degradation_for(Capability.CONTEXT_LIFECYCLE)
        assert path is not None and path.key == "no_self_context"
        assert "绝不自建上下文" in path.action

    def test_缺真有界同步则超时上报加轮询且超时不等于失败(self):
        path = degradation_for(Capability.TRUE_BOUNDED_SYNC)
        assert path is not None and path.key == "timeout_report_and_poll"
        assert "超时不等于失败" in path.action
        assert "不得阻塞等待" in path.action

    def test_缺数值错误码则走错误文本规则且不自建(self):
        path = degradation_for(Capability.VENDOR_ERROR_CODE_TABLE)
        assert path is not None and path.key == "error_text_rules"
        assert "不自建" in path.action

    def test_缺设备级真重建则只上报不自救(self):
        path = degradation_for(Capability.DEVICE_LEVEL_REBUILD)
        assert path is not None and path.key == "in_process_reclaim_only"
        assert "只上报不自救" in path.action

    def test_每条路径都带依据(self):
        for path in DEGRADATION_TABLE.values():
            assert path.basis.startswith("调研 §5.2")


class TestBoundary:
    """边界：同一张表在四家不同声明上的分支走法。"""

    def test_寒武纪档缺句柄_其余三家不缺(self):
        assert Capability.MEMORY_HANDLE not in CAMBRICON_LIKE
        for declared in (ASCEND_LIKE, KUNLUNXIN_LIKE, PPU_LIKE):
            assert Capability.MEMORY_HANDLE in declared

    def test_昇腾档一条降级都不用走(self):
        assert degradation_plan(ASCEND_LIKE) == ()

    def test_寒武纪档五条全要走(self):
        plan = degradation_plan(CAMBRICON_LIKE)
        assert [p.key for p in plan] == [
            "tensor_level_pool",
            "no_self_context",
            "timeout_report_and_poll",
            "error_text_rules",
            "in_process_reclaim_only",
        ]

    def test_昆仑芯档走三条_平头哥档走两条(self):
        # 昆仑芯：无上下文生命周期 + 无真有界 + 无错误码/重建 ⇒ 句柄那条不走。
        assert [p.key for p in degradation_plan(KUNLUNXIN_LIKE)] == [
            "no_self_context",
            "timeout_report_and_poll",
            "error_text_rules",
            "in_process_reclaim_only",
        ]
        # 平头哥：有上下文生命周期（L2，池仍不用，但不属「缺」）。
        assert [p.key for p in degradation_plan(PPU_LIKE)] == [
            "timeout_report_and_poll",
            "error_text_rules",
            "in_process_reclaim_only",
        ]

    def test_L0能力没有降级路径(self):
        for cap in L0_CAPABILITIES:
            assert degradation_for(cap) is None

    def test_空声明与全声明两个极端(self):
        assert len(degradation_plan(frozenset())) == 5
        assert degradation_plan(frozenset(Capability)) == ()

    def test_缺失集合只算降级表里的能力(self):
        missing = missing_capabilities(CAMBRICON_LIKE)
        assert missing == frozenset(DEGRADATION_TABLE.keys())
        # 指定 wanted 时只看 wanted
        assert missing_capabilities(CAMBRICON_LIKE, L1_CAPABILITIES) == frozenset(L1_CAPABILITIES)


class TestError:
    """异常：表只读；查非法键报错。"""

    def test_降级表不可写(self):
        with pytest.raises(TypeError):
            DEGRADATION_TABLE[Capability.STREAM] = DEFAULT_IMPLEMENTATION_PATH  # type: ignore[index]

    def test_路径对象不可改(self):
        with pytest.raises(Exception):
            DEFAULT_IMPLEMENTATION_PATH.action = "随便改"  # type: ignore[misc]

    def test_查非枚举键报KeyError或返回None不混淆(self):
        # degradation_for 对合法枚举返回 None 表示「无降级路径」；非枚举键属调用方的错。
        with pytest.raises(TypeError):
            degradation_for(["memory_handle"])  # type: ignore[arg-type]
