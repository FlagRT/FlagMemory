"""能力查询入口 has / require / report。全部不依赖真卡。"""

from __future__ import annotations

import pytest

from flagmemory.caps.backend import CapabilityNotDeclared
from flagmemory.caps.capabilities import (
    Capability,
    CapabilityLevel,
    L0_CAPABILITIES,
    L1_CAPABILITIES,
)
from flagmemory.caps.fake import FakeBackend
from flagmemory.caps.query import has, report, require


class TestNormal:
    """正常：能力查询返回已声明能力。"""

    def test_查已声明的能力返回True(self, l0_only_backend):
        for cap in L0_CAPABILITIES:
            assert has(l0_only_backend, cap) is True

    def test_查未声明的能力返回False(self, l0_only_backend):
        for cap in L1_CAPABILITIES:
            assert has(l0_only_backend, cap) is False

    def test_require已声明的能力不抛(self, l0_only_backend):
        require(l0_only_backend, Capability.DEVICE_MEMORY_INFO)

    def test_report给出声明缺失与降级计划(self, l0_only_backend):
        rep = report(l0_only_backend)
        assert rep.backend_name == "l0-only"
        assert rep.declared == frozenset(L0_CAPABILITIES)
        assert rep.core_path_ready is True
        assert len(rep.plan) == 5  # L0-only 档五条降级全要走

    def test_report按级别分组(self, full_backend):
        rep = report(full_backend)
        assert rep.by_level(CapabilityLevel.L0) == frozenset(L0_CAPABILITIES)
        assert rep.by_level(CapabilityLevel.L1) == frozenset(L1_CAPABILITIES)
        assert rep.by_level(CapabilityLevel.L2) == frozenset({Capability.CONTEXT_LIFECYCLE})

    def test_池真会用的能力不含L2与两项诊断能力(self, full_backend):
        usable = report(full_backend).usable_by_pool()
        assert Capability.CONTEXT_LIFECYCLE not in usable
        assert Capability.VENDOR_ERROR_CODE_TABLE not in usable
        assert Capability.DEVICE_LEVEL_REBUILD not in usable
        # 真有界同步是分支能力，池会用
        assert Capability.TRUE_BOUNDED_SYNC in usable


class TestBoundary:
    """边界：L0 齐备与不齐备的判读；L1 能力两种声明下的分支。"""

    def test_L1能力在声明与未声明两档上的分支(self, handle_backend, l0_only_backend):
        assert has(handle_backend, Capability.MEMORY_HANDLE) is True
        assert has(l0_only_backend, Capability.MEMORY_HANDLE) is False

    def test_缺一项L0就判核心路径不就绪(self):
        crippled = FakeBackend(
            name="no-event",
            declared=frozenset(L0_CAPABILITIES - {Capability.EVENT}),
        )
        rep = report(crippled)
        assert rep.core_path_ready is False
        assert rep.missing_core == frozenset({Capability.EVENT})

    def test_L0齐备时缺的核心能力为空集(self, l0_only_backend):
        rep = report(l0_only_backend)
        assert rep.missing_core == frozenset()

    def test_空声明后端(self):
        rep = report(FakeBackend(name="nothing", declared=frozenset()))
        assert rep.core_path_ready is False
        assert rep.missing_core == frozenset(L0_CAPABILITIES)
        assert rep.usable_by_pool() == frozenset()

    def test_全声明后端一条降级不走(self, full_backend):
        rep = report(full_backend)
        assert rep.missing == frozenset()
        assert rep.plan == ()

    def test_没有name属性时退回类名(self):
        class Anonymous:
            def declared_capabilities(self):
                return frozenset()

        assert report(Anonymous()).backend_name == "Anonymous"


class TestError:
    """异常：require 未声明能力必须抛，且带得出是哪项、哪个后端。"""

    def test_require未声明能力抛CapabilityNotDeclared(self, l0_only_backend):
        with pytest.raises(CapabilityNotDeclared) as excinfo:
            require(l0_only_backend, Capability.MEMORY_HANDLE)
        assert excinfo.value.capability is Capability.MEMORY_HANDLE
        assert excinfo.value.backend_name == "l0-only"

    def test_异常文案指向降级表而不是建议静默跳过(self, l0_only_backend):
        with pytest.raises(CapabilityNotDeclared) as excinfo:
            require(l0_only_backend, Capability.CONTEXT_OBSERVE)
        message = str(excinfo.value)
        assert "不得静默退化" in message
        assert "degradation" in message

    def test_report不吞后端的异常(self):
        class Broken:
            name = "broken"

            def declared_capabilities(self):
                raise RuntimeError("后端自己坏了")

        with pytest.raises(RuntimeError, match="后端自己坏了"):
            report(Broken())
