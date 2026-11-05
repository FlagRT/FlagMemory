"""公共夹具。本文件刻意不 import torch / vLLM：默认那批用例必须在无卡无框架环境跑通。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# 允许在未 `pip install -e .` 的情况下直接 pytest（CI 里是装过的，本机顺手跑时省一步）。
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flagmemory.caps.capabilities import L0_CAPABILITIES  # noqa: E402
from flagmemory.caps.fake import (  # noqa: E402
    ASCEND_LIKE,
    CAMBRICON_LIKE,
    FakeBackend,
    KUNLUNXIN_LIKE,
    PPU_LIKE,
)


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def l0_only_backend() -> FakeBackend:
    """只声明 L0 八项的后端（= 寒武纪 MLU590 那一档，L1 四项全未声明）。"""
    return FakeBackend(name="l0-only", declared=frozenset(L0_CAPABILITIES))


@pytest.fixture
def handle_backend() -> FakeBackend:
    """声明了统一内存句柄的后端（= 昆仑芯 P800 那一档：L0 + L1）。"""
    return FakeBackend(name="with-handle", declared=KUNLUNXIN_LIKE)


@pytest.fixture
def full_backend() -> FakeBackend:
    """四级全声明的后端（= 昇腾 910C 那一档，唯一有真有界同步与数值错误码表的一家）。"""
    return FakeBackend(name="full", declared=ASCEND_LIKE)


@pytest.fixture(params=["ascend-like", "kunlunxin-like", "cambricon-like", "ppu-like"])
def chip_like_backend(request) -> FakeBackend:
    """四家各一个预设声明，用来验「能力分支在四家上都有定义的走法」。"""
    presets = {
        "ascend-like": ASCEND_LIKE,
        "kunlunxin-like": KUNLUNXIN_LIKE,
        "cambricon-like": CAMBRICON_LIKE,
        "ppu-like": PPU_LIKE,
    }
    return FakeBackend(name=request.param, declared=presets[request.param])
