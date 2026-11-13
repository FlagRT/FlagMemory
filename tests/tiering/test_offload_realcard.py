"""四套 KV 卸载栈的真导入 —— **要卡**，默认不跑。

跑法（在带卡机器的宿主栈容器内，一次只有一套栈能通过，另三套的宿主栈版本不同）：

    pytest -m realcard tests/tiering/test_offload_realcard.py

这里刻意只验「能导入 + connector 类能取到 + 别名路径取到的是同一个类」。
真实的卸载行为用实测证据覆盖（归 ADR-0014 §3.5 第 3 件改造处置的那批数据），不在单测里跑。
"""

from __future__ import annotations

import importlib

import pytest

STACKS = [
    ("flagmemory.tiering.offload.npu", "offload.npu_offload", "NpuSimpleCPUOffloadConnector"),
    ("flagmemory.tiering.offload.ppu", "offload.ppu_offload", "PpuSimpleCPUOffloadConnector"),
    ("flagmemory.tiering.offload.xpu", "offload.xpu_offload", "XpuSimpleCPUOffloadConnector"),
    ("flagmemory.tiering.offload.mlu", "offload.mlu_offload", "MluSimpleCPUOffloadConnector"),
]


@pytest.mark.realcard
@pytest.mark.parametrize("new_path,alias_path,class_name", STACKS)
def test_栈可导入且新旧路径取到同一个类(new_path, alias_path, class_name):
    new_mod = importlib.import_module(new_path)
    new_cls = getattr(new_mod, class_name)
    alias_mod = importlib.import_module(alias_path)
    with pytest.warns(DeprecationWarning):
        alias_cls = getattr(alias_mod, class_name)
    assert alias_cls is new_cls
