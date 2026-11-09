"""昆仑芯接线层的真卡用例 —— **要卡**，默认不跑（跑卡窗口另排，本增量不执行）。

跑法（P800 宿主栈容器内，仓根在 PYTHONPATH 上）：

    pytest -m realcard tests/tiering/test_xpu_backend_realcard.py

覆盖主机用例碰不到的部分：懒加载真 `XpuCopyBackend`、bind 真块表、
copy_out/copy_in 全链路 + 事件完成确认、锁页实测。

⚠️ 本栈 API 面铁则：`torch.xpu.*` 恒不可用，一切设备操作按 `torch.cuda.*`
写（torch_xmlir 符号重写到 XPU）—— 本文件因此全程 torch.cuda.*。
"""

from __future__ import annotations

import time

import pytest

from flagmemory.tiering.backends.xpu_backend import XpuTierBackend

pytestmark = pytest.mark.realcard


def test_真connector懒加载与全链路搬运闭环():
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from flagmemory.tiering.offload.xpu.xpu_copy_backend import XpuCopyBackend

    adapter = XpuTierBackend(backend_id="kunlunxin")
    assert adapter._copy_backend is None  # 未 bind 前不持有真 connector

    # 设备面按 torch.cuda 写（符号重写到 XPU），不 touch torch.xpu
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    n_blocks, block_bytes = 4, 4096
    gpu = torch.arange(n_blocks * block_bytes, dtype=torch.int8, device=dev).view(
        n_blocks, block_bytes
    )
    cpu = torch.zeros(n_blocks, block_bytes, dtype=torch.int8, pin_memory=True)
    load_stream = torch.cuda.Stream()
    store_stream = torch.cuda.Stream()

    adapter.bind(
        {"layer.0": gpu},
        {"layer.0": cpu},
        dev,
        load_stream,
        store_stream,
        block_bytes=block_bytes,
    )
    assert isinstance(adapter._copy_backend, XpuCopyBackend)

    # D2H（换出方向）
    ticket = adapter.copy_out("e1", 2, block_bytes, stream_id=0)
    adapter.register_entry("e1", 2)
    for _ in range(1000):
        if adapter.confirm(ticket):
            break
        # 本栈环境事实（2026-11-09 P800 真卡实测）：主线程紧密 synchronize 会
        # 饿死后台拷贝线程的 H2D 提交（D2H 提交快、首轮 confirm 即真，未暴露）
        # —— 每轮让出 10ms，完成语义与逐字节判据不变（与 PPU 真卡用例的
        # 校准手段同款方向：轮询之间留间隔，不改语义只信事件）。
        torch.cuda.synchronize(dev)
        time.sleep(0.01)
    assert adapter.confirm(ticket) is True
    # 跨栈环境事实：torch.equal 跨设备不可靠（PPU 不支持、昇腾静默 False，
    # 2026-11-08 登记的同款坑）—— 比较一律取回主机侧，逐字节相等判据不变。
    assert torch.equal(cpu[2], gpu[2].cpu())

    # H2D（换入方向）：源行 = 条目 e1 的级二落点行（copy_out 发起时登记的行 2），
    # 目标行 = 新块号 3 —— 两套编号，copy_in 不得拿 block_id 当源行（契约见
    # xpu_backend.copy_in docstring）。判据较旧版更强：目标行拿到的必须是条目
    # 源行内容，block_id 行保持全零以证未误读。
    cpu[2] = 42  # 在条目真正的级二副本行上改数（cpu[3] 不写，保持全零）
    ticket = adapter.copy_in("e1", 3, block_bytes, stream_id=0)
    for _ in range(1000):
        if adapter.confirm(ticket):
            break
        # 本栈环境事实（2026-11-09 P800 真卡实测）：主线程紧密 synchronize 会
        # 饿死后台拷贝线程的 H2D 提交（D2H 提交快、首轮 confirm 即真，未暴露）
        # —— 每轮让出 10ms，完成语义与逐字节判据不变（与 PPU 真卡用例的
        # 校准手段同款方向：轮询之间留间隔，不改语义只信事件）。
        torch.cuda.synchronize(dev)
        time.sleep(0.01)
    assert adapter.confirm(ticket) is True
    assert torch.equal(gpu[3].cpu(), cpu[2])
    assert not bool(cpu[3].any())  # block_id 行未被读作源

    adapter.drop_entry("e1")
    adapter.shutdown()

    # 锁页实测在真环境应为 True（pin_memory 走 torch.cuda 符号重写面真锁）
    assert adapter.probe_capabilities().supports_pinned is True
