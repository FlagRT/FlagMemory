"""昇腾接线层的真卡用例 —— **要卡**，默认不跑（跑卡窗口另排，本增量不执行）。

跑法（910C 宿主栈容器内，仓根在 PYTHONPATH 上）：

    pytest -m realcard tests/tiering/test_npu_backend_realcard.py

覆盖主机用例碰不到的部分：懒加载真 `NpuDmaCopyBackend`、bind 真块表、
copy_out/copy_in 全链路 + `torch.npu.Event` 完成确认、锁页实测。
"""

from __future__ import annotations

import pytest

from flagmemory.tiering.backends.npu_backend import NpuTierBackend

pytestmark = pytest.mark.realcard


def test_真connector懒加载与全链路搬运闭环():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    from flagmemory.tiering.offload.npu.npu_copy_backend import NpuDmaCopyBackend

    adapter = NpuTierBackend(backend_id="ascend")
    assert adapter._copy_backend is None  # 未 bind 前不持有真 connector

    dev = torch.device("npu:0")
    torch.npu.set_device(dev)
    n_blocks, block_bytes = 4, 4096
    gpu = torch.arange(n_blocks * block_bytes, dtype=torch.int8, device=dev).view(
        n_blocks, block_bytes
    )
    cpu = torch.zeros(n_blocks, block_bytes, dtype=torch.int8, pin_memory=True)
    load_stream = torch.npu.Stream()
    store_stream = torch.npu.Stream()

    adapter.bind(
        {"layer.0": gpu},
        {"layer.0": cpu},
        dev,
        load_stream,
        store_stream,
        block_bytes=block_bytes,
    )
    assert isinstance(adapter._copy_backend, NpuDmaCopyBackend)

    # D2H（换出方向）
    ticket = adapter.copy_out("e1", 2, block_bytes, stream_id=0)
    adapter.register_entry("e1", 2)
    for _ in range(1000):
        if adapter.confirm(ticket):
            break
        torch.npu.synchronize(dev)
    assert adapter.confirm(ticket) is True
    assert torch.equal(cpu[2], gpu[2])

    # H2D（换入方向）
    cpu[3] = 42
    ticket = adapter.copy_in("e1", 3, block_bytes, stream_id=0)
    for _ in range(1000):
        if adapter.confirm(ticket):
            break
        torch.npu.synchronize(dev)
    assert adapter.confirm(ticket) is True
    assert torch.equal(gpu[3], cpu[3])

    adapter.drop_entry("e1")
    adapter.shutdown()

    # 锁页实测在真环境应为 True（torch_npu pin_memory 路线）
    assert adapter.probe_capabilities().supports_pinned is True
