"""平头哥 PPU 接线层的真卡用例 —— **要卡**，默认不跑。

跑法（PTG_0 一次性容器，PPU 0 号卡，仓根在 PYTHONPATH 上）：

    docker run --rm --device /dev/alixpu_ppu0 --device /dev/alixpu \
        --device /dev/alixpu_ctl -v <repo>:/ws -w /ws \
        flagrt/thead-ppu-runtime:1.0.0-ppu3.6-py312-torch2.10.0-flagtree0.7.0-triton3.6.0-x86_64 \
        bash -c "source /usr/local/PPU_SDK/envsetup.sh && \
                 UMD_PLATFORM_TYPE=1 HGGC_DRIVER_CANDIDATE=UMD \
                 python -m pytest -m realcard tests/tiering/test_ppu_backend_realcard.py"

覆盖主机用例碰不到的部分：懒加载真 `DmaCopyBackend`、bind 真块表、
copy_out/copy_in 全链路 + 事件完成确认（含未 record 事件误报拦截语义）、
锁页三级链真环境探测。
"""

from __future__ import annotations

import pytest

from flagmemory.tiering.backends.ppu_backend import (
    PINNED_LEVEL_HOST_REGISTER,
    PINNED_LEVEL_PIN_MEMORY,
    PpuTierBackend,
)

pytestmark = pytest.mark.realcard


def test_真connector懒加载与全链路搬运闭环():
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from flagmemory.tiering.offload.ppu.ppu_copy_backend import DmaCopyBackend

    adapter = PpuTierBackend(backend_id="ppu")
    assert adapter._copy_backend is None  # 未 bind 前不持有真 connector

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
    assert isinstance(adapter._copy_backend, DmaCopyBackend)

    # D2H（换出方向）
    ticket = adapter.copy_out("e1", 2, block_bytes, stream_id=0)
    adapter.register_entry("e1", 2)
    for _ in range(1000):
        if adapter.confirm(ticket):
            break
        torch.cuda.synchronize(dev)
    assert adapter.confirm(ticket) is True
    assert torch.equal(cpu[2], gpu[2].cpu())  # 该栈 equal 不支持跨设备，取回主机侧比较

    # H2D（换入方向）
    cpu[3] = 42
    ticket = adapter.copy_in("e1", 3, block_bytes, stream_id=0)
    for _ in range(1000):
        if adapter.confirm(ticket):
            break
        torch.cuda.synchronize(dev)
    assert adapter.confirm(ticket) is True
    assert torch.equal(gpu[3].cpu(), cpu[3])

    adapter.drop_entry("e1")
    adapter.shutdown()

    # 锁页三级链真环境探测：应探到一或二级（均真锁页），级别名如实登记
    probed = adapter.probe_capabilities()
    assert probed.supports_pinned is True
    assert adapter.probed_pinned_level in (
        PINNED_LEVEL_HOST_REGISTER,
        PINNED_LEVEL_PIN_MEMORY,
    )


def test_未record事件误报拦截():
    """该栈没 record 过的事件原生查询返回 True（设计 §4.4 第 6 条，2026-11-06
    实测复现）：适配器凭据不在事件表内必须给 False，绝不递原生 True。"""
    torch = pytest.importorskip("torch")

    adapter = PpuTierBackend(backend_id="ppu")
    # 先钉原生误报事实成立（未 record 的 Event().query() == True）
    assert torch.cuda.Event().query() is True
    # 事件表为空（无任何 record 过的事件）⇒ 任意凭据一律未完成
    assert adapter.confirm(1) is False
    assert adapter.confirm(999) is False
