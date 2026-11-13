"""寒武纪 MLU 接线层的真卡用例 —— **要卡**，默认不跑。

跑法（MLU590 宿主基座镜像一次性容器，空闲物理卡 <N> 映射为容器内 mlu:0，仓根在
PYTHONPATH 上；基座镜像不带 pytest，跑时另把宿主 pip --target 装的 pytest 纯
python 库目录挂进 PYTHONPATH）：

    docker run --rm --network none \
        --device /dev/cambricon_dev<N> --device /dev/cambricon_ctl \
        -v <repo>:/ws:ro -w /ws -e PYTHONPATH=/ws:<pytest库目录> \
        harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0 \
        python -m pytest -m realcard -p no:cacheprovider \
        tests/tiering/test_mlu_backend_realcard.py -v

覆盖主机用例碰不到的部分：bind 懒加载内嵌真拷贝通路 `MluTorchCopyBackend`、
bind 真块表、copy_out/copy_in 全链路 + 事件完成确认（含未 record 事件误报
拦截语义）、锁页实测。

MLU 栈已知事实（2026-11-10 真卡取证 + 2026-11-13 接线前事实钉死探针，判据
依据）：

- API 面 = `torch.mlu.*`（torch_mlu，与 torch.cuda 同族）；
- 未 record 的事件原生查询返回「已完成」（与平头哥 PPU 同列）—— 完成确认
  只走适配器 `confirm`（凭据不在事件表内一律 False）；
- `torch.equal` 该栈跨设备**抛 RuntimeError**（与昇腾静默 False 形态不同，
  同样不可用）—— 字节比较一律取回主机侧做；
- 主机 `pin_memory()` 真锁页 PASS；
- 设备侧 `arange` int8 直出可用（无需 int32 构造再 cast）；
- 三条硬约束（cnrtMemAllocFromPoolAsync 疑似静默崩溃 / fork 不可用 / IPC
  只认裸 cnrtMalloc）本套件全程不触碰：分配只走 torch 缓存分配器、单进程、
  无 IPC。
"""

from __future__ import annotations

import time

import pytest

from flagmemory.tiering.backends.mlu_backend import (
    MLU590_INSTANCE_BLOCK_BYTES,
    MluTierBackend,
    MluTorchCopyBackend,
)

pytestmark = pytest.mark.realcard


def test_真拷贝通路懒加载与全链路搬运闭环():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_mlu")

    adapter = MluTierBackend(backend_id="cambricon")
    assert adapter._copy_backend is None  # 未 bind 前不持有真拷贝通路

    dev = torch.device("mlu:0")
    torch.mlu.set_device(dev)
    n_blocks, block_bytes = 4, 4096
    # 该栈 arange int8 直出可用（2026-11-13 真卡探针 PASS），直接构造
    gpu = torch.arange(n_blocks * block_bytes, dtype=torch.int8, device=dev).view(
        n_blocks, block_bytes
    )
    cpu = torch.zeros(n_blocks, block_bytes, dtype=torch.int8, pin_memory=True)
    load_stream = torch.mlu.Stream()
    store_stream = torch.mlu.Stream()

    adapter.bind(
        {"layer.0": gpu},
        {"layer.0": cpu},
        dev,
        load_stream,
        store_stream,
        block_bytes=block_bytes,
    )
    assert isinstance(adapter._copy_backend, MluTorchCopyBackend)

    # D2H（换出方向）
    ticket = adapter.copy_out("e1", 2, block_bytes, stream_id=0)
    adapter.register_entry("e1", 2)
    for _ in range(1000):
        if adapter.confirm(ticket):
            break
        # 轮询之间同步 + 让出 10ms：后台拷贝线程要 CPU 时间把完成事件 record
        # 入表，主线程紧轮询可能饿死它（昆仑芯栈 2026-11-09 真卡实测坐实的
        # 同款坑，三栈一并防御，此处同款沿用；只影响进度不影响语义）
        torch.mlu.synchronize(dev)
        time.sleep(0.01)
    assert adapter.confirm(ticket) is True
    # 该栈 torch.equal 跨设备抛 RuntimeError（2026-11-13 真卡探针）：比较
    # 一律取回主机侧，逐字节相等判据不变
    assert torch.equal(cpu[2], gpu[2].cpu())

    # H2D（换入方向）：源行 = 条目 e1 的级二落点行（copy_out 发起时登记的行
    # 2），目标行 = 新块号 3 —— 两套编号，copy_in 不得拿 block_id 当源行
    # （契约见 mlu_backend.copy_in docstring）。判据较旧版更强：目标行拿到
    # 的必须是条目源行内容，block_id 行保持全零以证未误读。
    cpu[2] = 42  # 在条目真正的级二副本行上改数（cpu[3] 不写，保持全零）
    ticket = adapter.copy_in("e1", 3, block_bytes, stream_id=0)
    for _ in range(1000):
        if adapter.confirm(ticket):
            break
        torch.mlu.synchronize(dev)
        time.sleep(0.01)
    assert adapter.confirm(ticket) is True
    assert torch.equal(gpu[3].cpu(), cpu[2])
    assert not bool(cpu[3].any())  # block_id 行未被读作源

    adapter.drop_entry("e1")
    adapter.shutdown()

    # 锁页实测在真环境应为 True（torch_mlu pin_memory 走 cnrt 主机锁页路线，
    # 2026-11-13 真卡探针 PASS）
    assert adapter.probe_capabilities().supports_pinned is True
    # 未 bind 时粒度 = MLU590 实测实例值（地址间距法 8 KiB，T-9）
    assert MLU590_INSTANCE_BLOCK_BYTES == 8 * 1024


def test_未record事件误报拦截():
    """该栈没 record 过的事件原生查询返回 True（2026-11-10 取证 + 2026-11-13
    复核）：适配器凭据不在事件表内必须给 False，绝不递原生 True。"""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_mlu")

    adapter = MluTierBackend(backend_id="cambricon")
    # 先钉原生误报事实成立（未 record 的 Event().query() == True）
    assert torch.mlu.Event().query() is True
    # 事件表为空（无任何 record 过的事件）⇒ 任意凭据一律未完成
    assert adapter.confirm(1) is False
    assert adapter.confirm(999) is False
