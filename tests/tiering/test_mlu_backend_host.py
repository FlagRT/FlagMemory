"""寒武纪 MLU 栈接线层（backends/mlu_backend.py）的主机可验部分。

只验协议形状 / 能力声明映射 / 降级登记 / 注册路径 / 事件适配器语义 —— 全部不
import 真 torch_mlu（主机环境无寒武纪运行时；框架绑定路径由
`test_mlu_backend_realcard.py` 的 realcard 用例覆盖）。
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from flagmemory.tiering.backends.mlu_backend import (
    MLU590_INSTANCE_BLOCK_BYTES,
    MluTierBackend,
    MluTorchCopyBackend,
    register_mlu_tier_backend,
)
from flagmemory.tiering.manager import TierBackend


class FakeMluCopyBackend:
    """与 MluTorchCopyBackend 同形状（init/launch_copy/shutdown/事件表）的假件。

    `pending_once`：第一次 launch 的事件先不完成（confirm False），
    调 `complete_pending()` 后完成 —— 钉「轮询再问」语义。
    """

    def __init__(self, *, pending_once: bool = False) -> None:
        self.init_calls: list[dict] = []
        self.launches: list[tuple] = []
        self.shutdown_called = False
        self._pending_once = pending_once
        self._pending_event = None

    def init(self, gpu_caches, cpu_caches, device, load_stream, store_stream):
        self.init_calls.append(
            dict(gpu=gpu_caches, cpu=cpu_caches, device=device)
        )

    def launch_copy(self, src_blocks, dst_blocks, is_store, event_idx, events_list):
        self.launches.append((tuple(src_blocks), tuple(dst_blocks), is_store, event_idx))

        class _Event:
            def __init__(self) -> None:
                self.done = False

            def query(self) -> bool:
                return self.done

        ev = _Event()
        if self._pending_once and self._pending_event is None:
            self._pending_event = ev  # 先不完成
        else:
            ev.done = True
        events_list.append((event_idx, ev))

    def complete_pending(self) -> None:
        if self._pending_event is not None:
            self._pending_event.done = True

    def shutdown(self) -> None:
        self.shutdown_called = True


def bind_fake(adapter: MluTierBackend, copy_backend) -> None:
    adapter.bind(
        {"layer.0": object()},
        {"layer.0": object()},
        device=object(),
        load_stream=object(),
        store_stream=object(),
        block_bytes=64 << 10,
        # copy_backend 已注入
    )


def test_模块导入不引入框架():
    """接线层顶层零框架依赖：干净子进程里 import 后 vllm / torch / torch_mlu
    均不在 sys.modules（不受同进程其他用例的 import 顺序影响）；内嵌真拷贝
    通路 MluTorchCopyBackend 类可取（其 torch import 刻意延迟到方法体）。"""
    code = (
        "import sys; import flagmemory.tiering.backends.mlu_backend as m; "
        "bad=[x for x in ('vllm','torch','torch_mlu') if x in sys.modules]; "
        "assert not bad, bad; "
        "assert m.MluTierBackend and m.MluTorchCopyBackend"
    )
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True
    )


def test_协议形状():
    assert isinstance(MluTierBackend(), TierBackend)


def test_能力声明映射_按登记事实():
    adapter = MluTierBackend()
    declared = adapter.declare_capabilities()
    # MLU 栈 Python 面无批量搬运原语 = 登记事实（逐块 copy_ 即实现），声明即
    # 降级态
    assert declared.has_batch_copy is False
    # torch_mlu pin_memory 走 cnrt 主机锁页路线（真卡实测 PASS），声明支持；
    # 实际锁得住由 probe 实测登记
    assert declared.supports_pinned is True
    # 未 bind 时用 MLU590 实测分配粒度实例值（地址间距法 8 KiB）
    assert declared.transfer_granularity_bytes == MLU590_INSTANCE_BLOCK_BYTES
    assert declared.transfer_granularity_bytes == 8 * 1024


def test_绑定后粒度取推导值():
    adapter = MluTierBackend(copy_backend=FakeMluCopyBackend())
    bind_fake(adapter, adapter._copy_backend)
    assert adapter.declare_capabilities().transfer_granularity_bytes == 64 << 10


def test_未bind发起搬运按T4报错():
    adapter = MluTierBackend()
    with pytest.raises(Exception, match="未 bind"):
        adapter.copy_out("e1", 3, 8 << 10, stream_id=0)
    with pytest.raises(Exception, match="未 bind"):
        adapter.copy_in("e1", 3, 8 << 10, stream_id=0)


def test_逐块拷贝完成确认与轮询语义():
    copy_backend = FakeMluCopyBackend(pending_once=True)
    adapter = MluTierBackend(copy_backend=copy_backend)
    bind_fake(adapter, copy_backend)

    ticket = adapter.copy_out("e1", 7, 8 << 10, stream_id=0)
    assert isinstance(ticket, int)
    # 事件适配器：未完成返回 False（可恢复，轮询再问），不抛错
    assert adapter.confirm(ticket) is False
    copy_backend.complete_pending()
    assert adapter.confirm(ticket) is True
    # 凭据下标不复用、方向如实传给拷贝通路
    assert copy_backend.launches == [((7,), (7,), True, 1)]
    t2 = adapter.copy_in("e1", 7, 8 << 10, stream_id=0)
    assert copy_backend.launches[-1][2] is False and t2 == 2
    # 未知凭据（= 事件未 record 不在表内）⇒ 未完成（False），不抛错 ——
    # 该栈未 record 事件原生查询误报 True，适配器绝不递出去
    assert adapter.confirm(999) is False


def test_copy_in源行按条目定位_不得拿block_id当源行():
    """防回归（对齐 npu/xpu/ppu 三栈 2026-11-10 同构核查口径：copy_in 曾拿
    block_id 同时当 src/dst 行号、copy_out 从不登记级二落点行，拷错行换回全
    零；四栈同款同修，四形态对齐三栈主机防回归）：

    manager 换入时新块号 ≠ 条目级二落点行 —— copy_in 的源行必须取条目记录
    （copy_out 发起登记 / register_entry 预登记，后者会被同条目 copy_out 的
    实际发起覆盖），目标行才是传入的 block_id；未知条目按 T-4 报错不静默，
    绝不退化为「拿 block_id 当源行」。"""
    copy_backend = FakeMluCopyBackend()
    adapter = MluTierBackend(copy_backend=copy_backend)
    bind_fake(adapter, copy_backend)

    # 形态一：copy_out 发起即登记落点行 5；换入交块给新行 9 —— 源 5 目标 9
    adapter.copy_out("e1", 5, 8 << 10, stream_id=0)
    adapter.copy_in("e1", 9, 8 << 10, stream_id=0)
    assert copy_backend.launches[-1] == ((5,), (9,), False, 2)

    # 形态二：外部 register_entry 预登记旧行 7（宿主接管既有级二副本形态）
    adapter.register_entry("e2", 7)
    adapter.copy_in("e2", 12, 8 << 10, stream_id=0)
    assert copy_backend.launches[-1] == ((7,), (12,), False, 3)

    # 形态三：同条目 copy_out 以实际发起覆盖预登记（落点以最后发起为准）
    adapter.register_entry("e3", 3)
    adapter.copy_out("e3", 6, 8 << 10, stream_id=0)
    adapter.copy_in("e3", 8, 8 << 10, stream_id=0)
    assert copy_backend.launches[-1] == ((6,), (8,), False, 5)

    # 形态四：未知条目按 T-4 报错不静默，绝不退化为 block_id 当源行
    from flagmemory.tiering.manager import TierError

    with pytest.raises(TierError, match="级二落点"):
        adapter.copy_in("ghost", 4, 8 << 10, stream_id=0)

    # 撤账后同条目再换入 = 回到未知形态，同样报错
    adapter.drop_entry("e1")
    with pytest.raises(TierError, match="级二落点"):
        adapter.copy_in("e1", 10, 8 << 10, stream_id=0)


def test_条目统计取不到给None_drop_entry只动账面():
    adapter = MluTierBackend(copy_backend=FakeMluCopyBackend())
    bind_fake(adapter, adapter._copy_backend)
    adapter.register_entry("e1", 7)
    assert adapter.block_activity(7) is None  # 退用其余判据并标降级
    adapter.drop_entry("e1")
    adapter.drop_entry("e1")  # 幂等


def test_shutdown传递给拷贝通路():
    copy_backend = FakeMluCopyBackend()
    adapter = MluTierBackend(copy_backend=copy_backend)
    bind_fake(adapter, copy_backend)
    adapter.shutdown()
    assert copy_backend.shutdown_called


def test_锁页探测映射():
    """探测到 = supports_pinned True；探不到（主机无 torch / torch_mlu、或
    pin_memory 抛错）= False 按降级登记上报 —— 探到哪级按哪级登记，不静默。
    （真环境锁页实测在 realcard 用例验。）"""
    orig = MluTierBackend._probe_pinned
    try:
        MluTierBackend._probe_pinned = lambda self: True
        a = MluTierBackend()
        assert a.probe_capabilities().supports_pinned is True

        MluTierBackend._probe_pinned = lambda self: False
        a = MluTierBackend()
        assert a.probe_capabilities().supports_pinned is False
    finally:
        MluTierBackend._probe_pinned = orig


def test_注册路径与降级登记(pool, manager):
    # backend_id 取池后端名（夹具池固定为 kunlunxin-like；接线层与栈无关，
    # 协议面验证对任意池后端名成立）
    backend_id = pool.query_capability().backend_name
    adapter = MluTierBackend(backend_id=backend_id)
    # 主机探测不到 torch（或 pin_memory 抛错）⇒ 强制该分支以钉降级路径
    adapter._probe_pinned = lambda: False
    registration = manager.register_tier_backend(backend_id, adapter)
    assert registration.backend_id == backend_id
    assert registration.probed.supports_pinned is False
    keys = [d.key for d in registration.degradations]
    assert "tier.pinned-unavailable" in keys
    assert "tier.batch-copy-unavailable" not in keys
    # 注册后管理器口径可查（T-9 生效值可查）
    view = manager.query_tier_capability()
    assert view.available and view.backend_id == backend_id


def test_注册helper默认取池后端名(pool, manager):
    registration = register_mlu_tier_backend(manager, pool)
    assert registration.backend_id == pool.query_capability().backend_name


def test_注册helper后端名不一致报错(pool, manager):
    from flagmemory.tiering.manager import TierError

    with pytest.raises(TierError, match="不一致"):
        register_mlu_tier_backend(manager, pool, backend_id="not-a-backend")
