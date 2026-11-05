"""后端协议面与纯内存假后端。全部不依赖真卡（假后端刻意零框架依赖）。"""

from __future__ import annotations

import pytest

from flagmemory.caps.backend import (
    CapabilityNotDeclared,
    DeviceBackend,
    DoubleFree,
    EventNotRecorded,
    MemoryHandle,
    MemoryInfo,
)
from flagmemory.caps.capabilities import Capability, L0_CAPABILITIES
from flagmemory.caps.fake import FakeBackend


class TestNormal:
    """正常：协议面的每个口在声明了能力的后端上都能用。"""

    def test_假后端满足协议(self, handle_backend):
        assert isinstance(handle_backend, DeviceBackend)

    def test_显存三键(self, l0_only_backend):
        info = l0_only_backend.device_memory_info()
        assert isinstance(info, MemoryInfo)
        assert info.total == (64 << 30)
        assert info.used == 0
        assert info.free == info.total

    def test_流的获取与比较(self, l0_only_backend):
        a = l0_only_backend.current_stream()
        b = l0_only_backend.current_stream()
        # 两个流对象不是同一个 Python 对象，但是同一条流 —— 必须走 streams_equal 判。
        assert a is not b
        assert l0_only_backend.streams_equal(a, b)

    def test_事件先record再查询(self, l0_only_backend):
        ev = l0_only_backend.create_event()
        stream = l0_only_backend.current_stream()
        l0_only_backend.record_event(ev, stream)
        assert l0_only_backend.query_event(ev) is False
        ev.completed = True
        assert l0_only_backend.query_event(ev) is True

    def test_有界同步完成返回True(self, l0_only_backend):
        ev = l0_only_backend.create_event()
        l0_only_backend.record_event(ev, l0_only_backend.current_stream())
        ev.completed = True
        assert l0_only_backend.synchronize_event(ev, timeout_s=0.01) is True

    def test_句柄申请与成对释放_五个公共字段在位(self, handle_backend):
        h = handle_backend.alloc_handle(1 << 20, kind="device", purpose="KV缓存", stream_id=0)
        assert isinstance(h, MemoryHandle)
        assert (h.handle_id, h.kind, h.backend, h.device_index, h.nbytes) == (
            1,
            "device",
            "with-handle",
            0,
            1 << 20,
        )
        assert h.purpose == "KV缓存"
        assert handle_backend.live_handles() == 1
        assert handle_backend.device_memory_info().used == (1 << 20)
        handle_backend.free_handle(h)
        assert handle_backend.live_handles() == 0
        assert handle_backend.device_memory_info().used == 0

    def test_句柄不暴露裸指针(self):
        # 统一层纪律：厂商指针不进句柄。协议里不该有这么个字段。
        assert not any(
            name in MemoryHandle.__dataclass_fields__
            for name in ("ptr", "pointer", "address", "devptr")
        )


class TestBoundary:
    """边界：零值 / 满值 / 未声明用途 / 多卡开关。"""

    def test_三键零值(self):
        backend = FakeBackend(name="empty", declared=frozenset(L0_CAPABILITIES), total_bytes=0)
        info = backend.device_memory_info()
        assert (info.total, info.used, info.free) == (0, 0, 0)

    def test_三键满值_空闲为零(self, handle_backend):
        total = handle_backend.total_bytes
        h = handle_backend.alloc_handle(total)
        info = handle_backend.device_memory_info()
        assert (info.used, info.free) == (total, 0)
        # 满了之后再申请一个字节必须报显存不足，不许透支。
        with pytest.raises(MemoryError):
            handle_backend.alloc_handle(1)
        handle_backend.free_handle(h)
        assert handle_backend.device_memory_info().free == total

    def test_未声明用途的句柄_purpose为None(self, handle_backend):
        h = handle_backend.alloc_handle(4096)
        # 池对 purpose=None 按最保守处理（不回收、不复用），这里只钉协议的缺省值。
        assert h.purpose is None
        assert h.stream_id is None

    def test_单卡后端取0号卡可以_取1号卡越界(self, l0_only_backend):
        assert l0_only_backend.current_stream(0).device_index == 0
        with pytest.raises(IndexError):
            l0_only_backend.current_stream(1)

    def test_多卡后端的卡号边界(self):
        backend = FakeBackend(
            name="multi", declared=frozenset(L0_CAPABILITIES), device_count=8
        )
        assert backend.current_stream(7).device_index == 7
        with pytest.raises(IndexError):
            backend.current_stream(8)

    def test_同一能力在声明与未声明两种后端上的分支走法(self, l0_only_backend, handle_backend):
        # L1 能力的边界：同一句调用，一边能用、一边必须报错。池就是靠这个分支决定走不走句柄形态。
        assert handle_backend.alloc_handle(1024).nbytes == 1024
        with pytest.raises(CapabilityNotDeclared):
            l0_only_backend.alloc_handle(1024)

    def test_四家预设各自的能力声明都能查(self, chip_like_backend):
        declared = chip_like_backend.declared_capabilities()
        assert L0_CAPABILITIES <= declared  # L0 四家齐备
        assert isinstance(declared, frozenset)


class TestError:
    """异常：未声明能力必须报错而不是静默退化；三条不变量必须拦住。"""

    @pytest.mark.parametrize(
        "cap",
        [
            Capability.MEMORY_HANDLE,
        ],
    )
    def test_未声明能力被调用时报错(self, l0_only_backend, cap):
        with pytest.raises(CapabilityNotDeclared) as excinfo:
            l0_only_backend.alloc_handle(1024)
        assert excinfo.value.capability is cap
        assert excinfo.value.backend_name == "l0-only"
        assert "不得静默退化" in str(excinfo.value)

    def test_未声明时不许返回空值假装成功(self, l0_only_backend):
        # 「静默退化」的典型形态就是返回 None / 0 / False。这里断言它抛，而不是返回。
        try:
            result = l0_only_backend.free_handle(
                MemoryHandle(1, "device", "l0-only", 0, 1024)
            )
        except CapabilityNotDeclared:
            pass
        else:  # pragma: no cover - 走到这里就是回归
            pytest.fail(f"未声明的能力静默返回了 {result!r}，违反统一层纪律")

    def test_事件未record就查询必须报错(self, l0_only_backend):
        # 不变量二：寒武纪与平头哥原生会误报「已完成」，适配层必须自己拦。
        ev = l0_only_backend.create_event()
        with pytest.raises(EventNotRecorded):
            l0_only_backend.query_event(ev)
        with pytest.raises(EventNotRecorded):
            l0_only_backend.synchronize_event(ev, timeout_s=0.01)

    def test_二次释放必须报错_靠自己记账(self, handle_backend):
        # 不变量一：厂商行为四家相反（昇腾静默成功、昆仑芯报错），所以本层自己记账拦。
        h = handle_backend.alloc_handle(2048)
        handle_backend.free_handle(h)
        with pytest.raises(DoubleFree):
            handle_backend.free_handle(h)

    def test_释放非本层句柄必须报错(self, handle_backend):
        alien = MemoryHandle(99, "device", "someone-else", 0, 2048)
        with pytest.raises(DoubleFree):
            handle_backend.free_handle(alien)

    def test_超时未完成返回False而不是抛异常(self, l0_only_backend):
        # 降级路径三：超时不等于失败，要作可恢复态上报 ⇒ 这里必须是 False，不是异常。
        ev = l0_only_backend.create_event()
        l0_only_backend.record_event(ev, l0_only_backend.current_stream())
        assert l0_only_backend.synchronize_event(ev, timeout_s=0.0) is False

    def test_申请非正字节数报错(self, handle_backend):
        for bad in (0, -1):
            with pytest.raises(ValueError):
                handle_backend.alloc_handle(bad)

    def test_三键不接受负数与非整数(self):
        with pytest.raises(ValueError):
            MemoryInfo(total=1, used=0, free=-1)
        with pytest.raises(TypeError):
            MemoryInfo(total=1.5, used=0, free=1)  # type: ignore[arg-type]
