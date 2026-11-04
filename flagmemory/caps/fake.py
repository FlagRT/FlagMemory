"""纯内存假后端：给单测与无卡环境用（零框架依赖，不碰任何厂商运行时）。

它**不是**模拟某款芯片的行为，只做两件事：
1. 把「声明了哪些能力」变成可配置的，于是能表达「某项能力未声明」；
2. 对未声明能力的调用**抛 `CapabilityNotDeclared`**，把「不得静默退化」这条纪律变成可测的。

真后端（对接设备方向统一后端接口）是后续任务，不在本次改造范围内。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .backend import (
    CapabilityNotDeclared,
    DoubleFree,
    EventNotRecorded,
    MemoryHandle,
    MemoryInfo,
)
from .capabilities import (
    Capability,
    L0_CAPABILITIES,
    L1_CAPABILITIES,
    L2_CAPABILITIES,
)

__all__ = [
    "FakeEvent",
    "FakeStream",
    "FakeBackend",
    "ASCEND_LIKE",
    "KUNLUNXIN_LIKE",
    "CAMBRICON_LIKE",
    "PPU_LIKE",
    "CHIP_PRESETS",
]


@dataclass(eq=False)
class FakeStream:
    """假流。池必须强持有它（不变量三），所以这里刻意不做任何弱引用自动清理。"""

    device_index: int = 0
    stream_id: int = 0


@dataclass(eq=False)
class FakeEvent:
    """假事件。`recorded` 为 False 时查询必须报错，不得返回「已完成」（不变量二）。"""

    device_index: int = 0
    recorded: bool = False
    stream: FakeStream | None = None
    completed: bool = False


# —— 四家的能力声明预设（照抄调研 §5.1 的分级，不重新论证）——

#: 昇腾 910C：L0 + L1 + L2 + L3 全声明（唯一有真有界同步与数值错误码表的一家）。
ASCEND_LIKE = frozenset(L0_CAPABILITIES | L1_CAPABILITIES | L2_CAPABILITIES | {
    Capability.TRUE_BOUNDED_SYNC,
    Capability.VENDOR_ERROR_CODE_TABLE,
    Capability.DEVICE_LEVEL_REBUILD,
})

#: 昆仑芯 P800：L0 + L1；无上下文生命周期（平台只允许一个上下文）、无 L3。
KUNLUNXIN_LIKE = frozenset(L0_CAPABILITIES | L1_CAPABILITIES)

#: 寒武纪 MLU590：只有 L0 —— L1 四项全未声明（池在它上面走张量级池 + 自己的登记表）。
CAMBRICON_LIKE = frozenset(L0_CAPABILITIES)

#: 平头哥 PPU：L0 + L1 + 上下文生命周期（L2）；无 L3。
PPU_LIKE = frozenset(L0_CAPABILITIES | L1_CAPABILITIES | L2_CAPABILITIES)

CHIP_PRESETS = {
    "ascend-like": ASCEND_LIKE,
    "kunlunxin-like": KUNLUNXIN_LIKE,
    "cambricon-like": CAMBRICON_LIKE,
    "ppu-like": PPU_LIKE,
}


@dataclass
class FakeBackend:
    """可配置能力声明的纯内存后端，满足 `flagmemory.caps.backend.DeviceBackend`。"""

    name: str = "fake"
    declared: frozenset[Capability] = field(default_factory=lambda: frozenset(L0_CAPABILITIES))
    total_bytes: int = 64 << 30
    used_bytes: int = 0
    device_count: int = 1
    #: 句柄登记表：池的「回收不依赖厂商报错」要靠记账拦截（不变量一），假后端同样记账。
    _handles: dict[int, MemoryHandle] = field(default_factory=dict, repr=False)
    _next_handle_id: int = field(default=1, repr=False)

    # —— 声明 ——

    def declared_capabilities(self) -> frozenset[Capability]:
        return frozenset(self.declared)

    def _require(self, cap: Capability) -> None:
        if cap not in self.declared:
            raise CapabilityNotDeclared(cap, self.name)

    # —— L0：设备显存三键 ——

    def device_memory_info(self, device_index: int = 0) -> MemoryInfo:
        self._require(Capability.DEVICE_MEMORY_INFO)
        self._check_device(device_index)
        return MemoryInfo(
            total=self.total_bytes,
            used=self.used_bytes,
            free=self.total_bytes - self.used_bytes,
        )

    def _check_device(self, device_index: int) -> None:
        if device_index != 0:
            self._require(Capability.MULTI_DEVICE)
        if not 0 <= device_index < self.device_count:
            raise IndexError(f"设备序号越界：{device_index}（共 {self.device_count} 张）")

    # —— L0：流 ——

    def current_stream(self, device_index: int = 0) -> FakeStream:
        self._require(Capability.STREAM)
        self._check_device(device_index)
        return FakeStream(device_index=device_index, stream_id=0)

    def validate_stream_device(self, stream: FakeStream, device_index: int) -> None:
        self._require(Capability.STREAM)
        self._check_device(device_index)
        if not isinstance(stream, FakeStream) or stream.device_index != device_index:
            raise ValueError("流不属于目标设备")

    def streams_equal(self, left: FakeStream, right: FakeStream) -> bool:
        self._require(Capability.STREAM)
        return (left.device_index, left.stream_id) == (right.device_index, right.stream_id)

    # —— L0：事件 ——

    def create_event(self, device_index: int = 0) -> FakeEvent:
        self._require(Capability.EVENT)
        self._check_device(device_index)
        return FakeEvent(device_index=device_index)

    def record_event(self, event: FakeEvent, stream: FakeStream) -> None:
        self._require(Capability.EVENT)
        event.recorded = True
        event.stream = stream

    def query_event(self, event: FakeEvent) -> bool:
        self._require(Capability.EVENT)
        if not event.recorded:
            # 不变量二：寒武纪与平头哥原生会对未 record 的事件误报「已完成」，
            # 适配层必须自己拦住，绝不把那个 True 递给池。
            raise EventNotRecorded(
                "事件未 record 就查询完成状态：原生实现会误报「已完成」，此处按纪律报错。"
            )
        return event.completed

    def synchronize_event(self, event: FakeEvent, timeout_s: float) -> bool:
        self._require(Capability.BOUNDED_SYNC)
        if not event.recorded:
            raise EventNotRecorded("事件未 record 就同步。")
        if event.completed:
            return True
        # 真有界只有昇腾；其余三家是超时上报。超时不等于失败（降级路径三）。
        return False

    # —— L1（可选）：统一内存句柄 ——

    def alloc_handle(
        self,
        nbytes: int,
        device_index: int = 0,
        kind: str = "device",
        purpose: str | None = None,
        stream_id: int | None = None,
    ) -> MemoryHandle:
        self._require(Capability.MEMORY_HANDLE)
        self._check_device(device_index)
        if nbytes <= 0:
            raise ValueError(f"申请字节数必须为正：{nbytes}")
        if nbytes > self.total_bytes - self.used_bytes:
            raise MemoryError(
                f"显存不足：请求 {nbytes} 字节，剩余 {self.total_bytes - self.used_bytes} 字节"
            )
        handle = MemoryHandle(
            handle_id=self._next_handle_id,
            kind=kind,
            backend=self.name,
            device_index=device_index,
            nbytes=nbytes,
            purpose=purpose,
            stream_id=stream_id,
        )
        self._next_handle_id += 1
        self._handles[handle.handle_id] = handle
        self.used_bytes += nbytes
        return handle

    def free_handle(self, handle: MemoryHandle) -> None:
        self._require(Capability.MEMORY_HANDLE)
        # 不变量一：成对释放，重复释放与非本层句柄必须报错（厂商行为相反，统一层自己拦）。
        known = self._handles.pop(handle.handle_id, None)
        if known is None:
            raise DoubleFree(
                f"句柄 {handle.handle_id} 不在本层登记表里（重复释放或非本层句柄）。"
            )
        self.used_bytes -= known.nbytes

    def live_handles(self) -> int:
        """在世句柄数（审计用，泄漏判据入口）。"""
        self._require(Capability.MEMORY_HANDLE)
        return len(self._handles)
