"""Optional Ascend binding using only injected public framework interfaces.

The caller owns an initialized, stable framework context and retains streams
until the pool closes. Logical stream keys do not prove distinct hardware flows.
"""
from dataclasses import dataclass
from typing import Any, Callable

from flagmemory.adapters.device_context import DeviceContextBridge
from flagmemory.caps.backend import MemoryInfo
from flagmemory.pool import InvalidArgument
from flagmemory.pool.base import BaseAllocator


def _index(value: Any) -> bool:
    return type(value) is int and value >= 0


@dataclass(frozen=True)
class AscendProfile:
    bridge: DeviceContextBridge
    allocator: BaseAllocator
    new_stream: Callable[[], Any]


def make_ascend_profile(*, torch_module: Any, backend: Any, stream_type: type,
                        event_type: type, device_index: int, byte_limit: int) -> AscendProfile:
    if not _index(device_index) or type(byte_limit) is not int or byte_limit <= 0:
        raise InvalidArgument('invalid device index or byte limit')
    npu = torch_module.npu
    if npu.is_initialized() is not True:
        raise InvalidArgument('framework must already be initialized')

    def current() -> int:
        if npu.is_initialized() is not True:
            raise InvalidArgument('framework must remain initialized')
        value = npu.current_device()
        if not _index(value):
            raise InvalidArgument('invalid current device')
        return value

    def check() -> None:
        if current() != device_index:
            raise InvalidArgument('current device differs from profile')

    class Identity:
        def __init__(self) -> None:
            self.entries: list[tuple[Any, int]] = []

        def register(self, raw: Any) -> None:
            if not isinstance(raw, npu.Stream):
                raise InvalidArgument('unknown native stream type')
            dev = raw.device
            if dev.type != 'npu' or not _index(dev.index) or dev.index != device_index:
                raise InvalidArgument('native stream belongs to another device')
            check()
            if not any(old is raw for old, _ in self.entries):
                self.entries.append((raw, len(self.entries) + 1))

        def lookup(self, stream: Any) -> int:
            if not isinstance(stream, stream_type) or stream.backend is not backend:
                raise InvalidArgument('foreign unified stream')
            raw = stream.native
            if not isinstance(raw, npu.Stream):
                raise InvalidArgument('unknown native stream type')
            dev = raw.device
            if dev.type != 'npu' or not _index(dev.index) or dev.index != device_index:
                raise InvalidArgument('native stream belongs to another device')
            for old, key in self.entries:
                if old is raw:
                    check()
                    return key
            raise InvalidArgument('unregistered native stream')

        def stream_device(self, stream: Any) -> int:
            self.lookup(stream)
            return device_index

        def stream_key(self, stream: Any) -> int:
            return self.lookup(stream)

    identity = Identity()
    check()
    raw_default = backend.current_stream()
    check()
    identity.register(raw_default)
    default = stream_type(backend, raw_default)
    check()

    def new_stream() -> Any:
        check()
        raw = backend.create_stream()
        check()
        identity.register(raw)
        result = stream_type(backend, raw)
        identity.lookup(result)
        check()
        return result

    def make_event(index: int) -> Any:
        if index != device_index:
            raise InvalidArgument('wrong event device')
        check()
        raw = backend.create_event(enable_timing=False)
        check()
        event = event_type(backend, raw)
        check()
        return event

    def memory_info(index: int) -> MemoryInfo:
        if index != device_index:
            raise InvalidArgument('wrong memory device')
        check()
        result = npu.mem_get_info(index)
        check()
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise InvalidArgument('expected free and total byte counts')
        free, total = result
        if not _index(free) or not _index(total) or free > total:
            raise InvalidArgument('invalid byte counts')
        return MemoryInfo(total=total, used=total - free, free=free)

    bridge = DeviceContextBridge(backend, device_index=device_index, default_stream=default,
        identity=identity, make_event=make_event, current_device=current,
        memory_info_provider=memory_info)

    class Allocator:
        def __init__(self) -> None:
            self.live: list[Any] = []
            self.consumed = 0

        def allocate(self, nbytes: int, device_index: int = 0, *, stream: Any) -> Any:
            if type(nbytes) is not int or nbytes <= 0:
                raise InvalidArgument('allocation size must be positive integer')
            bridge.validate_stream_device(stream, device_index)
            if self.consumed + nbytes > byte_limit:
                raise MemoryError('allocator cumulative byte limit exceeded')
            check()
            try:
                with stream.context():
                    bridge.validate_stream_device(stream, device_index)
                    tensor = torch_module.empty(nbytes, dtype=torch_module.uint8,
                                                device=f'npu:{device_index}')
                    bridge.validate_stream_device(stream, device_index)
            except Exception as exc:
                oom = getattr(torch_module, 'OutOfMemoryError', None)
                if (not isinstance(exc, MemoryError) and isinstance(oom, type)
                        and issubclass(oom, RuntimeError) and isinstance(exc, oom)):
                    raise MemoryError('framework allocation failed') from exc
                raise
            dev = tensor.device
            if (dev.type != 'npu' or dev.index != device_index or
                    type(tensor.numel()) is not int or type(tensor.element_size()) is not int or
                    tensor.numel() * tensor.element_size() < nbytes):
                raise InvalidArgument('framework returned an invalid tensor')
            bridge.validate_stream_device(stream, device_index)
            self.live.append(tensor)
            self.consumed += nbytes
            return tensor

        def release(self, base_ref: Any) -> None:
            for pos, tensor in enumerate(self.live):
                if tensor is base_ref:
                    del self.live[pos]
                    return
            raise InvalidArgument('unknown or already returned tensor')

    return AscendProfile(bridge, Allocator(), new_stream)
