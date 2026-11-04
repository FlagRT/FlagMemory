"""Test-only candidate adapter; runtime semantics require on-device validation."""
from dataclasses import dataclass
from typing import Any
from flagmemory.caps.backend import CapabilityNotDeclared, EventNotRecorded, MemoryInfo
from flagmemory.caps.capabilities import Capability, L0_CAPABILITIES

MAX_BYTES = 1 << 30

def validate_stream(stream, index):
    device = getattr(stream, 'device', None)
    if getattr(device, 'type', None) != 'npu' or getattr(device, 'index', None) != index:
        raise ValueError('stream device is unknown or does not match')

@dataclass
class RecordedEvent:
    raw: Any
    device: int
    recorded: bool = False
    stream: Any = None

class AscendTestBackend:
    name = 'ascend-test-only'
    def __init__(self, runtime, device_index):
        self.runtime = runtime; self.device_index = device_index; self.events = []
    def _device(self, index):
        if index != self.device_index: raise ValueError('device mismatch')
    def declared_capabilities(self): return frozenset(L0_CAPABILITIES)
    def device_memory_info(self, device_index=0):
        self._device(device_index)
        free, total = self.runtime.npu.mem_get_info(device_index)
        return MemoryInfo(total=int(total), used=int(total-free), free=int(free))
    def current_stream(self, device_index=0):
        self._device(device_index)
        result = self.runtime.npu.current_stream(device_index)
        self.validate_stream_device(result, device_index)
        return result
    def validate_stream_device(self, stream, device_index):
        self._device(device_index); validate_stream(stream, device_index)
    def streams_equal(self, left, right): return left == right
    def create_event(self, device_index=0):
        self._device(device_index)
        event = RecordedEvent(self.runtime.npu.Event(), device_index)
        self.events.append(event)
        return event
    def record_event(self, event, stream):
        self.validate_stream_device(stream, event.device)
        if event.recorded: raise RuntimeError('event already recorded')
        event.raw.record(stream)
        event.recorded = True; event.stream = stream
    def query_event(self, event):
        if not event.recorded: raise EventNotRecorded('event not recorded')
        return bool(event.raw.query())
    def synchronize_event(self, event, timeout_s):
        raise RuntimeError('bounded event synchronization unavailable')
    def alloc_handle(self, *args, **kwargs):
        raise CapabilityNotDeclared(Capability.MEMORY_HANDLE, self.name)
    def free_handle(self, *args, **kwargs):
        raise CapabilityNotDeclared(Capability.MEMORY_HANDLE, self.name)

class AscendTensorAllocator:
    def __init__(self, runtime, device_index, byte_limit):
        if not isinstance(byte_limit, int) or isinstance(byte_limit, bool) or not 0 < byte_limit <= MAX_BYTES:
            raise ValueError('byte limit must be in (0, 1 GiB]')
        self.runtime = runtime; self.device_index = device_index; self.byte_limit = byte_limit
        self.live = {}; self.live_bytes = 0; self.allocated_bytes = 0
        self.allocate_calls = 0; self.release_calls = 0
    def allocate(self, nbytes, device_index=0, *, stream):
        if device_index != self.device_index: raise ValueError('device mismatch')
        validate_stream(stream, device_index)
        if not isinstance(nbytes, int) or isinstance(nbytes, bool) or nbytes <= 0: raise ValueError('invalid size')
        if self.allocated_bytes + nbytes > self.byte_limit: raise MemoryError('cumulative allocation quota')
        with self.runtime.npu.device(device_index), self.runtime.npu.stream(stream):
            tensor = self.runtime.empty(nbytes, dtype=self.runtime.uint8, device=f'npu:{device_index}')
        self.live[id(tensor)] = (tensor, nbytes)
        self.live_bytes += nbytes; self.allocated_bytes += nbytes; self.allocate_calls += 1
        return tensor
    def tensor(self, ref):
        item = self.live.get(id(ref))
        if item is None or item[0] is not ref: raise RuntimeError('unknown or released tensor')
        return ref
    def identity(self, ref): return id(self.tensor(ref))
    def release(self, ref):
        self.tensor(ref)
        _, nbytes = self.live.pop(id(ref))
        self.live_bytes -= nbytes; self.release_calls += 1
