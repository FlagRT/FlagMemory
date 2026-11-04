"""Single-device bridge over injected public device objects.

Identity and byte providers are mandatory trust boundaries. The bridge never
infers an arbitrary stream's device, refreshes the explicit default stream, or
observes external context identity. See docs/impl/设备统一接口张量桥接.md.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Protocol

from flagmemory.caps.backend import CapabilityNotDeclared, EventNotRecorded, MemoryHandle, MemoryInfo
from flagmemory.caps.capabilities import Capability
from flagmemory.pool.types import InvalidArgument
from flagmemory.pool.barrier import validate_timeout

__all__ = ["DeviceIdentity", "DeviceContextBridge"]

_CAPABILITIES = {
    "memory": Capability.DEVICE_MEMORY_INFO,
    "stream": Capability.STREAM,
    "event": Capability.EVENT,
    "bounded_sync": Capability.BOUNDED_SYNC,
    "graph_capture": Capability.GRAPH_CAPTURE,
    "recovery_probe": Capability.PROBE_LEVEL_RECOVERY,
    "multidevice": Capability.MULTI_DEVICE,
    "stream_priority": Capability.STREAM_PRIORITY_RANGE,
}


class DeviceIdentity(Protocol):
    """Injected public-source proof; unknown objects must return None or fail.

    Keys identify controlled logical streams, not necessarily distinct hardware
    streams. A key is only compared after backend and device verification.
    """
    def stream_device(self, stream: Any) -> int | None: ...
    def stream_key(self, stream: Any) -> Hashable | None: ...


@dataclass(eq=False)
class _BridgeEvent:
    owner: Any
    device_index: int
    public: Any
    recorded: bool = False
    attempted: bool = False
    stream: Any = None


class DeviceContextBridge:
    def __init__(self, backend: Any, *, device_index: int,
                 default_stream: Any, identity: DeviceIdentity | None,
                 make_event: Callable[[int], Any],
                 current_device: Callable[[], int],
                 memory_info_provider: Callable[[int], MemoryInfo]) -> None:
        self._index(device_index)
        if identity is None or not all(callable(getattr(identity, name, None))
                                       for name in ("stream_device", "stream_key")):
            raise InvalidArgument("a public-source stream identity provider is required")
        if not all(callable(provider) for provider in
                   (make_event, current_device, memory_info_provider)):
            raise InvalidArgument("event, current-device and byte providers are required")
        if not isinstance(backend.name, str) or not backend.name:
            raise InvalidArgument("backend.name must be nonempty text")
        self._backend = backend
        self._device_index = device_index
        self._identity = identity
        self._make_event = make_event
        self._current_device = current_device
        self._memory_info_provider = memory_info_provider
        declared = set()
        for key, cap in _CAPABILITIES.items():
            supported = self._invoke(cap, backend.supports, key)
            if type(supported) is not bool:
                raise InvalidArgument("backend.supports must return bool")
            if supported:
                declared.add(cap)
        self._declared = frozenset(declared)
        self._check_device(device_index, Capability.STREAM)
        self._stream_key(default_stream)
        self._default_stream = default_stream
        if Capability.DEVICE_MEMORY_INFO in self._declared:
            self.device_memory_info(device_index)

    @property
    def name(self) -> str:
        return self._backend.name

    def declared_capabilities(self) -> frozenset[Capability]:
        return self._declared

    def _require(self, cap: Capability) -> None:
        if cap not in self._declared:
            raise CapabilityNotDeclared(cap, self.name)

    def _invoke(self, cap: Capability, function: Callable[..., Any], *args: Any) -> Any:
        try:
            return function(*args)
        except NotImplementedError as exc:
            raise CapabilityNotDeclared(cap, self.name) from exc
        except ValueError as exc:
            raise InvalidArgument(str(exc)) from exc

    @staticmethod
    def _index(value: Any) -> None:
        if type(value) is not int or value < 0:
            raise InvalidArgument("device index must be a nonnegative int, not bool")

    def _check_device(self, device_index: int, cap: Capability) -> None:
        self._index(device_index)
        if device_index != self._device_index:
            raise InvalidArgument("bridge is bound to a different device")
        actual = self._invoke(cap, self._current_device)
        self._index(actual)
        if actual != self._device_index:
            raise InvalidArgument("current device does not match bridge binding")

    def _stream_key(self, stream: Any) -> Hashable:
        if getattr(stream, "backend", None) is not self._backend:
            raise InvalidArgument("expected a unified stream from this backend instance")
        device = self._invoke(Capability.STREAM, self._identity.stream_device, stream)
        self._index(device)
        if device != self._device_index:
            raise InvalidArgument("stream does not belong to the bound device")
        key = self._invoke(Capability.STREAM, self._identity.stream_key, stream)
        if key is None or (isinstance(key, (str, bytes, tuple, frozenset)) and not key):
            raise InvalidArgument("stream needs a nonempty stable identity key")
        try:
            hash(key)
        except TypeError as exc:
            raise InvalidArgument("stream identity key must be hashable") from exc
        return key

    def device_memory_info(self, device_index: int = 0) -> MemoryInfo:
        self._require(Capability.DEVICE_MEMORY_INFO)
        self._check_device(device_index, Capability.DEVICE_MEMORY_INFO)
        info = self._invoke(Capability.DEVICE_MEMORY_INFO, self._memory_info_provider, device_index)
        self._check_device(device_index, Capability.DEVICE_MEMORY_INFO)
        if not isinstance(info, MemoryInfo):
            raise InvalidArgument("byte provider must return MemoryInfo")
        for field in ("total", "used", "free"):
            value = getattr(info, field, None)
            if type(value) is not int or value < 0:
                raise InvalidArgument("MemoryInfo values must be nonnegative byte integers")
        if info.free > info.total or info.used != info.total - info.free:
            raise InvalidArgument("byte provider must report free<=total and used=total-free")
        return info

    def current_stream(self, device_index: int = 0) -> Any:
        self._require(Capability.STREAM)
        self._check_device(device_index, Capability.STREAM)
        self._stream_key(self._default_stream)
        return self._default_stream

    def validate_stream_device(self, stream: Any, device_index: int) -> None:
        self._require(Capability.STREAM)
        self._check_device(device_index, Capability.STREAM)
        self._stream_key(stream)

    def streams_equal(self, left: Any, right: Any) -> bool:
        self._require(Capability.STREAM)
        self._check_device(self._device_index, Capability.STREAM)
        left_key, right_key = self._stream_key(left), self._stream_key(right)
        if left is right:
            return True
        return bool(left_key == right_key)

    def create_event(self, device_index: int = 0) -> Any:
        self._require(Capability.EVENT)
        self._check_device(device_index, Capability.EVENT)
        public = self._invoke(Capability.EVENT, self._make_event, device_index)
        self._check_device(device_index, Capability.EVENT)
        if getattr(public, "backend", None) is not self._backend:
            raise InvalidArgument("event factory returned a different backend instance")
        if not all(callable(getattr(public, name, None)) for name in ("record", "query")):
            raise InvalidArgument("event factory must return a public Event")
        # This is a creation constraint, not proof of native event device binding.
        # On lazy native events the first successful explicit record binds it.
        return _BridgeEvent(self, device_index, public)

    def _event(self, event: Any) -> _BridgeEvent:
        self._require(Capability.EVENT)
        self._check_device(self._device_index, Capability.EVENT)
        if (not isinstance(event, _BridgeEvent) or event.owner is not self
                or event.device_index != self._device_index
                or event.public.backend is not self._backend):
            raise InvalidArgument("event is not owned by this bridge and device")
        return event

    def record_event(self, event: Any, stream: Any) -> None:
        owned = self._event(event)
        self.validate_stream_device(stream, self._device_index)
        if owned.attempted:
            raise InvalidArgument("event record may only be attempted once")
        # Keep evidence even if record has side effects and then raises.
        owned.attempted = True
        owned.stream = stream
        self._invoke(Capability.EVENT, owned.public.record, stream)
        self._check_device(self._device_index, Capability.EVENT)
        owned.recorded = True

    def query_event(self, event: Any) -> bool:
        owned = self._event(event)
        if not owned.recorded:
            raise EventNotRecorded("event has not been successfully recorded")
        result = self._invoke(Capability.EVENT, owned.public.query)
        self._check_device(self._device_index, Capability.EVENT)
        if type(result) is not bool:
            raise InvalidArgument("Event.query must return bool")
        return result

    def synchronize_event(self, event: Any, timeout_s: float) -> bool:
        self._require(Capability.BOUNDED_SYNC)
        budget = validate_timeout(timeout_s)
        deadline = time.monotonic() + budget
        while True:
            if self.query_event(event):
                return True
            if budget == 0:
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.001, remaining))

    def alloc_handle(self, nbytes: int, device_index: int = 0,
                     kind: str = "device", purpose: str | None = None,
                     stream_id: int | None = None) -> MemoryHandle:
        raise CapabilityNotDeclared(Capability.MEMORY_HANDLE, self.name)

    def free_handle(self, handle: MemoryHandle) -> None:
        raise CapabilityNotDeclared(Capability.MEMORY_HANDLE, self.name)
