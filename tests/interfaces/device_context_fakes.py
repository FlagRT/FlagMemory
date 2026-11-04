"""Public device API shaped doubles; identity truth lives outside device objects.

No pool FakeBackend inheritance. These doubles model call shapes, never hardware
semantics. In particular, native Python identity is not a hardware stream ID.
"""
from contextlib import contextmanager
import weakref
from flagmemory.caps.backend import MemoryInfo

CORE_KEYS = {"memory", "stream", "event", "bounded_sync", "graph_capture",
             "recovery_probe", "multidevice", "stream_priority"}


class NativeStream:
    pass


class UnifiedStream:
    def __init__(self, backend, native):
        self.backend = backend
        object.__setattr__(self, "_test_native", native)

    @property
    def native(self):
        self.backend.audit.append(("native", self))
        return object.__getattribute__(self, "_test_native")

    def context(self):
        return self.backend.stream_context(self.native)

    def __getattribute__(self, name):
        if name.startswith("_") and not name.startswith("__"):
            raise AssertionError(f"private stream access: {name}")
        return object.__getattribute__(self, name)

    def __getattr__(self, name):
        raise AssertionError(f"nonpublic or forbidden stream access: {name}")


class NativeEvent:
    def __init__(self):
        self.completed = False
        self.recorded_stream = None
        self.record_error = None
        self.query_error = None


class UnifiedEvent:
    def __init__(self, backend, native):
        self.backend = backend
        self.native = native

    def record(self, stream):
        self.backend.audit.append(("record", self, stream))
        if self.native.record_error:
            raise self.native.record_error
        self.native.recorded_stream = stream.native

    def query(self):
        self.backend.audit.append(("query", self))
        if self.native.query_error:
            raise self.native.query_error
        # Some native implementations report unrecorded events complete.
        return self.native.recorded_stream is None or self.native.completed

    def wait_host(self, timeout_ms):
        self.backend.forbidden.append("wait_host")
        raise AssertionError("bridge must poll, never wait_host")

    def __getattr__(self, name):
        self.backend.forbidden.append(name)
        raise AssertionError(f"nonpublic or forbidden event access: {name}")


class DeviceBackend:
    name = "public-device-double"

    def __init__(self, supported=None):
        self.supported = set(CORE_KEYS if supported is None else supported)
        self.audit = []
        self.forbidden = []
        self.events = []
        self.default_native = NativeStream()

    def supports(self, capability):
        self.audit.append(("supports", capability))
        return capability in self.supported

    def current_stream(self):
        self.audit.append(("current_stream",))
        return self.default_native

    def create_event(self, enable_timing=False):
        assert enable_timing is False
        native = NativeEvent()
        self.events.append(native)
        self.audit.append(("create_event", enable_timing))
        return native

    @contextmanager
    def stream_context(self, native):
        self.audit.append(("stream_context", native))
        yield

    def memory_stats(self, device_index=0):
        self.forbidden.append("memory_stats")
        raise AssertionError("quantized/fallback memory_stats is not exact bytes")

    def __getattr__(self, name):
        self.forbidden.append(name)
        raise AssertionError(f"nonpublic or forbidden backend access: {name}")


class TruthIdentity:
    """Test dependency only, not a production proof of identity."""
    def __init__(self):
        self.truth = {}

    def add(self, stream, device, key):
        self.truth[stream] = (device, key)
        return stream

    def stream_device(self, stream):
        return self.truth.get(stream, (None, None))[0]

    def stream_key(self, stream):
        return self.truth.get(stream, (None, None))[1]


class Bundle:
    def __init__(self, *, supported=None, device=2):
        self.backend = DeviceBackend(supported)
        self.identity = TruthIdentity()
        self.device = device
        self.current = device
        self.current_error = None
        self.memory_error = None
        self.memory_value = MemoryInfo(total=1000, used=877, free=123)
        self.memory_calls = []
        self.event_calls = []
        self.default = self.identity.add(
            UnifiedStream(self.backend, self.backend.current_stream()), device, "default")
        self.stream2 = self.stream(key="borrower")

    def stream(self, *, device=None, key="other", backend=None):
        return self.identity.add(UnifiedStream(backend or self.backend, NativeStream()),
                                 self.device if device is None else device, key)

    def current_device(self):
        if self.current_error:
            raise self.current_error
        return self.current

    def memory_info(self, index):
        self.memory_calls.append(index)
        if self.memory_error:
            raise self.memory_error
        return self.memory_value

    def make_event(self, index):
        self.event_calls.append(index)
        # Factory creates an unbound native event; record binds its domain.
        return UnifiedEvent(self.backend, self.backend.create_event(enable_timing=False))

    def kwargs(self):
        return dict(device_index=self.device, default_stream=self.default,
                    identity=self.identity, make_event=self.make_event,
                    current_device=self.current_device, memory_info_provider=self.memory_info)

    def build(self, **overrides):
        from flagmemory.adapters.device_context import DeviceContextBridge
        kwargs = self.kwargs()
        kwargs.update(overrides)
        self.bridge = DeviceContextBridge(self.backend, **kwargs)
        return self.bridge

    def complete_all(self):
        for event in self.backend.events:
            event.completed = True


class ByteStorage(bytearray):
    """Weak-referenceable, directly consumable host storage."""


class ReadyAllocator:
    """Injected host-only allocator; not the old AscendTensorAllocator."""
    def __init__(self, bridge):
        self.bridge = bridge
        self.allocate_calls = 0
        self.release_calls = 0
        self.live_bytes = 0
        self.live = {}
        self.refs = []
        self.streams = []
        self.fail_release = False

    def allocate(self, nbytes, device_index=0, *, stream):
        self.bridge.validate_stream_device(stream, device_index)
        # Only a stream context: never a device lifecycle or global sync.
        with stream.context():
            ref = ByteStorage(nbytes)
            ref.ready = True
        self.allocate_calls += 1
        self.live[id(ref)] = nbytes
        self.live_bytes += nbytes
        self.refs.append(weakref.ref(ref))
        self.streams.append(weakref.ref(stream))
        return ref

    def release(self, base_ref):
        self.release_calls += 1
        if self.fail_release:
            raise RuntimeError("release outcome uncertain")
        self.live_bytes -= self.live.pop(id(base_ref))
