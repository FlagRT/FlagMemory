import pytest
from types import SimpleNamespace
from contextlib import contextmanager
from device_context_fakes import DeviceBackend, UnifiedStream, UnifiedEvent
from flagmemory.pool import InvalidArgument

class Raw:
    def __init__(self, index=2): self.device = SimpleNamespace(type='npu', index=index)
class Backend(DeviceBackend):
    def __init__(self):
        super().__init__(); self.default_native = Raw(); self.active = False; self.created = 0
    def create_stream(self): self.created += 1; return Raw()
    @contextmanager
    def stream_context(self, native):
        self.active = True
        try: yield
        finally: self.active = False
class Tensor(bytearray):
    def __init__(self, n): super().__init__(n); self.device = SimpleNamespace(type='npu', index=2)
    def numel(self): return len(self)
    def element_size(self): return 1
class NPU:
    Stream = Raw
    def __init__(self): self.current = 2; self.initialized = True; self.info = (257,1025); self.calls = 0
    def is_initialized(self): return self.initialized
    def current_device(self):
        self.calls += 1
        if isinstance(self.current, Exception): raise self.current
        return self.current
    def mem_get_info(self, index):
        if isinstance(self.info, Exception): raise self.info
        return self.info
class Torch:
    uint8 = object()
    class OutOfMemoryError(RuntimeError): pass
    def __init__(self, backend): self.npu = NPU(); self.backend = backend; self.error = None; self.allocs = 0
    def empty(self, n, *, dtype, device):
        assert self.backend.active and dtype is self.uint8 and device == 'npu:2'
        if self.error: raise self.error
        self.allocs += 1; return Tensor(n)
def setup():
    from flagmemory.adapters.ascend import make_ascend_profile
    b = Backend(); t = Torch(b)
    def make(): return make_ascend_profile(torch_module=t, backend=b, stream_type=UnifiedStream, event_type=UnifiedEvent, device_index=2, byte_limit=128)
    return b, t, make

def test_profile_identity_allocator_and_bytes():
    b,t,make = setup(); p = make(); s = p.bridge.current_stream(2)
    assert p.bridge.device_memory_info(2).free == 257
    assert p.bridge.streams_equal(s, UnifiedStream(b, b.default_native))
    with pytest.raises(InvalidArgument): p.bridge.validate_stream_device(UnifiedStream(b, Raw()), 2)
    with pytest.raises(InvalidArgument): p.bridge.validate_stream_device(UnifiedStream(Backend(), b.default_native), 2)
    new = p.new_stream(); p.bridge.validate_stream_device(new, 2)
    x = p.allocator.allocate(20, 2, stream=new); x[0] = 42
    assert x[0] == 42 and not b.active
    p.allocator.release(x)
    with pytest.raises(InvalidArgument): p.allocator.release(x)
    with pytest.raises(MemoryError): p.allocator.allocate(109, 2, stream=new)
    assert t.allocs == 1

@pytest.mark.parametrize('bad', [(True, 3), (1.0, 3), (-1, 3), (4, 3), (1,), None])
def test_bad_bytes_rejected(bad):
    b,t,make = setup(); t.npu.info = bad
    with pytest.raises((InvalidArgument, ValueError)): make()

def test_uninitialized_never_queries_device():
    b,t,make = setup(); t.npu.initialized = False
    with pytest.raises(InvalidArgument): make()
    assert t.npu.calls == 0 and b.created == 0

def test_exceptions_propagate_and_new_stream_checks_before_create():
    b,t,make = setup(); p = make(); error = RuntimeError('current failed'); t.npu.current = error
    with pytest.raises(RuntimeError) as exc: p.new_stream()
    assert exc.value is error and b.created == 0
    t.npu.current = 2; error = RuntimeError('bytes failed'); t.npu.info = error
    with pytest.raises(RuntimeError) as exc: p.bridge.device_memory_info(2)
    assert exc.value is error

def test_oom_only_translates_public_class():
    b,t,make = setup(); p = make(); s = p.bridge.current_stream(2)
    t.error = t.OutOfMemoryError('oom')
    with pytest.raises(MemoryError) as exc: p.allocator.allocate(1, 2, stream=s)
    assert exc.value.__cause__ is t.error
    t.error = RuntimeError('other')
    with pytest.raises(RuntimeError) as exc: p.allocator.allocate(1, 2, stream=s)
    assert exc.value is t.error

def test_consumer_writes_and_incomplete_report_is_visible():
    from examples.device_context_tensor_pool import consume_once
    from device_context_fakes import Bundle, ReadyAllocator
    from flagmemory.pool import TierSpec
    b = Bundle(device=0); bridge = b.build(); allocator = ReadyAllocator(bridge)
    seen = []
    def consume(data): data[0] = 91; seen.append(data[0])
    report = consume_once(bridge, allocator, tier_spec=TierSpec(64,64,1), nbytes=10, consume=consume)
    assert seen == [91] and not report.completed and report.pending

def test_consumer_error_keeps_original_and_cleanup_report():
    from examples.device_context_tensor_pool import consume_once
    from device_context_fakes import Bundle, ReadyAllocator
    from flagmemory.pool import TierSpec
    b = Bundle(device=0); bridge = b.build(); allocator = ReadyAllocator(bridge)
    error = RuntimeError('consumer failed')
    def consume(data): raise error
    with pytest.raises(RuntimeError) as exc:
        consume_once(bridge, allocator, tier_spec=TierSpec(64,64,1), nbytes=10, consume=consume)
    assert exc.value is error and not exc.value.destroy_report.completed

def test_event_factory_checks_device_before_and_after_and_uses_timing_false():
    b,t,make = setup(); p = make()
    original = b.create_event
    def drift(*, enable_timing=False):
        e = original(enable_timing=enable_timing); t.npu.current = 0; return e
    b.create_event = drift
    with pytest.raises(InvalidArgument): p.bridge.create_event(2)
    assert ('create_event', False) in b.audit

def test_wrong_native_device_and_factory_drift_are_rejected():
    b,t,make = setup(); b.default_native = Raw(0)
    with pytest.raises(InvalidArgument): make()
    b,t,make = setup(); p = make()
    def drift(): t.npu.current = 0; return Raw()
    b.create_stream = drift
    with pytest.raises(InvalidArgument): p.new_stream()

def test_same_native_from_controlled_factory_reuses_key():
    b,t,make = setup(); p = make(); b.create_stream = lambda: b.default_native
    assert p.bridge.streams_equal(p.bridge.current_stream(2), p.new_stream())

def test_successful_consumer_close_after_completed_event():
    from examples.device_context_tensor_pool import consume_once
    from device_context_fakes import Bundle, ReadyAllocator
    from flagmemory.pool import TierSpec
    b = Bundle(device=0); bridge = b.build(); allocator = ReadyAllocator(bridge)
    original = b.backend.create_event
    def completed(*, enable_timing=False):
        event = original(enable_timing=enable_timing); event.completed = True; return event
    b.backend.create_event = completed
    report = consume_once(bridge, allocator, tier_spec=TierSpec(64,64,1), nbytes=10, consume=lambda x: x.__setitem__(0, 1))
    assert report.completed and allocator.release_calls == 1

def test_consumer_failure_still_attempts_destroy_after_free_failure(monkeypatch):
    import examples.device_context_tensor_pool as example
    from types import SimpleNamespace
    original = RuntimeError('consumer'); report = object(); calls = []
    class Pool:
        def alloc_on_stream(self, *a, **kw): return SimpleNamespace(block_id=1)
        def access_block(self, *a): return bytearray(1)
        def free_block(self, *a): calls.append('free'); raise RuntimeError('free')
        def destroy_pool(self, **kw): calls.append('destroy'); return report
    monkeypatch.setattr(example, 'create_pool', lambda *a, **kw: Pool())
    def fail(_): raise original
    with pytest.raises(RuntimeError) as exc: example.consume_once(None, None, tier_spec=None, nbytes=1, consume=fail)
    assert exc.value is original and exc.value.destroy_report is report
    assert calls == ['free', 'destroy']

def test_consumer_nonzero_device_and_binding_mismatch():
    from examples.device_context_tensor_pool import consume_once
    from device_context_fakes import Bundle, ReadyAllocator
    from flagmemory.pool import TierSpec
    b = Bundle(device=2); bridge = b.build(); allocator = ReadyAllocator(bridge)
    with pytest.raises(InvalidArgument):
        consume_once(bridge, allocator, tier_spec=TierSpec(64,64,1), nbytes=10, consume=lambda x: None, device_index=0)
    assert allocator.allocate_calls == 0
    report = consume_once(bridge, allocator, tier_spec=TierSpec(64,64,1), nbytes=10, consume=lambda x: x.__setitem__(0, 9), device_index=2)
    assert not report.completed and report.pending and allocator.allocate_calls == 1

def test_python_memory_error_propagates_unchanged():
    b,t,make = setup(); p = make(); s = p.bridge.current_stream(2)
    t.OutOfMemoryError = MemoryError
    error = MemoryError('host allocation'); t.error = error
    with pytest.raises(MemoryError) as exc: p.allocator.allocate(1, 2, stream=s)
    assert exc.value is error

def test_byte_provider_value_error_and_bridge_cause(monkeypatch):
    import flagmemory.adapters.ascend as module
    b,t,make = setup(); original_bridge = module.DeviceContextBridge; captured = {}
    def capture(*args, **kwargs):
        captured['provider'] = kwargs['memory_info_provider']
        return original_bridge(*args, **kwargs)
    monkeypatch.setattr(module, 'DeviceContextBridge', capture)
    p = make(); error = ValueError('native byte query'); t.npu.info = error
    with pytest.raises(ValueError) as direct: captured['provider'](2)
    assert direct.value is error
    with pytest.raises(InvalidArgument) as bridged: p.bridge.device_memory_info(2)
    assert bridged.value.__cause__ is error and b.forbidden == []
