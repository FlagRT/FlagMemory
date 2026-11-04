from contextlib import contextmanager
from types import SimpleNamespace
import pytest
from flagmemory.caps.backend import EventNotRecorded
from ascend_test_adapter import AscendTestBackend, AscendTensorAllocator

class Stream:
    def __init__(self, index): self.device = SimpleNamespace(type='npu', index=index)

class Event:
    def __init__(self): self.done = False; self.fail = False
    def record(self, stream):
        if self.fail: raise RuntimeError('record failed')
    def query(self): return self.done

class Runtime:
    uint8 = object()
    def __init__(self):
        self.current = None; self.calls = 0; self.fail = False; self.npu = self
    @contextmanager
    def device(self, index): yield
    @contextmanager
    def stream(self, stream):
        old = self.current; self.current = stream
        try: yield
        finally: self.current = old
    def empty(self, n, *, dtype, device):
        self.calls += 1
        if self.fail: raise RuntimeError('allocation failed')
        return SimpleNamespace(stream=self.current, size=n, device=device)
    def Event(self): return Event()

def test_events_are_independent_and_record_failure_is_unrecorded():
    rt = Runtime(); backend = AscendTestBackend(rt, 2); a = Stream(2); b = Stream(2)
    first = backend.create_event(2); second = backend.create_event(2)
    with pytest.raises(EventNotRecorded): backend.query_event(first)
    second.raw.fail = True
    with pytest.raises(RuntimeError): backend.record_event(second, b)
    with pytest.raises(EventNotRecorded): backend.query_event(second)
    second.raw.fail = False; backend.record_event(first, a); backend.record_event(second, b)
    first.raw.done = True
    assert backend.query_event(first) and not backend.query_event(second)

def test_allocation_context_quota_and_release():
    rt = Runtime(); a = AscendTensorAllocator(rt, 2, 16); stream = Stream(2)
    block = a.allocate(8, 2, stream=stream)
    assert block.stream is stream and rt.current is None and a.tensor(block) is block
    assert a.identity(block) == id(block) and a.live_bytes == 8 and a.allocate_calls == 1
    with pytest.raises(MemoryError): a.allocate(9, 2, stream=stream)
    assert rt.calls == 1
    a.release(block)
    assert a.live_bytes == 0 and a.release_calls == 1
    with pytest.raises(RuntimeError): a.release(block)
    rt.fail = True
    with pytest.raises(RuntimeError): a.allocate(8, 2, stream=stream)
    assert a.allocate_calls == 1 and rt.current is None

def test_reject_unknown_or_wrong_stream_before_allocation():
    rt = Runtime(); a = AscendTensorAllocator(rt, 2, 16)
    for stream in (Stream(1), object()):
        with pytest.raises(ValueError): a.allocate(8, 2, stream=stream)
    assert rt.calls == 0


def test_first_consumption_rejects_wrong_allocation_context():
    rt = Runtime(); target = Stream(2); other = Stream(2)
    alloc = AscendTensorAllocator(rt, 2, 16)
    tensor = alloc.allocate(8, 2, stream=target)
    def consume(value, stream):
        if value.stream is not stream: raise RuntimeError('not ready on target stream')
        return value.size
    assert consume(tensor, target) == 8
    alloc.release(tensor)
    class WrongAllocator(AscendTensorAllocator):
        def allocate(self, nbytes, device_index=0, *, stream):
            return super().allocate(nbytes, device_index, stream=other)
    wrong = WrongAllocator(rt, 2, 16)
    tensor = wrong.allocate(8, 2, stream=target)
    with pytest.raises(RuntimeError, match='not ready'):
        consume(tensor, target)
    wrong.release(tensor)


def test_release_failure_does_not_change_accounting():
    class FailingPop(dict):
        def pop(self, key): raise RuntimeError('release failed')
    rt = Runtime(); alloc = AscendTensorAllocator(rt, 2, 16)
    tensor = alloc.allocate(8, 2, stream=Stream(2))
    alloc.live = FailingPop(alloc.live)
    with pytest.raises(RuntimeError, match='release failed'):
        alloc.release(tensor)
    assert alloc.live_bytes == 8 and alloc.release_calls == 0
    assert alloc.tensor(tensor) is tensor
    alloc.live = dict(alloc.live)
    alloc.release(tensor)
    assert alloc.live_bytes == 0 and alloc.release_calls == 1
