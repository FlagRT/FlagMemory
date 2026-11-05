"""Host exercises of the actual Ascend acceptance flow and adapter failure paths."""
from contextlib import contextmanager
from types import SimpleNamespace
import gc
import sys
import weakref
import pytest
import test_pool_ascend_realcard as real
from ascend_test_adapter import AscendTestBackend, AscendTensorAllocator
from flagmemory.pool import PoolError, PoolForm, Purpose, TierSpec, create_pool
import flagmemory.pool.pool as pool_module

class Stream:
    def __init__(self, runtime, device):
        self.runtime = runtime; self.device = SimpleNamespace(type='npu', index=device); self.ready = set()
    def wait_event(self, event):
        if event.stream is None: raise RuntimeError('wait without record')
        self.ready.add(event.stream)

class Event:
    def __init__(self, rt): self.rt = rt; self.stream = None; self.done = False; self.queries = 0
    def record(self, stream):
        self.rt.records += 1
        if self.rt.fail_record == self.rt.records: raise RuntimeError('record failed')
        self.stream = stream
    def query(self):
        if self.rt.fail_query: raise RuntimeError('query failed')
        self.queries += 1
        return self.done or self.rt.complete_on_query or (self.rt.auto_finish and self.queries >= 4)

class Tensor:
    def __init__(self, rt, n, device):
        self.rt=rt; self.size=n; self.device=SimpleNamespace(type='npu', index=device)
        self.owner=rt.current; self.data=bytearray(n)
    def fill_(self, value):
        stream=self.rt.current
        if self.rt.current_device != self.device.index or (stream is not self.owner and self.owner not in stream.ready):
            raise RuntimeError('resource not ready on this stream/device')
        self.data[:] = bytes([value]) * self.size
        return self

class Runtime:
    uint8 = object()
    def __init__(self):
        self.npu=self; self.current_device=9; self.current=Stream(self, 9)
        self.calls=0; self.fail_empty=False; self.fail_record=None; self.records=0
        self.fail_query=False; self.complete_on_query=False; self.auto_finish=True
        self.default=Stream(self,2)
    @contextmanager
    def device(self, index):
        old=self.current_device; self.current_device=index
        try: yield
        finally: self.current_device=old
    @contextmanager
    def stream(self, stream):
        old=self.current; self.current=stream
        try: yield
        finally: self.current=old
    def Stream(self, device): return Stream(self, device)
    def Event(self): return Event(self)
    def current_stream(self, index): return self.default
    def mem_get_info(self,index): return 1024,2048
    def set_device(self,index): self.current_device=index; self.current=self.default
    def empty(self, n, *, dtype, device):
        self.calls += 1
        if self.fail_empty: raise RuntimeError('allocation failed')
        index=int(device.split(':')[1])
        if self.current_device != index or self.current.device.index != index: raise RuntimeError('wrong allocation context')
        return Tensor(self,n,index)

def make(rt, limit=64):
    rt.current_device=2; rt.current=rt.default
    backend=AscendTestBackend(rt,2); allocator=AscendTensorAllocator(rt,2,limit)
    pool=create_pool(backend,device_index=2,form=PoolForm.TENSOR,base_allocator=allocator,
                     tier_spec=TierSpec(16,16,2),reuse=real.ReusePolicy(opportunistic=False))
    return rt,2,backend,allocator,pool

def test_pending_acceptance_flow_and_race(monkeypatch):
    monkeypatch.setattr(real,'SIZE',16)
    rt=Runtime(); args=make(rt)
    real.test_pending_borrower_protection_when_observed(args)
    assert args[3].allocate_calls == 2
    assert args[4].destroy_pool().completed
    rt=Runtime(); rt.complete_on_query=True; args=make(rt)
    with pytest.raises(pytest.skip.Exception, match='uncovered'):
        real.test_pending_borrower_protection_when_observed(args)
    assert args[3].allocate_calls == 1
    assert args[4].destroy_pool().completed

def test_first_borrower_access_requires_wait(monkeypatch):
    monkeypatch.setattr(real,'SIZE',16)
    monkeypatch.setattr(Stream, 'wait_event', lambda self, event: None)
    rt=Runtime(); args=make(rt)
    with pytest.raises(RuntimeError,match='not ready'):
        real.test_pending_borrower_protection_when_observed(args)

def test_context_restored_and_wrong_variants_fail():
    rt=Runtime(); prior=rt.current; target=rt.Stream(2)
    allocator=AscendTensorAllocator(rt,2,64)
    tensor=allocator.allocate(16,2,stream=target)
    assert rt.current_device == 9 and rt.current is prior
    with rt.device(2), rt.stream(target): tensor.fill_(7)
    assert tensor.data == bytes([7])*16
    rt.fail_empty=True
    with pytest.raises(RuntimeError,match='allocation failed'): allocator.allocate(16,2,stream=target)
    assert rt.current_device == 9 and rt.current is prior
    rt.fail_empty=False
    original_device = rt.device
    @contextmanager
    def broken_device(index): yield
    rt.device = broken_device
    with pytest.raises(RuntimeError,match='wrong allocation context'):
        allocator.allocate(16,2,stream=target)
    assert rt.current_device == 9 and rt.current is prior
    rt.device = original_device
    with rt.device(2), rt.stream(rt.Stream(2)), pytest.raises(RuntimeError,match='not ready'):
        tensor.fill_(1)
    allocator.release(tensor)

def test_budget_shared_across_allocators_before_runtime_call():
    from ascend_test_adapter import AllocationBudget
    rt=Runtime(); budget=AllocationBudget(32); stream=rt.Stream(2)
    for _ in range(2):
        a=AscendTensorAllocator(rt,2,budget); block=a.allocate(16,2,stream=stream); a.release(block)
    a=AscendTensorAllocator(rt,2,budget)
    with pytest.raises(MemoryError): a.allocate(16,2,stream=stream)
    assert rt.calls == 2 and budget.used == 32 and a.allocate_calls == 0
    fresh=AllocationBudget(16); a=AscendTensorAllocator(rt,2,fresh); rt.fail_empty=True
    with pytest.raises(RuntimeError, match='allocation failed'): a.allocate(16,2,stream=stream)
    assert fresh.used == 0 and a.allocate_calls == 0

@pytest.mark.parametrize('fault',['record','query','release'])
def test_real_pool_fault_retains_resources_without_releasing_twice(fault):
    rt=Runtime(); rt.auto_finish=False; args=make(rt); _,_,backend,allocator,pool=args
    owner=rt.Stream(2); borrower=rt.Stream(2)
    sid=pool.register_stream(owner); bid=pool.register_stream(borrower)
    view=pool.alloc_on_stream(16,stream_id=sid,purpose=Purpose.ACTIVATION)
    tensor=pool.access_block(view.block_id); ref=weakref.ref(tensor)
    pool.declare_cross_stream_use(view.block_id,bid)
    if fault == 'record':
        rt.fail_record=2
        with pytest.raises(PoolError): pool.free_block(view.block_id)
        assert len(backend.events)==2 and sum(e.recorded for e in backend.events)==1
    else:
        pool.free_block(view.block_id)
        if fault == 'query':
            rt.fail_query=True
            assert pool.drain_stream(timeout_s=0).quarantined_blocks
        else:
            rt.complete_on_query=True; pool.drain_stream(timeout_s=0)
            class FailedPop(dict):
                def pop(self,key): raise RuntimeError('release failed')
            allocator.live=FailedPop(allocator.live)
            with pytest.raises(PoolError): pool.destroy_pool()
    report=pool.destroy_pool(wait_for_drain=False)
    assert report.quarantined and not report.completed
    custody=report.custody_id
    event_refs = [weakref.ref(event) for event in backend.events]
    stream_ref = weakref.ref(borrower)
    allocator_ref = weakref.ref(allocator)
    del tensor, pool, args, backend, allocator, owner, borrower, rt
    gc.collect()
    assert ref() is not None and stream_ref() is not None
    assert all(event() is not None for event in event_refs)
    pool_module.recover_custodied_pool(custody).destroy_pool(wait_for_drain=False)
    assert allocator_ref().release_calls == 0 and allocator_ref().live_bytes == 16


def test_actual_fixture_shares_budget_across_pools(monkeypatch):
    from ascend_test_adapter import AllocationBudget
    monkeypatch.setattr(real, 'SIZE', 16)
    monkeypatch.setenv('FLAGMEMORY_ASCEND_ENABLE','approved-mapping-and-bounds')
    monkeypatch.setenv('FLAGMEMORY_ASCEND_DEVICE','2')
    monkeypatch.setenv('FLAGMEMORY_ASCEND_BYTE_LIMIT','32')
    monkeypatch.setenv('FLAGMEMORY_ASCEND_QUERY_NONBLOCKING','verified')
    rt=Runtime(); rt.complete_on_query=True
    monkeypatch.setitem(sys.modules,'torch',rt)
    monkeypatch.setitem(sys.modules,'torch_npu',SimpleNamespace())
    budget=AllocationBudget(32)
    for index in range(3):
        fixture=real.setup.__wrapped__(budget)
        _,_,_,allocator,pool=next(fixture)
        if index < 2:
            block=pool.alloc_on_stream(16,purpose=Purpose.ACTIVATION)
            pool.free_block(block.block_id)
            pool.drain_stream(timeout_s=1)
        else:
            with pytest.raises(PoolError): pool.alloc_on_stream(16,purpose=Purpose.ACTIVATION)
            assert allocator.allocate_calls == 0
        with pytest.raises(StopIteration): next(fixture)
    assert rt.calls == 2 and budget.used == 32
