"""Explicitly selected, bounded Ascend tests; requires independently approved mapping."""
import os
import pytest
from ascend_test_adapter import AscendTestBackend, AscendTensorAllocator
from flagmemory.pool import BlockState, PoolForm, Purpose, ReusePolicy, TierSpec, create_pool

pytestmark = pytest.mark.realcard
SIZE = 8 << 20

@pytest.fixture
def setup():
    if os.environ.get('FLAGMEMORY_ASCEND_ENABLE') != 'approved-mapping-and-bounds':
        pytest.fail('explicit execution enable and validated device mapping required')
    try:
        device = int(os.environ['FLAGMEMORY_ASCEND_DEVICE'])
        limit = int(os.environ['FLAGMEMORY_ASCEND_BYTE_LIMIT'])
    except (KeyError, ValueError):
        pytest.fail('explicit device and cumulative byte limit required')
    if device < 0 or limit < 2 * SIZE or limit > 1 << 30:
        pytest.fail('invalid device or byte limit')
    # The operator must first confirm nonblocking query and all candidate APIs.
    if os.environ.get('FLAGMEMORY_ASCEND_QUERY_NONBLOCKING') != 'verified':
        pytest.fail('event query nonblocking behavior must be verified first')
    import torch
    import torch_npu  # noqa: F401 -- registers the runtime
    torch.npu.set_device(device)
    backend = AscendTestBackend(torch, device)
    allocator = AscendTensorAllocator(torch, device, limit)
    pool = create_pool(backend, device_index=device, form=PoolForm.TENSOR,
                       base_allocator=allocator, tier_spec=TierSpec(SIZE, SIZE, 2),
                       reuse=ReusePolicy(opportunistic=False), barrier_timeout_s=10)
    assert pool.tier_table.source == 'caller_config'
    yield torch, device, backend, allocator, pool
    report = pool.destroy_pool(timeout_s=10)
    assert report.completed and not report.pending and not report.quarantined and not report.leaked
    assert allocator.live_bytes == 0 and allocator.release_calls == allocator.allocate_calls

def wait_event(backend, event):
    import time
    deadline = time.monotonic() + 10
    while not backend.query_event(event):
        if time.monotonic() >= deadline: pytest.fail('readback event timeout')
        time.sleep(0.001)

def test_actual_tensor_and_cross_stream_dependency(setup):
    torch, device, backend, allocator, pool = setup
    owner = torch.npu.Stream(device=device); borrower = torch.npu.Stream(device=device)
    sid = pool.register_stream(owner); bid = pool.register_stream(borrower)
    view = pool.alloc_on_stream(SIZE, stream_id=sid, purpose=Purpose.ACTIVATION)
    tensor = pool.access_block(view.block_id)
    assert allocator.tensor(tensor) is tensor
    assert tensor.device.type == 'npu' and tensor.device.index == device
    assert tensor.dtype == torch.uint8 and tensor.numel() == SIZE
    with torch.npu.stream(owner):
        tensor[:4096].fill_(17)
        producer = torch.npu.Event(); producer.record(owner)
    with torch.npu.stream(borrower):
        borrower.wait_event(producer)
        tensor[:4096].add_(3)
    pool.declare_cross_stream_use(view.block_id, bid)
    ready = backend.create_event(device); backend.record_event(ready, borrower)
    wait_event(backend, ready)
    assert tensor[:4096].cpu().tolist() == [20] * 4096
    del tensor
    pool.free_block(view.block_id)
    assert not pool.drain_stream(timeout_s=10).recoverable_timeout

def test_completed_reuse_rejects_old_lease(setup):
    torch, device, backend, allocator, pool = setup
    stream = torch.npu.Stream(device=device); sid = pool.register_stream(stream)
    old = pool.alloc_on_stream(SIZE, stream_id=sid, purpose=Purpose.ACTIVATION)
    tensor = pool.access_block(old.block_id); identity = allocator.identity(tensor)
    with torch.npu.stream(stream): tensor[:4096].fill_(37)
    # Bounded externally by the process watchdog; the event query contract is prechecked.
    ready = backend.create_event(device); backend.record_event(ready, stream)
    wait_event(backend, ready)
    assert tensor[:4096].cpu().tolist() == [37] * 4096
    del tensor
    pool.free_block(old.block_id)
    assert not pool.drain_stream(timeout_s=10).recoverable_timeout
    new = pool.alloc_on_stream(SIZE, stream_id=sid, purpose=Purpose.ACTIVATION)
    assert new.block_id != old.block_id and new.storage_id == old.storage_id
    assert allocator.identity(pool.access_block(new.block_id)) == identity and allocator.allocate_calls == 1
    from flagmemory.pool import PoolError
    for operation in (pool.access_block, pool.block_view, pool.free_block):
        with pytest.raises(PoolError): operation(old.block_id)
    with pytest.raises(PoolError): pool.declare_cross_stream_use(old.block_id, sid)
    pool.free_block(new.block_id)
    assert not pool.drain_stream(timeout_s=10).recoverable_timeout

def test_pending_borrower_protection_when_observed(setup):
    torch, device, backend, allocator, pool = setup
    owner = torch.npu.Stream(device=device); borrower = torch.npu.Stream(device=device)
    sid = pool.register_stream(owner); bid = pool.register_stream(borrower)
    first = pool.alloc_on_stream(SIZE, stream_id=sid, purpose=Purpose.ACTIVATION)
    tensor = pool.access_block(first.block_id); identity = allocator.identity(tensor)
    with torch.npu.stream(borrower):
        for i in range(8): tensor.fill_(i)
    pool.declare_cross_stream_use(first.block_id, bid)
    del tensor
    pool.free_block(first.block_id)
    event = next(e for e in reversed(backend.events) if e.stream == borrower)
    observed = not backend.query_event(event)
    result = pool.drain_stream(timeout_s=0)
    print('borrower event query pending:', observed, 'drain:', result,
          'storage:', first.storage_id, 'identity:', identity, 'allocations:', allocator.allocate_calls)
    observed = (observed and result.recoverable_timeout and result.still_pending == 1
                and not result.quarantined_blocks and not backend.query_event(event))
    if observed:
        second = pool.alloc_on_stream(SIZE, stream_id=sid, purpose=Purpose.ACTIVATION)
        assert second.storage_id != first.storage_id
        assert allocator.identity(pool.access_block(second.block_id)) != identity and allocator.allocate_calls == 2
    assert not pool.drain_stream(timeout_s=10).recoverable_timeout
    if observed:
        reused = pool.alloc_on_stream(SIZE, stream_id=sid, purpose=Purpose.ACTIVATION)
        assert reused.storage_id == first.storage_id
        assert allocator.identity(pool.access_block(reused.block_id)) == identity
        assert allocator.allocate_calls == 2
        pool.free_block(reused.block_id)
        pool.free_block(second.block_id)
        assert not pool.drain_stream(timeout_s=10).recoverable_timeout
    if not observed:
        pytest.skip('pending borrower event was not observed; protection uncovered')
