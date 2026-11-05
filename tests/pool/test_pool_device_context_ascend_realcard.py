"""Explicit new public-device bridge lifecycle entry; collection is framework-free."""
import os
from pathlib import Path
import pytest
from device_context_ascend_case import (ADAPTER_KIND,load_manifest,load_runtime_types,
    make_case_profile,managed_pool,verify_runtime_origins,wait_event)
from flagmemory.pool import PoolError,PoolForm,Purpose,ReusePolicy,TierSpec,create_pool

pytestmark=pytest.mark.realcard
SIZE=8<<20
READ_BYTES=4096


@pytest.fixture(scope='session')
def bridge_session():
    if (os.environ.get('FLAGMEMORY_ASCEND_ENABLE') != 'approved-bridge-lifecycle'
        or os.environ.get('FLAGMEMORY_ASCEND_QUERY_NONBLOCKING') != 'verified'):
        pytest.fail('independently approved bridge lifecycle gate required')
    try:
        device=int(os.environ['FLAGMEMORY_ASCEND_DEVICE'])
        limit=int(os.environ['FLAGMEMORY_ASCEND_BYTE_LIMIT'])
        path=Path(os.environ['FLAGMEMORY_DEVICE_SOURCE'])
        digest=os.environ['FLAGMEMORY_DEVICE_MANIFEST_SHA256']
    except (KeyError,ValueError): pytest.fail('explicit bridge target, source and cumulative bounds required')
    if device != 0 or limit != 64<<20:
        pytest.fail('fixed single mapped target and 64 MiB cumulative budget required')
    manifest=load_manifest(path,digest)
    backend_type,stream_type,event_type=load_runtime_types(path.parent,manifest)
    import torch
    import torch_npu  # noqa: F401: only after all source and execution gates
    torch.npu.set_device(device)
    torch.npu.current_stream(device)  # caller owns stable initialized framework context
    backend=backend_type()
    profile=make_case_profile(torch,backend=backend,stream_type=stream_type,event_type=event_type,device_index=device,byte_limit=limit)
    verify_runtime_origins(path.parent)
    yield torch,device,backend,event_type,profile,path.parent
    verify_runtime_origins(path.parent)
    assert profile.allocator.live_bytes == 0
    assert profile.allocator.allocated_bytes <= limit


@pytest.fixture
def setup(bridge_session):
    torch,device,backend,event_type,profile,root=bridge_session
    verify_runtime_origins(root)
    pool=create_pool(profile.bridge,device_index=device,form=PoolForm.TENSOR,base_allocator=profile.allocator,
        tier_spec=TierSpec(SIZE,SIZE,2),reuse=ReusePolicy(opportunistic=False),barrier_timeout_s=10)
    assert pool.tier_table.source == 'caller_config'
    before=(profile.allocator.allocate_calls,profile.allocator.release_calls)
    print('bridge case:',ADAPTER_KIND,'target:',device,'units: bytes','context_ok: unverified',
        'device_global:',profile.bridge.device_memory_info(device),'allocator_before:',before)
    yield torch,device,backend,event_type,profile,pool
    assert profile.allocator.live_bytes == 0
    assert profile.allocator.allocate_calls-before[0] == profile.allocator.release_calls-before[1]
    print('runtime origins:',verify_runtime_origins(root))
    print('device_global_after:',profile.bridge.device_memory_info(device))
    print('allocator_after:',profile.allocator.allocate_calls,profile.allocator.release_calls,
        'cumulative_bytes:',profile.allocator.allocated_bytes,'live_bytes:',profile.allocator.live_bytes)


def _ready(profile,device,stream):
    event=profile.bridge.create_event(device); profile.bridge.record_event(event,stream)
    wait_event(profile.bridge,event)


def _drain(pool):
    result=pool.drain_stream(timeout_s=10)
    assert not result.recoverable_timeout and not result.still_pending and not result.quarantined_blocks


def _dependency(backend,event_type,owner,borrower):
    event=event_type(backend,backend.create_event(enable_timing=False))
    event.record(owner)
    event.wait(borrower)
    return event


def test_actual_tensor_and_cross_stream_dependency(setup):
    torch,device,backend,event_type,p,pool=setup
    with managed_pool(pool,p):
        owner=p.new_stream(); borrower=p.new_stream()
        sid=pool.register_stream(owner); bid=pool.register_stream(borrower)
        view=pool.alloc_on_stream(SIZE,stream_id=sid,purpose=Purpose.ACTIVATION)
        tensor=pool.access_block(view.block_id)
        assert tensor.device.type == 'npu' and tensor.device.index == device
        assert tensor.dtype == torch.uint8 and tensor.numel() == SIZE
        verdict=pool.verify_block_ownership(view.block_id)
        assert verdict.device_ok and verdict.ledger_ok and verdict.context_ok is None
        with owner.context(): tensor[:READ_BYTES].fill_(17)
        dependency=_dependency(backend,event_type,owner,borrower)
        with borrower.context(): tensor[:READ_BYTES].add_(3)
        pool.declare_cross_stream_use(view.block_id,bid)
        _ready(p,device,borrower)
        assert tensor[:READ_BYTES].cpu().tolist() == [20]*READ_BYTES
        del tensor,dependency
        pool.free_block(view.block_id); _drain(pool)


def test_completed_reuse_rejects_old_lease(setup):
    torch,device,backend,event_type,p,pool=setup
    with managed_pool(pool,p):
        stream=p.new_stream(); sid=pool.register_stream(stream)
        before=p.allocator.allocate_calls
        old=pool.alloc_on_stream(SIZE,stream_id=sid,purpose=Purpose.ACTIVATION)
        tensor=pool.access_block(old.block_id); pointer=tensor.untyped_storage().data_ptr()
        with stream.context(): tensor[:READ_BYTES].fill_(37)
        _ready(p,device,stream)
        assert tensor[:READ_BYTES].cpu().tolist() == [37]*READ_BYTES
        del tensor
        pool.free_block(old.block_id); _drain(pool)
        new=pool.alloc_on_stream(SIZE,stream_id=sid,purpose=Purpose.ACTIVATION)
        assert new.block_id != old.block_id and new.storage_id == old.storage_id
        assert pool.access_block(new.block_id).untyped_storage().data_ptr() == pointer
        assert p.allocator.allocate_calls-before == 1
        for op in (pool.access_block,pool.block_view,pool.free_block):
            with pytest.raises(PoolError): op(old.block_id)
        with pytest.raises(PoolError): pool.declare_cross_stream_use(old.block_id,sid)
        pool.free_block(new.block_id); _drain(pool)


def test_pending_borrower_protection_when_observed(setup):
    torch,device,backend,event_type,p,pool=setup
    with managed_pool(pool,p):
        owner=p.new_stream(); borrower=p.new_stream()
        sid=pool.register_stream(owner); bid=pool.register_stream(borrower)
        before=p.allocator.allocate_calls
        first=pool.alloc_on_stream(SIZE,stream_id=sid,purpose=Purpose.ACTIVATION)
        tensor=pool.access_block(first.block_id); pointer=tensor.untyped_storage().data_ptr()
        dependency=_dependency(backend,event_type,owner,borrower)
        with borrower.context():
            for value in range(8): tensor.fill_(value)
        pool.declare_cross_stream_use(first.block_id,bid)
        del tensor,dependency
        pool.free_block(first.block_id)
        event=next(e for e,s in reversed(p.bridge.records) if s is borrower)
        observed=not p.bridge.query_event(event)
        result=pool.drain_stream(timeout_s=0)
        observed=observed and result.still_pending == 1 and not result.quarantined_blocks and not p.bridge.query_event(event)
        print('bridge pending observed:',observed,'drain:',result,'storage:',first.storage_id,'pointer:',pointer)
        second=None
        if observed:
            second=pool.alloc_on_stream(SIZE,stream_id=sid,purpose=Purpose.ACTIVATION)
            assert second.storage_id != first.storage_id
            assert pool.access_block(second.block_id).untyped_storage().data_ptr() != pointer
            assert p.allocator.allocate_calls-before == 2
        _drain(pool)
        if observed:
            reused=pool.alloc_on_stream(SIZE,stream_id=sid,purpose=Purpose.ACTIVATION)
            assert reused.storage_id == first.storage_id
            assert pool.access_block(reused.block_id).untyped_storage().data_ptr() == pointer
            assert p.allocator.allocate_calls-before == 2
            pool.free_block(reused.block_id); pool.free_block(second.block_id); _drain(pool)
        else:
            pytest.skip('pending borrower not observed; bridge protection uncovered')
