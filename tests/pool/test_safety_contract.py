"""Safety regressions: real pool state, controlled asynchronous completion and lifetimes."""
import gc
import weakref

import pytest

from flagmemory.caps.capabilities import L0_CAPABILITIES
from flagmemory.caps.fake import FakeBackend
from flagmemory.pool import BlockState, InvalidArgument, LeakDetected, OwnershipViolation, PoolError, PoolForm, Purpose, TierSpec, create_pool
import flagmemory.pool.pool as pool_module
import flagmemory.pool.barrier as barrier_module


class Stream:
    pass


class Event:
    def __init__(self):
        self.recorded = False
        self.completed = False


class AsyncBackend(FakeBackend):
    def __init__(self):
        super().__init__(name="controlled", declared=frozenset(L0_CAPABILITIES))
        self.devices = weakref.WeakKeyDictionary()
        self.events = []
        self.event_devices = []
        self.fault = False
        self.fail_record_at = None

    def stream(self, device):
        stream = Stream()
        self.devices[stream] = device
        return stream

    def current_stream(self, device_index=0):
        return self.stream(device_index)

    def validate_stream_device(self, stream, device_index):
        if self.devices.get(stream) != device_index:
            raise ValueError("wrong stream device")

    def streams_equal(self, left, right):
        return left is right

    def create_event(self, device_index=0):
        self.event_devices.append(device_index)
        event = Event()
        self.events.append(weakref.ref(event))
        return event

    def record_event(self, event, stream):
        if self.fail_record_at == len(self.events):
            raise RuntimeError("record failed")
        event.recorded = True

    def query_event(self, event):
        if self.fault:
            raise RuntimeError("query failed")
        assert event.recorded
        return event.completed

    def complete(self):
        for ref in self.events:
            if ref() is not None:
                ref().completed = True


class Resource:
    def __init__(self, deaths):
        self.data = bytearray(16)
        self.deaths = deaths

    def __del__(self):
        self.deaths.append("destroyed")


class Allocator:
    def __init__(self):
        self.refs = []
        self.streams = []
        self.deaths = []
        self.releases = 0
        self.fail_release = False

    def allocate(self, nbytes, device_index=0, *, stream=None):
        resource = Resource(self.deaths)
        self.refs.append(weakref.ref(resource))
        self.streams.append(None if stream is None else weakref.ref(stream))
        return resource

    def release(self, resource):
        self.releases += 1
        if self.fail_release:
            raise RuntimeError("release side effect then failure")


def make_pool(device=0):
    backend, allocator = AsyncBackend(), Allocator()
    pool = create_pool(backend, device_index=device, form=PoolForm.TENSOR,
                       base_allocator=allocator, tier_spec=TierSpec(base_bytes=16, step_bytes=16, tier_count=4))
    return pool, backend, allocator


def allocate(pool):
    return pool.alloc_on_stream(16, purpose=Purpose.ACTIVATION)


def finish(pool, backend, *leases):
    for lease in leases:
        pool.free_block(lease.block_id)
    backend.complete()
    pool.destroy_pool()


def test_foreign_lease_does_not_mutate_local_live_block():
    a, ab, _ = make_pool()
    b, bb, _ = make_pool()
    x, y = allocate(a), allocate(b)
    with pytest.raises(OwnershipViolation):
        b.free_block(x.block_id)
    assert b.block_view(y.block_id).state is BlockState.IN_USE
    finish(a, ab, x)
    finish(b, bb, y)


def test_reused_storage_has_new_lease_and_rejects_every_old_entry():
    pool, backend, allocator = make_pool()
    old = allocate(pool)
    pool.free_block(old.block_id)
    backend.complete()
    pool.drain_stream(timeout_s=0)
    new = allocate(pool)
    assert new.block_id != old.block_id
    assert new.storage_id == old.storage_id
    assert len(allocator.refs) == 1
    for action in (pool.free_block, pool.block_view, pool.verify_block_ownership, pool.access_block):
        with pytest.raises(OwnershipViolation):
            action(old.block_id)
    with pytest.raises(OwnershipViolation):
        pool.declare_cross_stream_use(old.block_id, 0)
    assert pool.block_view(new.block_id).state is BlockState.IN_USE
    finish(pool, backend, new)


def test_access_reads_actual_storage_and_is_revoked_on_return():
    pool, backend, allocator = make_pool()
    lease = allocate(pool)
    verdict = pool.verify_block_ownership(lease.block_id)
    assert verdict.context_ok is None
    assert "no_context_observe" in {item.key for item in verdict.degradations}
    ref = pool.access_block(lease.block_id)
    ref.data[:] = b"0123456789abcdef"
    assert allocator.refs[0]().data == b"0123456789abcdef"
    pool.free_block(lease.block_id)
    with pytest.raises(OwnershipViolation):
        pool.access_block(lease.block_id)
    del ref
    backend.complete()
    pool.destroy_pool()


def test_target_stream_and_explicit_nonzero_event_device():
    pool, backend, allocator = make_pool(2)
    stream = backend.stream(2)
    sid = pool.register_stream(stream)
    lease = pool.alloc_on_stream(16, stream_id=sid, purpose=Purpose.ACTIVATION)
    assert allocator.streams[-1] is not None and allocator.streams[-1]() is stream
    pool.free_block(lease.block_id)
    assert backend.event_devices == [2]
    backend.complete()
    pool.destroy_pool()


def test_reject_wrong_device_before_registering():
    pool, backend, _ = make_pool(2)
    with pytest.raises(InvalidArgument):
        pool.register_stream(backend.stream(0))
    pool.destroy_pool()


def test_zero_polls_never_quarantine_healthy_pending_work():
    pool, backend, _ = make_pool()
    lease = allocate(pool)
    pool.free_block(lease.block_id)
    for _ in range(8):
        result = pool.drain_stream(timeout_s=0)
        assert result.still_pending == 1
        assert not result.recoverable_timeout
        assert not result.quarantined_blocks
    backend.complete()
    pool.destroy_pool()


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf"), -float("inf"), True])
@pytest.mark.parametrize("entry", ["drain", "destroy", "create"])
def test_invalid_wait_rejected_even_when_empty(timeout, entry):
    pool, _, _ = make_pool()
    with pytest.raises(InvalidArgument):
        if entry == "create":
            create_pool(AsyncBackend(), form=PoolForm.TENSOR, base_allocator=Allocator(),
                        tier_spec=TierSpec(base_bytes=16, step_bytes=16, tier_count=4), barrier_timeout_s=timeout)
        elif entry == "destroy":
            pool.destroy_pool(wait_for_drain=False, timeout_s=timeout)
        else:
            pool.drain_stream(timeout_s=timeout)
    pool.destroy_pool()


def test_all_blocks_share_one_deadline(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(barrier_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(barrier_module.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    pool, backend, _ = make_pool()
    leases = [allocate(pool) for _ in range(4)]
    for lease in leases:
        pool.free_block(lease.block_id)
    result = pool.drain_stream(timeout_s=0.0025)
    assert now[0] <= 0.0025 + 1e-12
    assert result.still_pending == 4
    backend.complete()
    pool.destroy_pool()


@pytest.mark.parametrize("wait", [False, True])
def test_close_pending_survives_drop_and_recovers(wait):
    pool, backend, allocator = make_pool()
    lease = allocate(pool)
    stream_ref = weakref.ref(pool._streams.get(0))
    pool.free_block(lease.block_id)
    report = pool.destroy_pool(wait_for_drain=wait, timeout_s=0)
    assert not pool.destroyed
    assert len(report.pending) == 1 and not report.completed
    custody_id = report.custody_id
    del pool
    gc.collect()
    assert allocator.refs[0]() is not None and stream_ref() is not None
    assert custody_id in {item.custody_id for item in pool_module.list_custodied_pools()}
    recovered = pool_module.recover_custodied_pool(custody_id)
    backend.complete()
    assert recovered.destroy_pool().completed
    assert allocator.releases == 1
    assert custody_id not in {item.custody_id for item in pool_module.list_custodied_pools()}
    gc.collect()
    assert allocator.refs[0]() is None and stream_ref() is None


def test_close_with_live_lease_allows_later_return():
    pool, backend, allocator = make_pool()
    lease = allocate(pool)
    with pytest.raises(LeakDetected) as exc:
        pool.destroy_pool()
    assert not pool.destroyed and exc.value.report.leaked
    for action in (lambda: allocate(pool), lambda: pool.access_block(lease.block_id),
                   lambda: pool.register_stream(backend.stream(0)),
                   lambda: pool.declare_cross_stream_use(lease.block_id, 0)):
        with pytest.raises(PoolError):
            action()
    finish(pool, backend, lease)
    assert allocator.releases == 1


@pytest.mark.parametrize("fault", ["query", "record", "release"])
def test_quarantine_retains_resources_after_pool_drop_without_double_release(fault):
    pool, backend, allocator = make_pool()
    lease = allocate(pool)
    stream_ref = weakref.ref(pool._streams.get(0))
    if fault == "record":
        backend.fail_record_at = 1
        with pytest.raises(PoolError):
            pool.free_block(lease.block_id)
    else:
        pool.free_block(lease.block_id)
        if fault == "query":
            backend.fault = True
            pool.drain_stream(timeout_s=0)
        else:
            backend.complete()
            pool.drain_stream(timeout_s=0)
            allocator.fail_release = True
            with pytest.raises(PoolError):
                pool.destroy_pool()
    report = pool.destroy_pool(wait_for_drain=False)
    assert not report.completed and len(report.quarantined) == 1
    custody_id = report.custody_id
    del pool
    gc.collect()
    assert allocator.refs[0]() is not None and stream_ref() is not None
    assert backend.events[0]() is not None
    recovered = pool_module.recover_custodied_pool(custody_id)
    recovered.destroy_pool(wait_for_drain=False)
    assert allocator.releases == (1 if fault == "release" else 0)


def test_nonzero_device_event_does_not_inspect_stream_attributes():
    pool, backend, _ = make_pool(2)
    lease = allocate(pool)
    pool.free_block(lease.block_id)
    assert backend.event_devices == [2]
    backend.complete()
    pool.destroy_pool()


def test_partial_ticket_failure_retains_all_created_events_without_close():
    pool, backend, allocator = make_pool()
    lease = allocate(pool)
    stream = backend.stream(0)
    stream_ref = weakref.ref(stream)
    sid = pool.register_stream(stream)
    pool.declare_cross_stream_use(lease.block_id, sid)
    backend.fail_record_at = 2
    before = {item.custody_id for item in pool_module.list_custodied_pools()}
    with pytest.raises(PoolError):
        pool.free_block(lease.block_id)
    entries = [item for item in pool_module.list_custodied_pools() if item.custody_id not in before]
    assert len(entries) == 1 and entries[0].quarantined_blocks == 1
    assert entries[0].held_bytes == lease.tier_bytes
    assert entries[0].quarantined_bytes == lease.tier_bytes
    assert entries[0].in_use_bytes == entries[0].pending_bytes == 0
    pool_ref = weakref.ref(pool)
    del pool, stream
    gc.collect()
    assert pool_ref() is not None and stream_ref() is not None
    assert allocator.refs[0]() is not None
    assert all(ref() is not None for ref in backend.events)
    assert allocator.releases == 0


def test_one_delayed_borrower_prevents_storage_reuse():
    pool, backend, allocator = make_pool()
    lease = allocate(pool)
    sid = pool.register_stream(backend.stream(0))
    pool.declare_cross_stream_use(lease.block_id, sid)
    pool.free_block(lease.block_id)
    backend.events[0]().completed = True
    assert pool.drain_stream(timeout_s=0).still_pending == 1
    other = allocate(pool)
    assert other.storage_id != lease.storage_id
    backend.complete()
    pool.drain_stream(timeout_s=0)
    reused = allocate(pool)
    assert reused.storage_id == lease.storage_id and reused.block_id != lease.block_id
    assert len(allocator.refs) == 2
    finish(pool, backend, other, reused)


def test_destroy_uses_one_deadline_for_all_pending(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(barrier_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(barrier_module.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    pool, backend, _ = make_pool()
    leases = [allocate(pool) for _ in range(3)]
    for lease in leases:
        pool.free_block(lease.block_id)
    report = pool.destroy_pool(timeout_s=0.0025)
    assert now[0] <= 0.0025 + 1e-12
    assert report.drain_timed_out and len(report.pending) == 3
    backend.complete()
    assert pool.destroy_pool().completed


def test_recover_missing_custody_is_explicit_error():
    with pytest.raises(InvalidArgument):
        pool_module.recover_custodied_pool(-1)


def test_backend_without_device_validation_is_rejected():
    backend = AsyncBackend()
    backend.validate_stream_device = None
    with pytest.raises(InvalidArgument):
        create_pool(backend, form=PoolForm.TENSOR, base_allocator=Allocator(),
                    tier_spec=TierSpec(base_bytes=16, step_bytes=16, tier_count=4))
