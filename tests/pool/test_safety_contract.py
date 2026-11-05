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
    assert (report.returned_blocks, report.returned_bytes) == (0, 0)
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


def test_repeated_storage_reuse_keeps_lease_metadata_bounded():
    """A single resident block must not accumulate one record per past lease."""
    pool, backend, allocator = make_pool()
    old_leases = []
    retained_records = []
    for cycle in range(256):
        lease = allocate(pool)
        old_leases.append(lease.block_id)
        pool.free_block(lease.block_id)
        backend.complete()
        pool.drain_stream(timeout_s=0)
        if cycle + 1 in (8, 128, 256):
            # Count retained ledger collection entries, not allocator/test history.
            retained_records.append(sum(
                len(value) for value in vars(pool._ledger).values()
                if isinstance(value, (dict, list, set))
            ))
    assert retained_records[0] == retained_records[1] == retained_records[2], retained_records
    current = allocate(pool)
    assert len(allocator.refs) == 1
    for stale in old_leases:
        for action in (pool.free_block, pool.block_view, pool.verify_block_ownership, pool.access_block):
            with pytest.raises(OwnershipViolation):
                action(stale)
        with pytest.raises(OwnershipViolation):
            pool.declare_cross_stream_use(stale, 0)
    assert pool.block_view(current.block_id).state is BlockState.IN_USE
    assert pool.query_audit().quarantined_blocks == 0
    finish(pool, backend, current)


class FirstUseNotReady(RuntimeError):
    pass


class FirstUseResource:
    """Host-only contract model: memory cannot be consumed on an unready stream."""
    def __init__(self, nbytes):
        self._data = bytearray(nbytes)
        self._ready_stream = None
        self.writes = 0

    def establish_ready(self, stream):
        self._ready_stream = stream

    def write_on(self, stream, content):
        if stream is not self._ready_stream:
            raise FirstUseNotReady("first consumption has no target-stream readiness")
        self._data[:] = content
        self.writes += 1

    def read_on(self, stream):
        if stream is not self._ready_stream:
            raise FirstUseNotReady("first consumption has no target-stream readiness")
        return bytes(self._data)


class FirstUseAllocator(Allocator):
    """A correct adapter and two deliberately broken adapters for the same consumer."""
    def __init__(self, mode, fallback_stream):
        super().__init__()
        self.mode = mode
        self.fallback_stream = fallback_stream

    def allocate(self, nbytes, device_index=0, *, stream):
        resource = FirstUseResource(nbytes)
        if self.mode == "target":
            resource.establish_ready(stream)
        elif self.mode == "ignored_target":
            resource.establish_ready(self.fallback_stream)
        elif self.mode != "missing_dependency":
            raise AssertionError("unknown adapter mode")
        self.refs.append(weakref.ref(resource))
        return resource


def consume_first_target_allocation(mode):
    """Exercise the real pool/access path; model only the adapter's readiness rule."""
    backend = AsyncBackend()
    target, fallback = backend.stream(0), backend.stream(0)
    allocator = FirstUseAllocator(mode, fallback)
    pool = create_pool(backend, form=PoolForm.TENSOR, base_allocator=allocator,
                       tier_spec=TierSpec(base_bytes=16, step_bytes=16, tier_count=4))
    sid = pool.register_stream(target)
    first = pool.alloc_on_stream(16, stream_id=sid, purpose=Purpose.ACTIVATION)
    try:
        ref = pool.access_block(first.block_id)
        # This is the first consumer operation, not an assertion about a logged argument.
        ref.write_on(target, b"0123456789abcdef")
        assert ref.read_on(target) == b"0123456789abcdef"
        assert allocator.refs[0]() is ref
    finally:
        del ref
        pool.free_block(first.block_id)
        backend.complete()
        pool.drain_stream(timeout_s=0)
        # On a deliberately broken adapter, close before propagating consumption failure.
        if mode != "target":
            pool.destroy_pool()
    second = pool.alloc_on_stream(16, stream_id=sid, purpose=Purpose.ACTIVATION)
    assert second.storage_id == first.storage_id and second.block_id != first.block_id
    assert len(allocator.refs) == 1
    with pytest.raises(OwnershipViolation):
        pool.access_block(first.block_id)
    ref = pool.access_block(second.block_id)
    assert ref.read_on(target) == b"0123456789abcdef"
    ref.write_on(target, b"fedcba9876543210")
    assert ref.read_on(target) == b"fedcba9876543210"
    del ref
    finish(pool, backend, second)


def test_first_target_consumption_reads_and_writes_public_pool_storage():
    consume_first_target_allocation("target")


@pytest.mark.parametrize("mode", ["ignored_target", "missing_dependency"])
def test_first_target_consumption_rejects_broken_readiness_adapters(mode):
    # The same first consumer operation must reject an adapter that only accepts stream.
    with pytest.raises(FirstUseNotReady, match="no target-stream readiness"):
        consume_first_target_allocation(mode)


@pytest.mark.parametrize("purpose,threshold", [
    (None, None),
    (Purpose.ACTIVATION, 0),
    (Purpose.ACTIVATION, None),
], ids=["default-purpose", "zero-retention", "reusable-control"])
def test_destroy_report_counts_every_successful_release(purpose, threshold):
    backend, allocator = AsyncBackend(), Allocator()
    pool = create_pool(backend, form=PoolForm.TENSOR, base_allocator=allocator,
                       tier_spec=TierSpec(16, 16, 2),
                       reuse=pool_module.ReusePolicy(release_threshold_bytes=threshold))
    lease = pool.alloc_on_stream(16, **({} if purpose is None else {"purpose": purpose}))
    pool.free_block(lease.block_id)
    backend.complete()
    report = pool.destroy_pool(timeout_s=0)
    assert report.completed
    assert allocator.releases == 1
    assert (report.returned_blocks, report.returned_bytes) == (1, 16)


def test_destroy_report_combines_direct_and_retained_mixed_tiers():
    backend, allocator = AsyncBackend(), Allocator()
    pool = create_pool(backend, form=PoolForm.TENSOR, base_allocator=allocator,
                       tier_spec=TierSpec(16, 16, 3),
                       reuse=pool_module.ReusePolicy(release_threshold_bytes=16))
    leases = [pool.alloc_on_stream(16),
              pool.alloc_on_stream(16, purpose=Purpose.ACTIVATION),
              pool.alloc_on_stream(32, purpose=Purpose.SCRATCH)]
    for lease in leases:
        pool.free_block(lease.block_id)
    backend.complete()
    report = pool.destroy_pool(timeout_s=0)
    assert report.completed
    assert allocator.releases == 3
    assert (report.returned_blocks, report.returned_bytes) == (3, 64)


def test_destroy_report_excludes_releases_before_call():
    pool, backend, allocator = make_pool()
    earlier = pool.alloc_on_stream(16)
    current = allocate(pool)
    pool.free_block(earlier.block_id)
    backend.complete()
    pool.drain_stream(timeout_s=0)
    assert allocator.releases == 1
    pool.free_block(current.block_id)
    backend.complete()
    report = pool.destroy_pool(timeout_s=0)
    assert report.completed and allocator.releases == 2
    assert (report.returned_blocks, report.returned_bytes) == (1, 16)


@pytest.mark.parametrize("live", [False, True], ids=["pending", "live-lease"])
def test_destroy_report_counts_only_this_call_on_retry(live):
    pool, backend, allocator = make_pool()
    ready = pool.alloc_on_stream(16)
    later = pool.alloc_on_stream(32, purpose=Purpose.ACTIVATION)
    pool.free_block(ready.block_id)
    backend.complete()
    if live:
        with pytest.raises(LeakDetected) as exc:
            pool.destroy_pool(timeout_s=0)
        report = exc.value.report
        assert len(report.leaked) == 1
    else:
        pool.free_block(later.block_id)
        report = pool.destroy_pool(timeout_s=0)
        assert len(report.pending) == 1
    assert not report.completed and allocator.releases == 1
    assert (report.returned_blocks, report.returned_bytes) == (1, 16)
    if live:
        pool.free_block(later.block_id)
    backend.complete()
    report = pool.destroy_pool(timeout_s=0)
    assert report.completed and allocator.releases == 2
    assert (report.returned_blocks, report.returned_bytes) == (1, 32)


@pytest.mark.parametrize("purpose", [Purpose.UNDECLARED, Purpose.ACTIVATION])
def test_destroy_report_never_counts_failed_release_or_retries_it(purpose):
    pool, backend, allocator = make_pool()
    lease = pool.alloc_on_stream(16, purpose=purpose)
    pool.free_block(lease.block_id)
    backend.complete()
    allocator.fail_release = True
    with pytest.raises(PoolError, match="release side effect then failure"):
        pool.destroy_pool(timeout_s=0)
    for _ in range(2):
        report = pool.destroy_pool(timeout_s=0)
        assert not report.completed and len(report.quarantined) == 1
        assert (report.returned_blocks, report.returned_bytes) == (0, 0)
        assert allocator.releases == 1
        assert allocator.refs[0]() is not None


def test_destroy_report_excludes_success_from_call_interrupted_by_release_error():
    class FailSecondRelease(Allocator):
        def release(self, resource):
            self.fail_release = self.releases == 1
            super().release(resource)

    backend, allocator = AsyncBackend(), FailSecondRelease()
    pool = create_pool(backend, form=PoolForm.TENSOR, base_allocator=allocator,
                       tier_spec=TierSpec(16, 16, 3))
    leases = [pool.alloc_on_stream(size) for size in (16, 32, 48)]
    for lease in leases:
        pool.free_block(lease.block_id)
    backend.complete()
    with pytest.raises(PoolError, match="release side effect then failure"):
        pool.destroy_pool(timeout_s=0)
    assert allocator.releases == 2
    report = pool.destroy_pool(timeout_s=0)
    assert len(report.quarantined) == 1 and not report.completed
    # The 48-byte request occupies the 64-byte tier.
    assert (report.returned_blocks, report.returned_bytes) == (1, 64)
    assert allocator.releases == 3
    report = pool.destroy_pool(timeout_s=0)
    assert (report.returned_blocks, report.returned_bytes) == (0, 0)
    assert allocator.releases == 3


def test_destroy_report_does_not_count_event_quarantine():
    pool, backend, allocator = make_pool()
    lease = pool.alloc_on_stream(16)
    pool.free_block(lease.block_id)
    backend.fault = True
    report = pool.destroy_pool(timeout_s=0)
    assert len(report.quarantined) == 1 and not report.completed
    assert (report.returned_blocks, report.returned_bytes) == (0, 0)
    assert allocator.releases == 0 and allocator.refs[0]() is not None
