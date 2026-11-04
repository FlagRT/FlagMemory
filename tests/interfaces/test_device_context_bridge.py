"""Host contracts for the optional public-device bridge (no hardware)."""
import pytest

from flagmemory.caps.backend import CapabilityNotDeclared, MemoryInfo
from flagmemory.caps.capabilities import Capability
from flagmemory.pool import InvalidArgument, PoolError, PoolForm, TierSpec, create_pool
from device_context_fakes import Bundle, DeviceBackend, UnifiedStream, NativeStream


@pytest.mark.parametrize("field", ["identity", "current_device", "memory_info_provider", "make_event"])
def test_missing_required_provider_fails_before_event_or_allocation(field):
    b = Bundle()
    with pytest.raises(InvalidArgument):
        b.build(**{field: None})
    assert b.event_calls == []


@pytest.mark.parametrize("index", [True, -1, 2.0, None])
def test_binding_requires_nonnegative_integer_device(index):
    b = Bundle()
    with pytest.raises(InvalidArgument):
        b.build(device_index=index)
    assert b.event_calls == []


def test_raw_current_stream_is_not_mistaken_for_unified_wrapper():
    b = Bundle()
    with pytest.raises(InvalidArgument):
        b.build(default_stream=b.backend.current_stream())
    assert b.event_calls == []


@pytest.mark.parametrize("current", [0, None, True, 2.0])
def test_current_device_must_be_reliably_bound_before_side_effects(current):
    b = Bundle()
    b.current = current
    with pytest.raises(InvalidArgument):
        b.build()
    assert b.event_calls == [] and b.memory_calls == []


def test_current_device_failure_is_not_suppressed_or_replaced_with_zero():
    b = Bundle()
    failure = RuntimeError("current unavailable")
    b.current_error = failure
    with pytest.raises(RuntimeError) as exc:
        b.build()
    assert exc.value is failure and b.event_calls == [] and b.memory_calls == []


def test_byte_provider_failure_is_fail_closed_at_construction():
    b = Bundle()
    failure = RuntimeError("driver bytes unavailable")
    b.memory_error = failure
    with pytest.raises(RuntimeError) as exc:
        b.build()
    assert exc.value is failure and b.event_calls == []
    assert b.backend.forbidden == []


def test_exact_byte_watermark_is_not_quantized_stats_or_allocator_accounting():
    b = Bundle()
    bridge = b.build()
    assert bridge.device_memory_info(2) == MemoryInfo(total=1000, used=877, free=123)
    assert b.memory_calls == [2, 2]
    assert b.backend.forbidden == []


@pytest.mark.parametrize("value", [None, {}, (123, 1000)])
def test_missing_or_non_memory_info_value_is_rejected(value):
    b = Bundle()
    b.memory_value = value
    with pytest.raises(InvalidArgument):
        b.build()


@pytest.mark.parametrize("field,value", [("free", -1), ("free", True), ("total", 1.0),
                                        ("free", 1001), ("used", 1), ("used", None)])
def test_malformed_or_inconsistent_byte_values_are_rejected(field, value):
    b = Bundle()
    # Simulate malformed results bypassing the provider's own dataclass checks.
    info = object.__new__(MemoryInfo)
    for name, original in dict(total=1000, used=877, free=123).items():
        object.__setattr__(info, name, value if field == name else original)
    b.memory_value = info
    with pytest.raises(InvalidArgument):
        b.build()


CORE_MAPPING = [("memory", Capability.DEVICE_MEMORY_INFO), ("stream", Capability.STREAM),
                ("event", Capability.EVENT), ("bounded_sync", Capability.BOUNDED_SYNC),
                ("graph_capture", Capability.GRAPH_CAPTURE),
                ("recovery_probe", Capability.PROBE_LEVEL_RECOVERY),
                ("multidevice", Capability.MULTI_DEVICE),
                ("stream_priority", Capability.STREAM_PRIORITY_RANGE)]


@pytest.mark.parametrize("key,capability", CORE_MAPPING)
def test_each_missing_device_declaration_removes_capability_and_pool_refuses(key, capability):
    from device_context_fakes import CORE_KEYS
    b = Bundle(supported=CORE_KEYS - {key})
    bridge = b.build()
    assert capability not in bridge.declared_capabilities()
    with pytest.raises(PoolError):
        create_pool(bridge, device_index=2, form=PoolForm.TENSOR,
                    base_allocator=object(), tier_spec=TierSpec(64, 64, 1))
    if key == "memory":
        assert b.memory_calls == []


def test_only_whitelisted_core_capabilities_are_declared_even_if_every_optional_key_is_true():
    from device_context_fakes import CORE_KEYS
    b = Bundle(supported=CORE_KEYS | {"memory_alloc", "memory_alloc_stat", "record_stream",
                                    "context_query", "context_lifecycle", "error_map", "recovery_real"})
    bridge = b.build()
    assert bridge.declared_capabilities() == frozenset(cap for _, cap in CORE_MAPPING)
    for call in (lambda: bridge.alloc_handle(16, 2), lambda: bridge.free_handle(None)):
        with pytest.raises(CapabilityNotDeclared) as exc:
            call()
        assert exc.value.capability is Capability.MEMORY_HANDLE
        assert exc.value.backend_name == bridge.name
    assert b.backend.forbidden == []


@pytest.mark.parametrize("key,method,cap", [
    ("memory", lambda bridge: bridge.device_memory_info(2), Capability.DEVICE_MEMORY_INFO),
    ("stream", lambda bridge: bridge.current_stream(2), Capability.STREAM)])
def test_undeclared_public_operation_fails_without_calling_provider(key, method, cap):
    from device_context_fakes import CORE_KEYS
    b = Bundle(supported=CORE_KEYS - {key})
    bridge = b.build()
    with pytest.raises(CapabilityNotDeclared) as exc:
        method(bridge)
    assert exc.value.capability is cap


def test_default_stream_is_explicit_stable_binding_and_does_not_refresh_global_current():
    b = Bundle()
    bridge = b.build()
    assert bridge.current_stream(2) is b.default
    assert [entry for entry in b.backend.audit if entry[0] == "current_stream"] == [("current_stream",)]
    assert b.backend.forbidden == []


@pytest.mark.parametrize("device,key", [(0, "default"), (None, "known"), (True, "known"),
                                        (2.0, "known"), (2, None), (2, []), (2, "")])
def test_stream_unknown_device_or_unusable_key_rejected_before_pool_registration(device, key):
    b = Bundle()
    bridge = b.build()
    stream = b.identity.add(UnifiedStream(b.backend, NativeStream()), device, key)
    with pytest.raises(InvalidArgument):
        bridge.validate_stream_device(stream, 2)


def test_same_backend_name_does_not_make_backend_instances_interchangeable():
    b = Bundle()
    bridge = b.build()
    foreign = b.stream(backend=DeviceBackend(), key="default")
    with pytest.raises(InvalidArgument):
        bridge.validate_stream_device(foreign, 2)
    with pytest.raises(InvalidArgument):
        bridge.streams_equal(b.default, foreign)


def test_stable_key_equality_is_scoped_to_verified_device_and_backend():
    b = Bundle()
    bridge = b.build()
    alias = b.stream(key="default")
    assert bridge.streams_equal(b.default, b.default)
    assert bridge.streams_equal(b.default, alias)
    assert not bridge.streams_equal(b.default, b.stream2)
    with pytest.raises(InvalidArgument):
        bridge.streams_equal(b.default, b.stream(device=0, key="default"))
    assert not [entry for entry in b.backend.audit if entry[0] == "native"]


def test_every_device_operation_rechecks_current_without_switching_it():
    b = Bundle()
    bridge = b.build()
    b.current = 0
    for operation in (lambda: bridge.current_stream(2), lambda: bridge.device_memory_info(2),
                      lambda: bridge.validate_stream_device(b.default, 2)):
        with pytest.raises(InvalidArgument):
            operation()
    assert b.event_calls == [] and b.backend.forbidden == []


from flagmemory.caps.backend import EventNotRecorded


def test_unrecorded_native_true_never_reaches_pool_completion():
    b = Bundle()
    bridge = b.build()
    event = bridge.create_event(2)
    with pytest.raises(EventNotRecorded):
        bridge.query_event(event)
    assert b.event_calls == [2]
    assert not [item for item in b.backend.audit if item[0] == "query"]
    assert ("create_event", False) in b.backend.audit


def test_record_uses_exact_verified_unified_stream_then_queries_public_event():
    b = Bundle()
    bridge = b.build()
    event = bridge.create_event(2)
    bridge.record_event(event, b.stream2)
    records = [entry for entry in b.backend.audit if entry[0] == "record"]
    assert len(records) == 1 and records[0][2] is b.stream2
    assert not bridge.query_event(event)
    b.complete_all()
    assert bridge.query_event(event)
    assert b.backend.forbidden == []


@pytest.mark.parametrize("error", [RuntimeError("record uncertain"), ValueError("record rejected"),
                                   NotImplementedError("record unsupported")])
def test_record_failure_never_creates_completion_or_permits_uncertain_retry(error):
    b = Bundle()
    bridge = b.build()
    event = bridge.create_event(2)
    b.backend.events[0].record_error = error
    expected = (CapabilityNotDeclared if isinstance(error, NotImplementedError)
                else InvalidArgument if isinstance(error, ValueError) else RuntimeError)
    with pytest.raises(expected) as exc:
        bridge.record_event(event, b.default)
    if expected is RuntimeError:
        assert exc.value is error
    else:
        assert exc.value.__cause__ is error
    b.backend.events[0].record_error = None
    with pytest.raises(EventNotRecorded):
        bridge.query_event(event)
    with pytest.raises(InvalidArgument):
        bridge.record_event(event, b.default)
    assert not [entry for entry in b.backend.audit if entry[0] == "query"]
    assert len([entry for entry in b.backend.audit if entry[0] == "record"]) == 1


def test_duplicate_record_and_cross_bridge_or_foreign_event_are_rejected():
    b = Bundle()
    bridge = b.build()
    other = b.build()
    event = bridge.create_event(2)
    bridge.record_event(event, b.default)
    for call in (lambda: bridge.record_event(event, b.default),
                 lambda: other.record_event(event, b.default),
                 lambda: other.query_event(event), lambda: bridge.query_event(object())):
        with pytest.raises(InvalidArgument):
            call()
    assert len([entry for entry in b.backend.audit if entry[0] == "record"]) == 1


def test_event_factory_wrong_backend_is_rejected_even_with_same_name():
    from device_context_fakes import UnifiedEvent, NativeEvent
    b = Bundle()
    bridge = b.build(make_event=lambda index: UnifiedEvent(DeviceBackend(), NativeEvent()))
    with pytest.raises(InvalidArgument):
        bridge.create_event(2)


def test_cross_device_stream_is_rejected_before_event_record():
    b = Bundle()
    bridge = b.build()
    event = bridge.create_event(2)
    with pytest.raises(InvalidArgument):
        bridge.record_event(event, b.stream(device=0))
    assert not [entry for entry in b.backend.audit if entry[0] == "record"]


def test_wrong_current_device_prevents_event_creation_record_and_query():
    b = Bundle()
    bridge = b.build()
    event = bridge.create_event(2)
    bridge.record_event(event, b.default)
    b.current = 0
    for call in (lambda: bridge.create_event(2), lambda: bridge.record_event(event, b.default),
                 lambda: bridge.query_event(event)):
        with pytest.raises(InvalidArgument):
            call()
    assert b.event_calls == [2]
    assert not [entry for entry in b.backend.audit if entry[0] == "query"]


def test_device_change_during_factory_fails_without_record():
    b = Bundle()
    def factory(index):
        event = b.make_event(index)
        b.current = 0
        return event
    bridge = b.build(make_event=factory)
    with pytest.raises(InvalidArgument):
        bridge.create_event(2)
    assert b.event_calls == [2]
    assert not [entry for entry in b.backend.audit if entry[0] == "record"]


def test_device_change_after_record_keeps_event_uncertain_and_not_retriable():
    b = Bundle()
    bridge = b.build()
    event = bridge.create_event(2)
    public_event = [entry for entry in b.backend.audit if entry[0] == "create_event"]
    assert len(public_event) == 1
    from device_context_fakes import UnifiedEvent
    original = UnifiedEvent.record
    # Patch only this public operation, preserving its record side effect.
    def changing_record(self, stream):
        original(self, stream)
        b.current = 0
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(UnifiedEvent, "record", changing_record)
        with pytest.raises(InvalidArgument):
            bridge.record_event(event, b.default)
    b.current = 2
    with pytest.raises(EventNotRecorded):
        bridge.query_event(event)
    with pytest.raises(InvalidArgument):
        bridge.record_event(event, b.default)


@pytest.mark.parametrize("error,expected", [(NotImplementedError("not supported"), CapabilityNotDeclared),
                                          (ValueError("bad input"), InvalidArgument),
                                          (RuntimeError("vendor fault"), RuntimeError)])
@pytest.mark.parametrize("operation", ["bytes", "factory", "query"])
def test_public_errors_map_only_unsupported_and_invalid_arguments(error, expected, operation):
    b = Bundle()
    bridge = b.build()
    if operation == "bytes":
        b.memory_error = error
        call = lambda: bridge.device_memory_info(2)
        cap = Capability.DEVICE_MEMORY_INFO
    elif operation == "factory":
        def failing_factory(index):
            raise error
        bridge = b.build(make_event=failing_factory)
        call = lambda: bridge.create_event(2)
        cap = Capability.EVENT
    else:
        event = bridge.create_event(2)
        bridge.record_event(event, b.default)
        b.backend.events[-1].query_error = error
        call = lambda: bridge.query_event(event)
        cap = Capability.EVENT
    with pytest.raises(expected) as exc:
        call()
    if expected is RuntimeError:
        assert exc.value is error
    else:
        assert exc.value.__cause__ is error
        if expected is CapabilityNotDeclared:
            assert exc.value.capability is cap


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf"), -float("inf"), True, None, "1"])
def test_wait_budget_must_be_finite_nonnegative_real(timeout):
    b = Bundle()
    bridge = b.build()
    event = bridge.create_event(2)
    bridge.record_event(event, b.default)
    with pytest.raises(InvalidArgument):
        bridge.synchronize_event(event, timeout)
    assert not [entry for entry in b.backend.audit if entry[0] == "query"]


def test_zero_wait_queries_once_without_sleep_or_host_wait(monkeypatch):
    import flagmemory.adapters.device_context as module
    b = Bundle()
    bridge = b.build()
    event = bridge.create_event(2)
    bridge.record_event(event, b.default)
    def forbid_sleep(seconds):
        raise AssertionError("zero wait must not sleep")
    monkeypatch.setattr(module.time, "sleep", forbid_sleep)
    assert bridge.synchronize_event(event, 0) is False
    assert len([entry for entry in b.backend.audit if entry[0] == "query"]) == 1
    assert b.backend.forbidden == []


@pytest.mark.parametrize("complete_after,expected", [(None, False), (0.002, True)])
def test_positive_wait_uses_one_monotonic_deadline(monkeypatch, complete_after, expected):
    import flagmemory.adapters.device_context as module
    b = Bundle()
    bridge = b.build()
    event = bridge.create_event(2)
    bridge.record_event(event, b.default)
    now, sleeps = [0.0], []
    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds
        if complete_after is not None and now[0] >= complete_after:
            b.complete_all()
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(module.time, "sleep", sleep)
    assert bridge.synchronize_event(event, 0.0025) is expected
    assert now[0] <= 0.0025 + 1e-12
    assert sum(sleeps) == pytest.approx(now[0])
    assert b.backend.forbidden == []


@pytest.mark.parametrize("missing,operation,cap", [
    ("event", "create", Capability.EVENT), ("bounded_sync", "wait", Capability.BOUNDED_SYNC)])
def test_undeclared_event_and_wait_capabilities_do_not_call_lower_api(missing, operation, cap):
    from device_context_fakes import CORE_KEYS
    b = Bundle(supported=CORE_KEYS - {missing})
    bridge = b.build()
    event = None
    if operation == "wait":
        event = bridge.create_event(2)
        bridge.record_event(event, b.default)
    with pytest.raises(CapabilityNotDeclared) as exc:
        if operation == "create":
            bridge.create_event(2)
        else:
            bridge.synchronize_event(event, 0)
    assert exc.value.capability is cap
    assert not [entry for entry in b.backend.audit if entry[0] == "query"]


from flagmemory.pool import OwnershipViolation, Purpose, ReusePolicy
from device_context_fakes import ReadyAllocator


@pytest.fixture
def bridge_bundle():
    b = Bundle()
    b.build()
    b.allocator = ReadyAllocator(b.bridge)
    return b


def make_tensor_pool(b):
    # 64 bytes is a host fixture tier, never a chip granularity claim.
    return create_pool(b.bridge, device_index=2, form=PoolForm.TENSOR,
                       base_allocator=b.allocator, tier_spec=TierSpec(64, 64, 1),
                       reuse=ReusePolicy(opportunistic=False))


def test_pool_uses_borrower_event_before_reuse(bridge_bundle):
    b = bridge_bundle
    pool = make_tensor_pool(b)
    a = pool.alloc_on_stream(16, purpose=Purpose.SCRATCH)
    ref = pool.access_block(a.block_id)
    assert ref.ready
    ref[:4] = b"pool"
    assert ref[:4] == b"pool"
    assert b.allocator.streams[0]() is b.default
    sid = pool.register_stream(b.stream2)
    pool.declare_cross_stream_use(a.block_id, sid)
    del ref
    verdict = pool.verify_block_ownership(a.block_id)
    assert verdict.context_ok is None
    assert "no_context_observe" in {d.key for d in verdict.degradations}
    assert "tensor_level_pool" in {d.key for d in pool.query_capability().degradations}
    pool.free_block(a.block_id)
    assert len(b.backend.events) == 2
    pool.drain_stream(timeout_s=0)
    assert pool.query_watermark().pool_pending == 64
    assert pool.query_watermark().pool_reusable == 0
    b.backend.events[0].completed = True  # Owner done, borrower still running.
    pool.drain_stream(timeout_s=0)
    assert pool.query_watermark().pool_pending == 64
    assert pool.query_watermark().pool_reusable == 0
    b.backend.events[1].completed = True
    pool.drain_stream(timeout_s=0)
    assert pool.query_watermark().pool_reusable == 64
    c = pool.alloc_on_stream(16, purpose=Purpose.SCRATCH)
    assert c.storage_id == a.storage_id and c.block_id != a.block_id
    for operation in (pool.access_block, pool.free_block):
        with pytest.raises(OwnershipViolation):
            operation(a.block_id)
    pool.free_block(c.block_id)
    b.complete_all()
    report = pool.destroy_pool(timeout_s=0)
    assert report.completed and not report.pending and not report.quarantined and not report.leaked
    assert b.allocator.allocate_calls == b.allocator.release_calls == 1
    assert b.allocator.live_bytes == 0
    assert b.event_calls == [2, 2, 2]
    assert len([entry for entry in b.backend.audit if entry[0] == "record"]) == 3
    # Two first polls check both streams; confirmed owner is skipped thereafter.
    # Borrower completion + the new lease's owner event add one query each.
    assert len([entry for entry in b.backend.audit if entry[0] == "query"]) == 6
    assert b.backend.forbidden == []


def test_first_use_is_ready_on_original_unified_stream_and_invalid_stream_never_allocates(bridge_bundle):
    b = bridge_bundle
    pool = make_tensor_pool(b)
    for stream in (b.stream(device=0), b.stream(key=None), UnifiedStream(b.backend, NativeStream())):
        with pytest.raises(InvalidArgument):
            pool.register_stream(stream)
    with pytest.raises(InvalidArgument):
        pool.alloc_on_stream(16, stream_id=999, purpose=Purpose.SCRATCH)
    assert b.allocator.allocate_calls == 0
    sid = pool.register_stream(b.stream2)
    lease = pool.alloc_on_stream(16, stream_id=sid, purpose=Purpose.SCRATCH)
    assert b.allocator.streams[0]() is b.stream2
    ref = pool.access_block(lease.block_id)
    assert ref.ready
    ref[:5] = b"ready"
    assert ref[:5] == b"ready"
    del ref
    pool.free_block(lease.block_id)
    b.complete_all()
    assert pool.destroy_pool(timeout_s=0).completed
    assert b.backend.forbidden == []


def test_pool_rejects_mismatched_binding_before_any_base_allocation(bridge_bundle):
    b = bridge_bundle
    with pytest.raises(InvalidArgument):
        create_pool(b.bridge, device_index=0, form=PoolForm.TENSOR,
                    base_allocator=b.allocator, tier_spec=TierSpec(64, 64, 1))
    assert b.allocator.allocate_calls == 0 and b.event_calls == []


@pytest.mark.parametrize("fault", ["record", "query", "release"])
def test_fault_custody_keeps_resources_and_never_retries_uncertain_release(bridge_bundle, fault):
    import gc
    import weakref
    from flagmemory.pool import list_custodied_pools, recover_custodied_pool
    from device_context_fakes import UnifiedEvent
    b = bridge_bundle
    pool = make_tensor_pool(b)
    lease = pool.alloc_on_stream(16, purpose=Purpose.SCRATCH)
    sid = pool.register_stream(b.stream2)
    pool.declare_cross_stream_use(lease.block_id, sid)
    stream_ref = weakref.ref(b.stream2)
    allocator = b.allocator
    if fault == "record":
        original = UnifiedEvent.record
        def fail_record(event, stream):
            original(event, stream)  # Side effect may already have happened.
            raise RuntimeError("record outcome uncertain")
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(UnifiedEvent, "record", fail_record)
            with pytest.raises(PoolError):
                pool.free_block(lease.block_id)
    else:
        pool.free_block(lease.block_id)
        if fault == "query":
            b.backend.events[0].query_error = RuntimeError("query failed")
            outcome = pool.drain_stream(timeout_s=0)
            assert outcome.promoted_blocks == 0
        else:
            b.complete_all()
            pool.drain_stream(timeout_s=0)
            allocator.fail_release = True
            with pytest.raises(PoolError):
                pool.destroy_pool(timeout_s=0)
    assert pool.query_watermark().pool_reusable == 0
    report = pool.destroy_pool(wait_for_drain=False, timeout_s=0)
    assert not report.completed and len(report.quarantined) == 1
    assert not report.pending and not report.leaked
    custody_id = report.custody_id
    assert custody_id in {entry.custody_id for entry in list_custodied_pools()}
    pool_ref = weakref.ref(pool)
    assert b.backend.forbidden == []
    del pool, b
    gc.collect()
    assert pool_ref() is not None and allocator.refs[0]() is not None and stream_ref() is not None
    recovered = recover_custodied_pool(custody_id)
    for _ in range(2):
        assert not recovered.destroy_pool(wait_for_drain=False, timeout_s=0).completed
    assert allocator.release_calls == (1 if fault == "release" else 0)
    assert allocator.live_bytes == 64
    # No public quarantine-unlock protocol exists. Keep safe process custody;
    # never clear the production custody table or retry an uncertain release.


def test_healthy_pending_custody_can_complete_through_public_recovery(bridge_bundle):
    from flagmemory.pool import list_custodied_pools, recover_custodied_pool
    b = bridge_bundle
    pool = make_tensor_pool(b)
    lease = pool.alloc_on_stream(16, purpose=Purpose.SCRATCH)
    pool.free_block(lease.block_id)
    report = pool.destroy_pool(timeout_s=0)
    assert not report.completed and report.pending
    recovered = recover_custodied_pool(report.custody_id)
    b.complete_all()
    assert recovered.destroy_pool(timeout_s=0).completed
    assert report.custody_id not in {entry.custody_id for entry in list_custodied_pools()}
    assert b.allocator.release_calls == 1 and b.allocator.live_bytes == 0
    assert b.backend.forbidden == []


@pytest.mark.parametrize("observation", ["stream_device", "stream_key"])
@pytest.mark.parametrize("entry", ["validate", "current", "equal", "record"])
def test_identity_callback_device_drift_rejected_before_dependent_action(observation, entry):
    b = Bundle()
    bridge = b.build()
    event = bridge.create_event(2)
    original = getattr(b.identity, observation)
    def drifting_identity(stream):
        result = original(stream)
        b.current = 0
        return result
    setattr(b.identity, observation, drifting_identity)
    operations = {
        "validate": lambda: bridge.validate_stream_device(b.default, 2),
        "current": lambda: bridge.current_stream(2),
        "equal": lambda: bridge.streams_equal(b.default, b.stream2),
        "record": lambda: bridge.record_event(event, b.default),
    }
    with pytest.raises(InvalidArgument):
        operations[entry]()
    # A post-record rejection is too late: neither the public record nor its
    # native side effect may occur after identity observation changed scope.
    assert not [item for item in b.backend.audit if item[0] == "record"]
    assert b.backend.events[0].recorded_stream is None
    b.current = 2
    with pytest.raises(EventNotRecorded):
        bridge.query_event(event)
    assert not [item for item in b.backend.audit if item[0] == "query"]


def test_stream_key_hash_device_drift_rejected_before_public_record():
    b = Bundle()
    bridge = b.build()
    event = bridge.create_event(2)
    class DriftingKey:
        def __hash__(self):
            b.current = 0
            return 42
    b.identity.add(b.default, 2, DriftingKey())
    with pytest.raises(InvalidArgument):
        bridge.record_event(event, b.default)
    assert not [item for item in b.backend.audit if item[0] == "record"]
    assert b.backend.events[0].recorded_stream is None
    b.current = 2
    with pytest.raises(EventNotRecorded):
        bridge.query_event(event)


@pytest.mark.parametrize("failure,expected", [(RuntimeError("probe failed"), RuntimeError),
                                             (ValueError("probe rejected"), InvalidArgument)])
def test_final_probe_failure_after_identity_keeps_original_exception_contract(failure, expected):
    b = Bundle()
    bridge = b.build()
    event = bridge.create_event(2)
    original = b.identity.stream_key
    def identity_then_failed_probe(stream):
        result = original(stream)
        b.current_error = failure
        return result
    b.identity.stream_key = identity_then_failed_probe
    with pytest.raises(expected) as exc:
        bridge.record_event(event, b.default)
    if expected is RuntimeError:
        assert exc.value is failure
    else:
        assert exc.value.__cause__ is failure
    assert not [item for item in b.backend.audit if item[0] == "record"]
    assert b.backend.events[0].recorded_stream is None
    b.current_error = None
    with pytest.raises(EventNotRecorded):
        bridge.query_event(event)
