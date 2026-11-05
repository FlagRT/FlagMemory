import pytest


def test_invalid_input_rejected_before_allocation():
    from pooled_model_inputs import lease_input_ids
    class IDs:
        shape = (2, 3)
    class Pool:
        def alloc_on_stream(self, *args, **kwargs):
            pytest.fail('allocation must not happen')
    with pytest.raises(ValueError):
        lease_input_ids(Pool(), object(), IDs(), stream=object(), stream_id=1, device=2)


def test_gate_rejects_missing_weights():
    from test_onerec_inputs_ascend import validate_model_gate
    with pytest.raises(ValueError, match='weight'):
        validate_model_gate({'weights_confirmed': False})


@pytest.mark.parametrize('left,right', [([1.0, 2.0], [1.0, float('nan')]),
                                       ([1.0, 2.0], [1.0, 3.0]),
                                       ([1.0, 2.0], [2.0, 1.0])])
def test_compare_rejects_bad_logits(left, right):
    from test_onerec_inputs_ascend import compare_forward
    with pytest.raises(ValueError):
        compare_forward(left, right, atol=1e-3, rtol=1e-3)


from contextlib import nullcontext
import struct
from types import SimpleNamespace


class FakeTensor:
    def __init__(self, values=None, *, buf=None, shape=(1, 3), dtype='int64', device='cpu', index=None, offset=0, contiguous=True, stride=None):
        self.shape, self.dtype = shape, dtype
        self.device = SimpleNamespace(type=device, index=index)
        self.offset, self.contiguous = offset, contiguous
        self._stride = stride or ((shape[1], 1) if len(shape) == 2 else (1,))
        self.buf = buf if buf is not None else bytearray(max(8, self.numel() * self.element_size() + offset))
        if values is not None:
            for i, v in enumerate(values): struct.pack_into('q', self.buf, offset + 8*i, v)
    def numel(self):
        from math import prod
        return prod(self.shape)
    def element_size(self): return 8 if self.dtype == 'int64' else 1
    def stride(self): return self._stride
    def is_contiguous(self): return self.contiguous
    def storage_offset(self): return self.offset // self.element_size()
    def untyped_storage(self): return SimpleNamespace(data_ptr=lambda: id(self.buf))
    def data_ptr(self): return id(self.buf) + self.offset
    def narrow(self, dim, start, length):
        return FakeTensor(buf=self.buf, shape=(length,), dtype=self.dtype, device=self.device.type, index=self.device.index, offset=self.offset+start)
    def view(self, *args):
        dtype = args[0] if len(args) == 1 and isinstance(args[0], str) else self.dtype
        shape = (self.numel() // 8,) if dtype == 'int64' and self.dtype == 'uint8' else args
        return FakeTensor(buf=self.buf, shape=shape, dtype=dtype, device=self.device.type, index=self.device.index, offset=self.offset)
    def copy_(self, source, non_blocking=False):
        self.buf[self.offset:self.offset+self.numel()*8] = source.buf[source.offset:source.offset+source.numel()*8]
    def clone(self):
        out = FakeTensor(shape=self.shape, dtype=self.dtype, device=self.device.type, index=self.device.index)
        out.buf[:] = self.buf[self.offset:self.offset+len(out.buf)]
        return out
    def values(self): return [struct.unpack_from('q', self.buf, self.offset+i*8)[0] for i in range(self.numel())]


class FakePool:
    def __init__(self):
        self.base = FakeTensor(shape=(8192,), dtype='uint8', device='npu', index=2)
        self.next = 0; self.free_calls = 0; self.fail_free = False
    def alloc_on_stream(self, *args, **kwargs):
        self.next += 1
        return SimpleNamespace(block_id=self.next, storage_id=1)
    def access_block(self, block): return self.base
    def free_block(self, block):
        self.free_calls += 1
        if self.fail_free: raise RuntimeError('quarantined')
    def drain_stream(self, **kwargs): return SimpleNamespace(recoverable_timeout=False, still_pending=0, quarantined_blocks=[])


class FakeBackend:
    def __init__(self, *, done=True, fail=None): self.done, self.fail = done, fail
    def create_event(self, device_index): return object()
    def record_event(self, event, stream):
        if self.fail == 'record': raise RuntimeError('record')
    def query_event(self, event):
        if self.fail == 'query': raise RuntimeError('query')
        return self.done


@pytest.fixture
def input_case():
    pool = FakePool()
    runtime = SimpleNamespace(int64='int64', uint8='uint8', npu=SimpleNamespace(device=lambda _: nullcontext(), stream=lambda _: nullcontext()))
    stream = SimpleNamespace(device=SimpleNamespace(type='npu', index=2))
    return pool, runtime, FakeTensor([3, 9, 17]), stream


def test_consumer_reads_pool_bytes_and_reuse(input_case):
    from pooled_model_inputs import lease_input_ids, assert_consumed_input, release_input_ids
    pool, runtime, ids, stream = input_case
    first = lease_input_ids(pool, runtime, ids, stream=stream, stream_id=3, device=2)
    assert first.tensor.values() == [3, 9, 17]
    assert assert_consumed_input(first, first.tensor, device=2)['storage_id'] == 1
    with pytest.raises(ValueError): assert_consumed_input(first, first.tensor.clone(), device=2)
    release_input_ids(first, pool, FakeBackend(), stream=stream, timeout_s=1)
    assert first.closed and pool.free_calls == 1
    with pytest.raises(ValueError): assert_consumed_input(first, pool.base, device=2)
    second = lease_input_ids(pool, runtime, ids, stream=stream, stream_id=3, device=2)
    assert second.storage_id == first.storage_id and second.block_id != first.block_id


@pytest.mark.parametrize('failure', ['record', 'query', 'pending'])
def test_pre_return_failure_keeps_live_lease(input_case, failure):
    from pooled_model_inputs import lease_input_ids, release_input_ids, assert_consumed_input
    pool, runtime, ids, stream = input_case
    lease = lease_input_ids(pool, runtime, ids, stream=stream, stream_id=3, device=2)
    with pytest.raises((RuntimeError, TimeoutError)):
        release_input_ids(lease, pool, FakeBackend(done=False, fail=failure), stream=stream, timeout_s=0)
    assert not lease.closed and not lease.return_attempted and pool.free_calls == 0
    assert_consumed_input(lease, lease.tensor, device=2)


def test_failed_return_cannot_be_consumed_or_retried(input_case):
    from pooled_model_inputs import lease_input_ids, release_input_ids, assert_consumed_input
    pool, runtime, ids, stream = input_case
    lease = lease_input_ids(pool, runtime, ids, stream=stream, stream_id=3, device=2)
    alias = lease.tensor; pool.fail_free = True
    with pytest.raises(RuntimeError): release_input_ids(lease, pool, FakeBackend(), stream=stream, timeout_s=1)
    assert not lease.closed and lease.return_attempted and pool.free_calls == 1
    with pytest.raises(ValueError): assert_consumed_input(lease, alias, device=2)
    with pytest.raises(ValueError): release_input_ids(lease, pool, FakeBackend(), stream=stream, timeout_s=1)


@pytest.mark.parametrize('shape,dtype,device,contiguous,stride', [((1,0),'int64','cpu',True,None), ((1,129),'int64','cpu',True,None), ((2,3),'int64','cpu',True,None), ((1,3),'uint8','cpu',True,None), ((1,3),'int64','npu',True,None), ((1,3),'int64','cpu',False,None), ((1,3),'int64','cpu',True,(4,1))])
def test_invalid_cpu_input_rejected(input_case, shape, dtype, device, contiguous, stride):
    from pooled_model_inputs import lease_input_ids
    pool, runtime, _, stream = input_case
    ids = FakeTensor(shape=shape, dtype=dtype, device=device, contiguous=contiguous, stride=stride)
    with pytest.raises(ValueError): lease_input_ids(pool, runtime, ids, stream=stream, stream_id=3, device=2)
    assert pool.next == 0


def test_invalid_backing_yields_custodied_lease(input_case):
    from pooled_model_inputs import lease_input_ids, InputLeaseError
    pool, runtime, ids, stream = input_case
    pool.base = FakeTensor(shape=(8,), dtype='uint8', device='npu', index=2)
    with pytest.raises(InputLeaseError) as failure:
        lease_input_ids(pool, runtime, ids, stream=stream, stream_id=3, device=2)
    assert failure.value.lease.block_id == 1 and pool.free_calls == 0


def test_gate_requires_resource_and_evidence(tmp_path, monkeypatch):
    import datetime, hashlib, json
    from test_onerec_inputs_ascend import validate_model_gate, MODEL, REVISION
    monkeypatch.setattr('test_onerec_inputs_ascend._check_readonly', lambda _: None)
    model_dir = tmp_path / 'model'; model_dir.mkdir()
    files = {'config.json': '{}', 'tokenizer.json': '{}', 'model.safetensors.index.json': json.dumps({'weight_map': {'w': 'one.safetensors'}}), 'one.safetensors': 'sample'}
    digests = {}
    for name, value in files.items():
        data = value.encode(); (model_dir / name).write_bytes(data); digests[name] = hashlib.sha256(data).hexdigest()
    evidence_items = {}
    for stage in ('mapping', 'health'):
        evidence = tmp_path / (stage + '.json')
        evidence.write_text(json.dumps({'stage': stage, 'status': 'passed', 'image_id': 'image', 'source_sha256': 'source', 'run_id': stage, 'user_device': 2, 'logical_device': 2, 'physical_device': 15, 'stream_device': 2, 'readback_matches': True, 'cleanup': {'completion_confirmed': True, 'released': True}}))
        evidence_items[stage] = {'path': str(evidence), 'sha256': hashlib.sha256(evidence.read_bytes()).hexdigest()}
    resources = {'host_memory_bytes': 8 << 30, 'device_observation_limit_bytes': 9 << 30, 'timeout_s': 300}
    config = dict(weights_confirmed=True, model_id=MODEL, revision=REVISION, resource_scope='onerec-single-forward', resource_approved=True, approved_resources=resources.copy(), **resources, atol=.001, rtol=.001, expires_at=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=1)).isoformat(), device=2, image_id='image', source_commit='commit', physical_device=15, probe_source_sha256='source', model_mount_read_only=True, external_watchdog_confirmed=True, torch_version='2', torch_npu_version='2', transformers_version='4', mapping=evidence_items['mapping'], health=evidence_items['health'], model_path=str(model_dir), files=digests)
    assert validate_model_gate(config) is config
    for field, value in [('host_memory_bytes', 4 << 30), ('resource_approved', False), ('revision', 'other'), ('expires_at', '2000-01-01T00:00:00Z')]:
        bad = config.copy(); bad[field] = value
        with pytest.raises(ValueError): validate_model_gate(bad)
    bad = config.copy(); bad['mapping'] = dict(evidence_items['mapping'], sha256='wrong')
    with pytest.raises(ValueError): validate_model_gate(bad)
    (model_dir / 'one.safetensors').unlink()
    with pytest.raises(ValueError): validate_model_gate(config)


def test_gate_must_read_stage_contents_not_merely_matching_file_hash(tmp_path):
    import hashlib
    from test_onerec_inputs_ascend import _validate_stage
    path = tmp_path / 'stage.json'; path.write_text('{"stage":"health","status":"failed"}')
    item = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    config = {'device': 0, 'physical_device': 15, 'image_id': 'image', 'probe_source_sha256': 'source'}
    with pytest.raises(ValueError): _validate_stage(item, 'health', config)


def test_actual_pool_reuses_completed_storage(input_case):
    from ascend_test_adapter import AscendTestBackend, AscendTensorAllocator
    from flagmemory.pool import create_pool, PoolForm, TierSpec, ReusePolicy
    from pooled_model_inputs import lease_input_ids, release_input_ids
    _, runtime, ids, stream = input_case
    class Event:
        def record(self, stream): self.recorded = True
        def query(self): return self.recorded
    runtime.npu.Event = Event
    runtime.npu.current_stream = lambda _: stream
    runtime.npu.mem_get_info = lambda _: (1 << 30, 2 << 30)
    runtime.empty = lambda nbytes, dtype, device: FakeTensor(shape=(nbytes,), dtype=dtype, device='npu', index=2)
    backend = AscendTestBackend(runtime, 2)
    allocator = AscendTensorAllocator(runtime, 2, 16 << 20)
    pool = create_pool(backend, device_index=2, form=PoolForm.TENSOR, base_allocator=allocator, tier_spec=TierSpec(8 << 20, 8 << 20, 2), reuse=ReusePolicy(opportunistic=False))
    sid = pool.register_stream(stream)
    first = lease_input_ids(pool, runtime, ids, stream=stream, stream_id=sid, device=2)
    assert first.tensor.values() == [3, 9, 17]
    release_input_ids(first, pool, backend, stream=stream, timeout_s=1)
    second = lease_input_ids(pool, runtime, ids, stream=stream, stream_id=sid, device=2)
    assert first.storage_id == second.storage_id and first.block_id != second.block_id
    assert allocator.allocate_calls == 1
    release_input_ids(second, pool, backend, stream=stream, timeout_s=1)
    assert pool.destroy_pool(timeout_s=1).completed and allocator.live_bytes == 0


def test_gate_rejects_writable_mount(monkeypatch, tmp_path):
    from test_onerec_inputs_ascend import _check_readonly
    monkeypatch.setattr('test_onerec_inputs_ascend.os.ST_RDONLY', 1, raising=False)
    monkeypatch.setattr('test_onerec_inputs_ascend.os.statvfs', lambda _: SimpleNamespace(f_flag=0), raising=False)
    with pytest.raises(ValueError, match='read.only'): _check_readonly(tmp_path)


@pytest.mark.parametrize('base', [
    FakeTensor(shape=(8,), dtype='uint8', device='npu', index=2),
    FakeTensor(shape=(8192,), dtype='uint8', device='npu', index=1),
    FakeTensor(shape=(8192,), dtype='uint8', device='npu', index=2, offset=1),
])
def test_invalid_backings_retain_lease(input_case, base):
    from pooled_model_inputs import lease_input_ids, InputLeaseError
    pool, runtime, ids, stream = input_case; pool.base = base
    with pytest.raises(InputLeaseError) as err:
        lease_input_ids(pool, runtime, ids, stream=stream, stream_id=3, device=2)
    assert err.value.lease.block_id == 1 and pool.free_calls == 0


def test_copy_failure_returns_lease_to_caller(input_case, monkeypatch):
    from pooled_model_inputs import lease_input_ids, InputLeaseError
    pool, runtime, ids, stream = input_case
    def failed_copy(*args, **kwargs): raise RuntimeError('copy failed after possible submission')
    monkeypatch.setattr(FakeTensor, 'copy_', failed_copy)
    with pytest.raises(InputLeaseError) as err:
        lease_input_ids(pool, runtime, ids, stream=stream, stream_id=3, device=2)
    assert err.value.lease.tensor is not None and not err.value.lease.return_attempted and pool.free_calls == 0


def test_entry_cleanup_reuse_runs_in_inference_mode_and_records():
    from test_onerec_inputs_ascend import _finish_pool
    from pooled_model_inputs import lease_input_ids
    pool, runtime, ids, stream = make_entry_case()
    with runtime.inference_mode():
        lease = lease_input_ids(pool, runtime, ids, stream=stream, stream_id=3, device=2)
    record = {}
    _finish_pool(lease, pool=pool, backend=FakeBackend(), runtime=runtime, ids=ids,
                 stream=stream, sid=3, device=2, successful=True, record=record)
    assert pool.free_calls == 2 and record['reuse']['storage_id'] == 1
    assert record['destroy']['completed'] is True
    assert all(x['free']['status'] == 'completed' and x['drain']['status'] == 'completed' for x in record['returns'])


def test_entry_cleanup_second_copy_failure_is_primary_and_new_lease_returned():
    from test_onerec_inputs_ascend import _finish_pool
    from pooled_model_inputs import lease_input_ids, InputLeaseError
    pool, runtime, ids, stream = make_entry_case(fail_second_copy=True)
    with runtime.inference_mode():
        lease = lease_input_ids(pool, runtime, ids, stream=stream, stream_id=3, device=2)
    record = {}
    with pytest.raises(InputLeaseError, match='construction'):
        _finish_pool(lease, pool=pool, backend=FakeBackend(), runtime=runtime, ids=ids,
                     stream=stream, sid=3, device=2, successful=True, record=record)
    assert pool.free_calls == 2 and record['destroy']['completed'] is True
    assert record['returns'][1]['block_id'] == 2


def test_entry_cleanup_pending_and_destroy_error_preserve_primary_and_report():
    from test_onerec_inputs_ascend import _finish_pool
    from pooled_model_inputs import lease_input_ids
    pool, runtime, ids, stream = make_entry_case(destroy_error=True)
    with runtime.inference_mode():
        lease = lease_input_ids(pool, runtime, ids, stream=stream, stream_id=3, device=2)
    record = {}
    with pytest.raises(TimeoutError):
        _finish_pool(lease, pool=pool, backend=FakeBackend(done=False), runtime=runtime, ids=ids,
                     stream=stream, sid=3, device=2, successful=False, record=record, timeout_s=0)
    assert pool.free_calls == 0 and record['destroy']['leaked'] == [lease.block_id]
    assert record['returns'][0]['completion']['status'] == 'failed'
    assert 'destroy' in record['cleanup_errors']


def make_entry_case(*, fail_second_copy=False, destroy_error=False):
    from contextlib import contextmanager
    pool = FakePool(); mode = {'active': False, 'copies': 0}
    class ModeTensor(FakeTensor):
        def narrow(self, dim, start, length):
            return ModeTensor(buf=self.buf, shape=(length,), dtype=self.dtype, device=self.device.type, index=self.device.index, offset=self.offset+start)
        def view(self, *args):
            dtype = args[0] if len(args) == 1 and isinstance(args[0], str) else self.dtype
            shape = (self.numel() // 8,) if dtype == 'int64' and self.dtype == 'uint8' else args
            return ModeTensor(buf=self.buf, shape=shape, dtype=dtype, device=self.device.type, index=self.device.index, offset=self.offset)
        def copy_(self, source, non_blocking=False):
            if not mode['active']: raise RuntimeError('inference tensor cannot be written outside inference mode')
            mode['copies'] += 1
            if fail_second_copy and mode['copies'] == 2: raise RuntimeError('second copy failed')
            super().copy_(source, non_blocking=non_blocking)
    pool.base = ModeTensor(shape=(8192,), dtype='uint8', device='npu', index=2)
    class DestroyError(RuntimeError):
        def __init__(self): self.report = {'completed': False, 'leaked': [1]}; super().__init__('destroy failed')
    def destroy_pool(**kwargs):
        if destroy_error: raise DestroyError()
        return {'completed': True, 'leaked': [], 'pending': [], 'quarantined': []}
    pool.destroy_pool = destroy_pool
    @contextmanager
    def inference_mode():
        old = mode['active']; mode['active'] = True
        try: yield
        finally: mode['active'] = old
    runtime = SimpleNamespace(int64='int64', uint8='uint8', inference_mode=inference_mode,
        npu=SimpleNamespace(device=lambda _: nullcontext(), stream=lambda _: nullcontext()))
    return pool, runtime, FakeTensor([3, 9, 17]), SimpleNamespace(device=SimpleNamespace(type='npu', index=2))


def test_entry_second_copy_and_destroy_error_keep_original_error():
    from test_onerec_inputs_ascend import _finish_pool
    from pooled_model_inputs import lease_input_ids, InputLeaseError
    pool, runtime, ids, stream = make_entry_case(fail_second_copy=True, destroy_error=True)
    with runtime.inference_mode():
        lease = lease_input_ids(pool, runtime, ids, stream=stream, stream_id=3, device=2)
    record = {}
    with pytest.raises(InputLeaseError) as failure:
        _finish_pool(lease, pool=pool, backend=FakeBackend(), runtime=runtime, ids=ids,
                     stream=stream, sid=3, device=2, successful=True, record=record)
    assert 'second copy failed' in repr(failure.value.__cause__)
    assert record['destroy']['completed'] is False and 'destroy' in record['cleanup_errors']
    assert pool.free_calls == 2


@pytest.mark.parametrize('device', [True, -1, 1.5])
def test_explicit_device_rejects_before_event(device):
    from pooled_model_inputs import confirm_completion
    class B:
        def create_event(self, **kw): pytest.fail('event side effect')
    with pytest.raises(ValueError): confirm_completion(B(), stream=object(), timeout_s=0, device_index=device)


def test_unified_stream_explicit_device_and_context(input_case):
    from pooled_model_inputs import lease_input_ids, release_input_ids
    pool, runtime, ids, _ = input_case
    class Unified:
        def __getattr__(self, name): raise AssertionError('nonpublic stream access: '+name)
    stream = Unified(); entered = []
    from contextlib import contextmanager
    @contextmanager
    def context():
        entered.append(True); yield
    runtime.npu = None
    lease = lease_input_ids(pool, runtime, ids, stream=stream, stream_id=3, device=2, stream_context=context)
    class B(FakeBackend):
        def create_event(self, device_index):
            assert device_index == 2
            return object()
    release_input_ids(lease, pool, B(), stream=stream, timeout_s=1, device_index=2)
    assert entered == [True] and lease.closed
