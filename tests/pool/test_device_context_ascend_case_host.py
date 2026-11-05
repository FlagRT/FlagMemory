"""CPU contracts for the separately gated public device bridge cases."""
import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "interfaces"))


def case_module():
    spec = importlib.util.find_spec('device_context_ascend_case')
    assert spec is not None, 'new bridge assembly module is required'
    return importlib.import_module('device_context_ascend_case')


def test_missing_source_rejected_before_import(tmp_path):
    module = case_module()
    before = set(sys.modules)
    with pytest.raises(ValueError): module.validate_runtime_export(tmp_path, {'files': []})
    assert not (set(sys.modules) - before) & {'runtime', 'torch', 'torch_npu'}


@pytest.fixture
def export_case(tmp_path, monkeypatch):
    m = case_module()
    root = tmp_path / 'device-runtime'; package = root / 'runtime'; package.mkdir(parents=True)
    raw = b'# host fixture, never a hardware implementation\n'
    (package / '__init__.py').write_bytes(raw)
    blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    tree = hashlib.sha1(b'tree 39\0'+b'100644 __init__.py\0'+bytes.fromhex(blob)).hexdigest()
    monkeypatch.setattr(m, 'RUNTIME_TREE', tree)
    manifest = dict(source_repository='https://github.com/FlagRT/runtime-team',
        source_commit='719de8cbaea75691b38ad87d6706a949e045b892', runtime_tree=tree,
        count=1, total_bytes=len(raw), files=[dict(path='__init__.py', bytes=len(raw),
        git_blob=blob, sha256=hashlib.sha256(raw).hexdigest(), mode='100644')])
    return root, manifest


def test_valid_export_bytes_and_git_tree(export_case):
    case_module().validate_runtime_export(*export_case)


@pytest.mark.parametrize('fault', ['changed', 'missing', 'extra', 'blob', 'commit', 'escape', 'duplicate'])
def test_export_rejects_changed_missing_extra_or_forged_content(export_case, fault):
    root, manifest = export_case
    if fault == 'changed': (root/'runtime/__init__.py').write_bytes(b'changed')
    if fault == 'missing': (root/'runtime/__init__.py').unlink()
    if fault == 'extra': (root/'runtime/extra.py').write_bytes(b'')
    if fault == 'blob': manifest['files'][0]['git_blob'] = '0'*40
    if fault == 'commit': manifest['source_commit'] = '0'*40
    if fault == 'escape': manifest['files'][0]['path'] = '../outside.py'
    if fault == 'duplicate': manifest['files'].append(manifest['files'][0])
    with pytest.raises(ValueError): case_module().validate_runtime_export(root, manifest)


@pytest.mark.parametrize('name', ['runtime', 'errors', 'recovery', 'device_state', 'npu_events'])
def test_foreign_import_rejected_before_runtime_load(export_case, monkeypatch, name):
    root, manifest = export_case
    monkeypatch.setitem(sys.modules, name, SimpleNamespace(__file__='/foreign/'+name+'.py'))
    with pytest.raises(ValueError): case_module().load_runtime_types(root, manifest)


def test_profile_refuses_uninitialized_and_wrong_current():
    from test_ascend_profile import setup
    m = case_module(); b,t,_ = setup()
    from device_context_fakes import UnifiedStream, UnifiedEvent
    t.npu.initialized = False
    with pytest.raises(Exception): m.make_case_profile(t, backend=b, stream_type=UnifiedStream, event_type=UnifiedEvent, device_index=2, byte_limit=64)
    assert t.allocs == 0 and t.npu.calls == 0
    t.npu.initialized = True; t.npu.current = 0
    with pytest.raises(Exception): m.make_case_profile(t, backend=b, stream_type=UnifiedStream, event_type=UnifiedEvent, device_index=2, byte_limit=64)
    assert t.allocs == 0


def test_profile_budget_and_opaque_event_observation():
    from test_ascend_profile import setup, Raw
    from device_context_fakes import UnifiedStream, UnifiedEvent
    m=case_module(); b,t,_=setup()
    p=m.make_case_profile(t,backend=b,stream_type=UnifiedStream,event_type=UnifiedEvent,device_index=2,byte_limit=32)
    s=p.new_stream(); x=p.allocator.allocate(20,2,stream=s); p.allocator.release(x)
    with pytest.raises(MemoryError): p.allocator.allocate(13,2,stream=s)
    assert p.allocator.allocate_calls == 1 and p.allocator.release_calls == 1 and p.allocator.live_bytes == 0
    with pytest.raises(Exception): p.bridge.validate_stream_device(UnifiedStream(b,Raw()),2)
    event=p.bridge.create_event(2); p.bridge.record_event(event,s)
    assert p.bridge.events[-1] is event and p.bridge.records[-1] == (event,s)
    assert p.bridge.query_event(event) is False
    assert p.bridge.queries[-1] == (event,False)


@pytest.mark.parametrize('case_name', ['test_actual_tensor_and_cross_stream_dependency','test_completed_reuse_rejects_old_lease','test_pending_borrower_protection_when_observed'])
def test_new_realcard_cases_use_real_pool_and_profile(case_name, monkeypatch):
    m=case_module()
    spec=importlib.util.find_spec('test_pool_device_context_ascend_realcard')
    assert spec is not None, 'separate bridge lifecycle entry is required'
    real=importlib.import_module('test_pool_device_context_ascend_realcard')
    monkeypatch.setattr(real,'SIZE',16); monkeypatch.setattr(real,'READ_BYTES',4)
    t,b,event_type,stream_type=cpu_runtime()
    p=m.make_case_profile(t,backend=b,stream_type=stream_type,event_type=event_type,device_index=2,byte_limit=64)
    from flagmemory.pool import PoolForm,TierSpec,ReusePolicy,create_pool
    pool=create_pool(p.bridge,device_index=2,form=PoolForm.TENSOR,base_allocator=p.allocator,tier_spec=TierSpec(16,16,2),reuse=ReusePolicy(opportunistic=False))
    getattr(real,case_name)((t,2,b,event_type,p,pool))
    assert p.allocator.live_bytes == 0 and p.allocator.release_calls == p.allocator.allocate_calls
    assert p.allocator.allocate_calls == (2 if 'pending' in case_name else 1)


def cpu_runtime():
    from contextlib import contextmanager
    from test_ascend_profile import Backend, Torch, Raw
    from device_context_fakes import UnifiedStream, UnifiedEvent
    class B(Backend):
        def __init__(self): super().__init__(); self.current=self.default_native; self.ready={}; self.fast=False
        @contextmanager
        def stream_context(self, raw):
            old=self.current; self.current=raw; self.active=True
            try: yield
            finally: self.current=old; self.active=False
    class E(UnifiedEvent):
        def query(self):
            n=getattr(self.native,'host_queries',0)+1; self.native.host_queries=n
            self.native.completed=self.backend.fast or n>=4
            return super().query()
        def wait(self, stream):
            assert self.native.recorded_stream is not None
            self.backend.ready.setdefault(stream.native,set()).add(self.native.recorded_stream)
    class T(bytearray):
        def __init__(self,n,b):
            super().__init__(n); self.backend=b; self.owner=b.current; self.device=SimpleNamespace(type='npu',index=2); self.dtype='u8'
        def numel(self): return len(self)
        def element_size(self): return 1
        def untyped_storage(self): return self
        def data_ptr(self): return id(self)
        def fill_(self,value):
            assert self.backend.current is self.owner or self.owner in self.backend.ready.get(self.backend.current,set()), 'missing producer dependency'
            self[:] = bytes([value])*len(self); return self
        def __getitem__(self,key):
            if isinstance(key,slice): return Slice(self,key)
            return super().__getitem__(key)
    class Slice:
        def __init__(self,base,key): self.base=base; self.key=key
        def fill_(self,value): self.base[self.key]=bytes([value])*len(bytearray.__getitem__(self.base,self.key)); return self
        def add_(self,value):
            assert self.base.owner in self.base.backend.ready.get(self.base.backend.current,set()), 'missing producer dependency'
            self.base[self.key]=bytes(x+value for x in bytearray.__getitem__(self.base,self.key)); return self
        def cpu(self): return self
        def tolist(self): return list(bytearray.__getitem__(self.base,self.key))
    b=B(); t=Torch(b); t.uint8='u8'
    def empty(n,*,dtype,device):
        assert b.active and device=='npu:2'; t.allocs+=1; return T(n,b)
    t.empty=empty
    return t,b,E,UnifiedStream


def test_cleanup_preserves_primary_and_destroy_failure():
    m=case_module(); error=RuntimeError('body'); close=RuntimeError('close'); report=object(); close.report=report
    class Pool:
        def destroy_pool(self,**kwargs): raise close
    with pytest.raises(RuntimeError) as caught:
        with m.managed_pool(Pool(),object()): raise error
    assert caught.value is error and error.destroy_report is report and error.cleanup_error is close

def test_manifest_gate_hash_before_any_import(export_case, tmp_path):
    root, manifest=export_case
    import json
    file=root/'manifest.json'; file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): case_module().load_manifest(file,'0'*64)
    good=hashlib.sha256(file.read_bytes()).hexdigest()
    assert case_module().load_manifest(file,good)==manifest


def test_new_pending_nonobservation_is_skipped_and_drained(monkeypatch):
    m=case_module(); real=importlib.import_module('test_pool_device_context_ascend_realcard')
    monkeypatch.setattr(real,'SIZE',16); monkeypatch.setattr(real,'READ_BYTES',4)
    t,b,event_type,stream_type=cpu_runtime(); b.fast=True
    p=m.make_case_profile(t,backend=b,stream_type=stream_type,event_type=event_type,device_index=2,byte_limit=64)
    from flagmemory.pool import PoolForm,TierSpec,ReusePolicy,create_pool
    pool=create_pool(p.bridge,device_index=2,form=PoolForm.TENSOR,base_allocator=p.allocator,tier_spec=TierSpec(16,16,2),reuse=ReusePolicy(opportunistic=False))
    with pytest.raises(pytest.skip.Exception,match='uncovered') as caught:
        real.test_pending_borrower_protection_when_observed((t,2,b,event_type,p,pool))
    assert p.allocator.allocate_calls == 1 and caught.value.destroy_report.completed and p.allocator.live_bytes == 0


def test_cross_stream_case_detects_missing_public_wait(monkeypatch):
    m=case_module(); real=importlib.import_module('test_pool_device_context_ascend_realcard')
    monkeypatch.setattr(real,'SIZE',16); monkeypatch.setattr(real,'READ_BYTES',4)
    t,b,event_type,stream_type=cpu_runtime()
    monkeypatch.setattr(event_type,'wait',lambda *args:None)
    p=m.make_case_profile(t,backend=b,stream_type=stream_type,event_type=event_type,device_index=2,byte_limit=64)
    from flagmemory.pool import PoolForm,TierSpec,ReusePolicy,create_pool
    pool=create_pool(p.bridge,device_index=2,form=PoolForm.TENSOR,base_allocator=p.allocator,tier_spec=TierSpec(16,16,2),reuse=ReusePolicy(opportunistic=False))
    with pytest.raises(AssertionError,match='missing producer dependency'):
        real.test_actual_tensor_and_cross_stream_dependency((t,2,b,event_type,p,pool))
    # This is a host-only fault; explicitly close after the synchronous fake has completed.
    from flagmemory.pool import LeakDetected
    with pytest.raises(LeakDetected) as caught: pool.destroy_pool(wait_for_drain=False)
    for view in caught.value.report.leaked: pool.free_block(view.block_id)
    assert not pool.drain_stream(timeout_s=1).recoverable_timeout
    assert pool.destroy_pool().completed


def test_source_import_disables_debug_hardware_discovery(export_case,monkeypatch):
    import logging
    root,manifest=export_case; m=case_module(); root_logger=logging.getLogger(); previous=root_logger.level
    root_logger.setLevel(logging.DEBUG)
    class ProbeAttempt(Exception): pass
    def import_module(name):
        if logging.getLogger('runtime.backends.registry').isEnabledFor(logging.DEBUG):
            raise ProbeAttempt('debug registration would evaluate backend.info')
        return SimpleNamespace(Stream=object,Event=object,AscendBackend=object)
    monkeypatch.setattr(m.importlib,'import_module',import_module)
    try: assert m.load_runtime_types(root,manifest)==(object,object,object)
    finally: root_logger.setLevel(previous)


def test_export_link_is_refused(export_case,monkeypatch):
    root,manifest=export_case; original=Path.is_symlink
    monkeypatch.setattr(Path,'is_symlink',lambda path: path==root/'runtime/__init__.py' or original(path))
    with pytest.raises(ValueError,match='links'):case_module().validate_runtime_export(root,manifest)


def test_cleanup_does_not_replace_original_cause_or_context():
    m=case_module(); root=ValueError('root'); original=RuntimeError('body'); close=RuntimeError('destroy')
    class Pool:
        def destroy_pool(self,**kwargs):raise close
    with pytest.raises(RuntimeError) as caught:
        with m.managed_pool(Pool(),object()):raise original from root
    assert caught.value is original and original.__cause__ is root and original.__context__ is None
    assert original.cleanup_error is close


def test_case_body_exception_receives_its_destroy_report(monkeypatch):
    m=case_module(); real=importlib.import_module('test_pool_device_context_ascend_realcard')
    monkeypatch.setattr(real,'SIZE',16); monkeypatch.setattr(real,'READ_BYTES',4)
    t,b,event_type,stream_type=cpu_runtime()
    monkeypatch.setattr(event_type,'wait',lambda *args:None)
    p=m.make_case_profile(t,backend=b,stream_type=stream_type,event_type=event_type,device_index=2,byte_limit=64)
    from flagmemory.pool import PoolForm,TierSpec,ReusePolicy,create_pool
    pool=create_pool(p.bridge,device_index=2,form=PoolForm.TENSOR,base_allocator=p.allocator,tier_spec=TierSpec(16,16,2),reuse=ReusePolicy(opportunistic=False))
    with pytest.raises(AssertionError) as caught:
        real.test_actual_tensor_and_cross_stream_dependency((t,2,b,event_type,p,pool))
    assert caught.value.destroy_report.leaked and caught.value.cleanup_error is not None
    for view in caught.value.destroy_report.leaked:pool.free_block(view.block_id)
    assert not pool.drain_stream(timeout_s=1).recoverable_timeout
    assert pool.destroy_pool().completed


def test_skip_with_cleanup_failure_is_failed_not_uncovered():
    m=case_module(); close=RuntimeError('destroy failed')
    class Pool:
        def destroy_pool(self,**kwargs):raise close
    caught=None
    try:
        with m.managed_pool(Pool(),object()):pytest.skip('pending not observed')
    except BaseException as exc:caught=exc
    assert caught is close and isinstance(close.__cause__,pytest.skip.Exception)


@pytest.mark.parametrize('sibling',['torch.py','sitecustomize.py','torch','empty'])
def test_staged_source_parent_cannot_shadow_vendor_or_hide_extra_directory(export_case,sibling):
    import json
    root,manifest=export_case; file=root/'manifest.json'; file.write_text(json.dumps(manifest))
    path=root/sibling
    if sibling.endswith('.py'):path.write_text("raise AssertionError('must not import this sibling')")
    else:path.mkdir()
    digest=hashlib.sha256(file.read_bytes()).hexdigest()
    with pytest.raises(ValueError):case_module().load_manifest(file,digest)


@pytest.fixture
def real_namespace(tmp_path, monkeypatch):
    root = tmp_path / 'frozen'
    package = root / 'runtime'
    (package / 'conformance').mkdir(parents=True)
    (package / '__init__.py').write_text('')
    spec = importlib.util.spec_from_file_location('runtime', package / '__init__.py')
    parent = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, 'runtime', parent)
    monkeypatch.setitem(sys.modules, 'runtime.conformance', None)
    sys.modules.pop('runtime.conformance')
    namespace = importlib.import_module('runtime.conformance')
    yield root, namespace
    sys.modules.pop('runtime.conformance', None)


def test_real_conformance_namespace_origin(real_namespace):
    root, namespace = real_namespace
    assert namespace.__file__ is None
    origins = case_module().verify_runtime_origins(root)
    assert origins['runtime.conformance'] == {
        'kind': 'namespace', 'path': str((root/'runtime/conformance').resolve())}


@pytest.mark.parametrize('fault', ['external', 'mixed', 'empty', 'spec_path',
    'name', 'origin', 'loader', 'module_loader', 'missing_spec', 'ordinary', 'unknown'])
def test_namespace_origin_rejects_pollution(real_namespace, monkeypatch, fault):
    root, namespace = real_namespace
    target = str((root/'runtime/conformance').resolve())
    if fault == 'external': namespace.__path__ = ['/foreign/conformance']
    if fault == 'mixed': namespace.__path__ = [target, '/foreign/conformance']
    if fault == 'empty': namespace.__path__ = []
    if fault == 'spec_path': namespace.__spec__.submodule_search_locations = ['/foreign']
    if fault == 'name': namespace.__spec__.name = 'other'
    if fault == 'origin': namespace.__spec__.origin = 'forged.py'
    if fault == 'loader': namespace.__spec__.loader = object()
    if fault == 'module_loader': namespace.__loader__ = object()
    if fault == 'missing_spec': namespace.__spec__ = None
    if fault == 'ordinary': namespace.__file__ = target + '/fake.py'
    if fault == 'unknown':
        monkeypatch.setitem(sys.modules, 'runtime.unknown', SimpleNamespace(__file__=None))
    with pytest.raises(ValueError): case_module().verify_runtime_origins(root)
