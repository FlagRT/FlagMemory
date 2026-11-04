"""Test-only, explicitly bound assembly over frozen public runtime objects.

Importing this module never imports the device runtime or a vendor framework.
"""
from __future__ import annotations
import hashlib
import importlib
import json
import logging
from pathlib import Path, PurePosixPath
import sys
import time
from contextlib import contextmanager
from flagmemory.adapters.ascend import AscendProfile, make_ascend_profile

RUNTIME_COMMIT = '719de8cbaea75691b38ad87d6706a949e045b892'
RUNTIME_TREE = '3a456b091d11fd7b30bb573be28ec013be80fa1f'
ADAPTER_KIND = 'device-context-ascend-profile'
_DYNAMIC = ('errors', 'recovery', 'device_state', 'npu_events')
_CUSTODY = []


def _git_object(kind, raw):
    return hashlib.sha1(kind.encode()+b' '+str(len(raw)).encode()+b'\0'+raw).hexdigest()


def _tree(files):
    children = {}
    for name, blob in files.items():
        head, sep, tail = name.partition('/')
        if sep:
            children.setdefault(head, {})[tail] = blob
        else:
            children[head] = blob
    entries = []
    for name, value in children.items():
        isdir = isinstance(value, dict)
        entries.append((name + ('/' if isdir else ''),
            (b'40000' if isdir else b'100644')+b' '+name.encode()+b'\0'+bytes.fromhex(_tree(value) if isdir else value)))
    return _git_object('tree', b''.join(raw for _,raw in sorted(entries)))


def validate_runtime_export(root: Path, manifest: dict) -> None:
    root = Path(root)
    if root.is_symlink() or not root.is_dir() or not isinstance(manifest, dict):
        raise ValueError('independent runtime export required')
    package = root/'runtime'
    files = manifest.get('files')
    if (manifest.get('source_repository') != 'https://github.com/FlagRT/runtime-team'
        or manifest.get('source_commit') != RUNTIME_COMMIT or manifest.get('runtime_tree') != RUNTIME_TREE
        or not isinstance(files,list) or not files or manifest.get('count') != len(files)):
        raise ValueError('frozen public runtime identity required')
    actual = set()
    for p in root.rglob('*'):
        if p.is_symlink(): raise ValueError('runtime export links refused')
        if p.is_file(): actual.add(p.relative_to(root).as_posix())
    expected = {'manifest.json'} if (root/'manifest.json').is_file() else set()
    blobs = {}; total = 0
    for item in files:
        name = item.get('path')
        if (not isinstance(name,str) or '\\' in name or ':' in name or
            PurePosixPath(name).is_absolute() or any(p in ('','.', '..') for p in name.split('/'))
            or name in blobs or item.get('mode') != '100644'):
            raise ValueError('invalid or duplicate public runtime path')
        file = package/name
        if not file.is_file(): raise ValueError('missing runtime file')
        raw=file.read_bytes(); blob=_git_object('blob',raw)
        if (type(item.get('bytes')) is not int or item['bytes'] != len(raw)
            or item.get('git_blob') != blob or item.get('sha256') != hashlib.sha256(raw).hexdigest()):
            raise ValueError('runtime content hash or Git blob mismatch')
        expected.add('runtime/'+name); blobs[name]=blob; total+=len(raw)
    if actual != expected or manifest.get('total_bytes') != total or _tree(blobs) != RUNTIME_TREE:
        raise ValueError('complete public runtime tree required')


def verify_runtime_origins(root):
    package=(Path(root)/'runtime').resolve()
    origins={}
    for name, module in tuple(sys.modules.items()):
        if name == 'runtime' or name.startswith('runtime.') or name in _DYNAMIC:
            file=getattr(module,'__file__',None)
            if not file: raise ValueError('runtime module has no verifiable origin: '+name)
            file=Path(file).resolve()
            expected=package/'conformance'/(name+'.py') if name in _DYNAMIC else None
            if (expected is not None and file != expected) or not file.is_relative_to(package):
                raise ValueError('foreign runtime module: '+name)
            if name == 'runtime' and file != package/'__init__.py':
                raise ValueError('foreign runtime package')
            origins[name]=str(file)
    return origins


def load_runtime_types(root: Path, manifest: dict) -> tuple[type,type,type]:
    validate_runtime_export(root,manifest)
    verify_runtime_origins(root)
    sys.path.insert(0,str(Path(root).resolve()))
    # registry.register evaluates backend.info only when DEBUG is enabled.
    # During import keep discovery metadata-only even under an inherited DEBUG setup.
    root_logger=logging.getLogger(); previous_level=root_logger.level
    previous_disable=logging.root.manager.disable
    root_logger.setLevel(logging.WARNING); logging.disable(max(previous_disable,logging.DEBUG))
    try:
        runtime=importlib.import_module('runtime')
        backend=importlib.import_module('runtime.backends.ascend.backend')
    finally:
        root_logger.setLevel(previous_level); logging.disable(previous_disable)
    verify_runtime_origins(root)
    return backend.AscendBackend,runtime.Stream,runtime.Event


class ObservedBridge:
    """Transparent delegation; event objects and query results stay unchanged."""
    def __init__(self, bridge):
        self.delegate=bridge; self.events=[]; self.records=[]; self.queries=[]
    def __getattr__(self,name): return getattr(self.delegate,name)
    def create_event(self,device_index=0):
        event=self.delegate.create_event(device_index); self.events.append(event); return event
    def record_event(self,event,stream):
        self.delegate.record_event(event,stream); self.records.append((event,stream))
    def query_event(self,event):
        result=self.delegate.query_event(event); self.queries.append((event,result)); return result


class ObservedAllocator:
    def __init__(self, allocator):
        self.delegate=allocator; self.allocate_calls=0; self.release_calls=0; self.allocated_bytes=0
        self.live=[]; self.release_attempts=[]
    @property
    def live_bytes(self): return sum(n for _,n in self.live)
    def allocate(self,nbytes,device_index=0,*,stream):
        tensor=self.delegate.allocate(nbytes,device_index,stream=stream)
        self.live.append((tensor,nbytes)); self.allocate_calls+=1; self.allocated_bytes+=nbytes
        return tensor
    def release(self,tensor):
        if any(old is tensor for old in self.release_attempts): raise ValueError('release must not be retried')
        self.release_attempts.append(tensor)
        self.delegate.release(tensor)
        self.live=[(old,n) for old,n in self.live if old is not tensor]; self.release_calls+=1
        # Successful release must drop the observation alias too.
        self.release_attempts=[old for old in self.release_attempts if old is not tensor]


def make_case_profile(torch_module, *, backend, stream_type, event_type, device_index: int, byte_limit: int) -> AscendProfile:
    profile=make_ascend_profile(torch_module=torch_module,backend=backend,stream_type=stream_type,
        event_type=event_type,device_index=device_index,byte_limit=byte_limit)
    return AscendProfile(ObservedBridge(profile.bridge),ObservedAllocator(profile.allocator),profile.new_stream)


def wait_event(bridge,event,timeout_s=10):
    deadline=time.monotonic()+timeout_s
    while not bridge.query_event(event):
        if time.monotonic() >= deadline: raise TimeoutError('bridge event query deadline')
        time.sleep(.001)


def load_manifest(path, expected_sha):
    file=Path(path)
    if not file.is_absolute() or file.is_symlink() or not file.is_file() or file.stat().st_size > 1<<20:
        raise ValueError('absolute bounded device source manifest required')
    if file.name != 'manifest.json' or {p.name for p in file.parent.iterdir()} != {'manifest.json','runtime'}:
        raise ValueError('staged device source must contain only manifest.json and runtime')
    raw=file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha: raise ValueError('device source manifest hash mismatch')
    manifest=json.loads(raw)
    validate_runtime_export(file.parent,manifest)
    return manifest


@contextmanager
def managed_pool(pool, profile):
    primary=None; report=None; cleanup=None
    try:
        yield pool
    except BaseException as exc:
        primary=exc
        raise
    finally:
        try:
            report=pool.destroy_pool(timeout_s=10)
            if not report.completed or report.pending or report.quarantined or report.leaked:
                raise RuntimeError('bridge pool cleanup incomplete')
        except BaseException as exc:
            cleanup=exc; report=getattr(exc,'report',report)
            _CUSTODY.append((pool,profile))
        print('bridge destroy:', report, 'cleanup_error:', repr(cleanup))
        if primary is not None:
            primary.destroy_report=report
            if cleanup is not None:
                primary.cleanup_error=cleanup
                # A missing pending observation cannot hide a failed close.
                import pytest
                if isinstance(primary,pytest.skip.Exception): raise cleanup from primary
        elif cleanup is not None:
            cleanup.destroy_report=report
            raise cleanup
