"""CPU checks for the bounded Ascend preparation tools."""
import importlib.util
from pathlib import Path
import sys
import json
import hashlib
import pytest

ROOT = Path(__file__).resolve().parents[1]

def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

probe = load_tool("ascend_probe")

class FakeAcl:
    physical_argument_semantics = "identity-only-user-equals-logic"
    def __init__(self, logic=0, physical=15, count=1, fault=None):
        self.logic, self.physical, self.count = logic, physical, count
        self.calls = []
        self.fault = fault
    def rc(self, name): return 1 if self.fault == name else 0
    def init(self): self.calls.append(("init",)); return self.rc("init")
    def finalize(self): self.calls.append(("finalize",)); return self.rc("finalize")
    def get_count(self): self.calls.append(("count",)); return self.rc("count"), self.count
    def get_logic(self, user): self.calls.append(("logic",user)); return self.rc("logic"), self.logic
    def get_physical(self, user): self.calls.append(("physical",user)); return self.rc("physical"), self.physical

def test_mapping_checks_identity_and_finalizes():
    acl = FakeAcl()
    result = probe.map_device(acl, expected_physical=15, environment={})
    assert (result["user_device"], result["logical_device"], result["physical_device"]) == (0,0,15)
    assert result["status"] == "passed" and acl.calls[-1] == ("finalize",)

@pytest.mark.parametrize("kwargs", [{"logic":4},{"count":0},{"count":2},{"physical":14},{"physical":-1}])
def test_ambiguous_or_wrong_mapping_refused(kwargs):
    acl = FakeAcl(**kwargs)
    with pytest.raises(probe.ProbeRefused):
        probe.map_device(acl, expected_physical=15, environment={})
    assert acl.calls[-1] == ("finalize",)
    if kwargs.get("logic") == 4: assert ("physical",0) not in acl.calls

@pytest.mark.parametrize("fault", ["init","count","logic","physical","finalize"])
def test_mapping_errors_preserve_codes(fault):
    acl = FakeAcl(fault=fault)
    with pytest.raises(probe.ProbeFailure) as info:
        probe.map_device(acl, expected_physical=15, environment={})
    assert info.value.evidence["raw_return_codes"][fault] == 1
    assert (("finalize",) in acl.calls) == (fault != "init")

@pytest.mark.parametrize("key", ["ASCEND_RT_VISIBLE_DEVICES","ASCEND_VISIBLE_DEVICES","NPU_VISIBLE_DEVICES"])
def test_visibility_reordering_refused_before_init(key):
    acl = FakeAcl()
    with pytest.raises(probe.ProbeRefused):
        probe.map_device(acl, expected_physical=15, environment={key:"15"})
    assert not acl.calls

def test_missing_evidence_cannot_load_library(tmp_path):
    called = []
    with pytest.raises(probe.ProbeRefused):
        probe.load_reviewed_acl(str(tmp_path/"missing.json"),cdll_factory=called.append)
    assert called == []

def abi_case(tmp_path):
    library=tmp_path/"lib.so"; library.write_bytes(b"host fake, never loaded")
    header=tmp_path/"acl.h"; header.write_text("host fake header")
    digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    data={"schema":1,"image_id":probe.IMAGE_ID,"library_realpath":str(library.resolve()),
          "library_sha256":digest(library),"header_sha256s":{str(header.resolve()):digest(header)},
          "signatures":dict(probe.SIGNATURES),"init_required":True,"init_argument":None,
          "physical_argument_semantics":"identity-only-user-equals-logic","reviewer":"CPU fixture only",
          "reviewed_at":"2026-11-04T00:00:00+00:00","review_status":"accepted",
          "exports":list(probe.SIGNATURES)}
    path=tmp_path/"reviewed.json"; path.write_text(json.dumps(data))
    return path,data

@pytest.mark.parametrize("mutation", ["unreviewed","hash","signature","image","missing_symbol"])
def test_invalid_review_refuses_before_cdll(tmp_path,mutation):
    path,data=abi_case(tmp_path)
    if mutation=="unreviewed": data["review_status"]="unreviewed"
    if mutation=="hash": data["library_sha256"]="0"*64
    if mutation=="signature": data["signatures"]["aclInit"]="guessed"
    if mutation=="image": data["image_id"]="sha256:"+"0"*64
    if mutation=="missing_symbol": data["exports"].pop()
    path.write_text(json.dumps(data)); called=[]
    with pytest.raises(probe.ProbeRefused):
        probe.load_reviewed_acl(str(path),cdll_factory=called.append)
    assert not called

from contextlib import contextmanager
from types import SimpleNamespace

class Clock:
    def __init__(self): self.value=0
    def __call__(self): self.value += 0.05; return self.value

class HealthRuntime:
    uint8=object()
    def __init__(self,fault=None):
        self.npu=self; self.fault=fault; self.allocate_calls=0; self.current=None; self.index=9
    @contextmanager
    def device(self,index):
        old=self.index; self.index=index
        try: yield
        finally: self.index=old
    @contextmanager
    def stream(self,stream):
        old=self.current; self.current=stream
        try: yield
        finally: self.current=old
    def current_device(self): return self.index
    def set_device(self,index): self.index=index
    def Stream(self,device):
        return SimpleNamespace(device=SimpleNamespace(type="npu",index=9 if self.fault=="device" else device))
    def current_stream(self,index): return self.current
    def empty(self,n,*,dtype,device):
        self.allocate_calls+=1
        if self.fault=="allocate": raise RuntimeError("allocation failed")
        assert self.index==2 and self.current.device.index==2
        return HealthTensor(self,n)
    def Event(self): return HealthEvent(self)

class HealthTensor:
    def __init__(self,rt,n): self.rt=rt; self.data=[0]*n; self.device=SimpleNamespace(type="npu",index=2); self.dtype=rt.uint8
    def numel(self): return len(self.data)
    def fill_(self,value): self.data[:]=[value]*len(self.data)
    def cpu(self): return self
    def tolist(self): return [0]*len(self.data) if self.rt.fault=="readback" else self.data

class HealthEvent:
    def __init__(self,rt): self.rt=rt
    def record(self,stream):
        if self.rt.fault=="record": raise RuntimeError("record failed")
    def query(self):
        if self.rt.fault=="query": raise RuntimeError("query failed")
        return self.rt.fault!="pending"

@pytest.mark.parametrize("limit", [0,4095,True,(16<<20)+1])
def test_health_bounds_rejected_before_allocate(limit):
    rt=HealthRuntime()
    with pytest.raises(probe.ProbeRefused):
        probe.health(rt,device=2,byte_limit=limit,timeout_s=1,clock=Clock())
    assert rt.allocate_calls==0

def test_health_readback_and_release():
    rt=HealthRuntime()
    result=probe.health(rt,device=2,byte_limit=16<<20,timeout_s=1,clock=Clock())
    assert result["status"]=="passed" and result["readback_matches"]
    assert result["allocated_bytes"]==4096 and result["release_calls"]==1
    assert result["cleanup"]["completion_confirmed"] and result["cleanup"]["released"]

@pytest.mark.parametrize("fault", ["device","allocate","record","query","pending","readback"])
def test_health_failure_never_claims_success_or_releases_unknown_work(fault):
    result=probe.health(HealthRuntime(fault),device=2,byte_limit=4096,timeout_s=.5,clock=Clock())
    assert result["status"]=="failed"
    assert result["release_calls"]==(1 if fault=="readback" else 0)
    if fault in ("record","query","pending"):
        assert not result["cleanup"]["completion_confirmed"] and not result["cleanup"]["released"]

def test_health_failed_release_is_not_retried(monkeypatch):
    adapter=probe._adapter()
    calls=[]
    def release(self,ref):
        calls.append(ref)
        raise RuntimeError("release failed")
    monkeypatch.setattr(adapter.AscendTensorAllocator,"release",release)
    monkeypatch.setattr(probe,"_adapter",lambda:adapter)
    result=probe.health(HealthRuntime(),device=2,byte_limit=4096,timeout_s=1,clock=Clock())
    assert result["status"]=="failed" and not result["cleanup"]["released"]
    assert len(calls)==1

def test_accepted_review_binds_exact_signatures_and_uses_pointer_outputs(tmp_path):
    path,data=abi_case(tmp_path)
    class Function:
        def __init__(self,name): self.name=name
        def __call__(self,*args):
            if self.name=="aclrtGetDeviceCount": args[0]._obj.value=1
            if self.name=="aclrtGetLogicDevIdByUserDevId": args[1]._obj.value=0
            if self.name=="aclrtGetPhyDevIdByLogicDevId": args[1]._obj.value=15
            return 0
    library=SimpleNamespace(**{name:Function(name) for name in probe.SIGNATURES})
    called=[]
    def factory(value): called.append(value); return library
    acl=probe.load_reviewed_acl(str(path),cdll_factory=factory)
    result=probe.map_device(acl,expected_physical=15,environment={})
    assert result["physical_device"]==15 and called==[data["library_realpath"]]
    assert library.aclInit.argtypes==[probe.ctypes.c_char_p]
    assert library.aclrtGetDeviceCount.restype is probe.ctypes.c_int32

def test_static_collection_never_loads_a_library_or_marks_reviewed(tmp_path,monkeypatch):
    library=tmp_path/"lib.so"; library.write_bytes(b"fixture")
    header=tmp_path/"acl.h"; header.write_text("typedef int aclError;\naclError aclInit(const char*);\n")
    monkeypatch.setattr(probe.shutil,"which",lambda name:"/fixture/readelf")
    monkeypatch.setattr(probe,"bounded_symbols",lambda argv:"\n".join("1 0 1 FUNC GLOBAL DEFAULT 1 "+n for n in probe.SIGNATURES))
    monkeypatch.setattr(probe.ctypes,"CDLL",lambda *args:pytest.fail("static collection loaded ACL"))
    result=probe.collect_abi_evidence([str(header)],str(library),str(tmp_path/"new"))
    assert result["review_status"]=="unreviewed" and result["status"]=="uncovered"
    assert set(result["exports"])==set(probe.SIGNATURES)
    assert (tmp_path/"new"/"collected.json").is_file()
    with pytest.raises(probe.ProbeRefused):
        probe.collect_abi_evidence([str(header)],str(library),str(tmp_path/"new"))

def test_runtime_review_accepts_exact_source_contract_without_prior_hardware_result(tmp_path):
    meta=tmp_path/"version.py"; meta.write_text("CPU fixture: exact package build identity")
    data={"schema":1,"image_id":probe.IMAGE_ID,"review_status":"accepted","reviewer":"CPU fixture",
          "reviewed_at":"2026-11-04T00:00:00+00:00",
          "torch_npu_git_version":"94f8a8e6b523d7ba553e1b80d5b5248478391526",
          "event_query_contract":"polls-status-without-waiting-for-device-completion",
          "contract_source":["https://github.com/Ascend/pytorch/blob/94f8a8e6b523d7ba553e1b80d5b5248478391526/torch_npu/csrc/core/npu/NPUEvent.cpp"],
          "stream_contract":"explicit-device-and-restoring-stream-context",
          "source_sha256s":{str(meta):hashlib.sha256(meta.read_bytes()).hexdigest()},
          "runtime_query_observed":False,"realcard_passed":False}
    path=tmp_path/"runtime.json"; path.write_text(json.dumps(data))
    assert probe.validate_runtime_evidence(str(path))["runtime_query_observed"] is False

def test_lifecycle_result_rejects_any_skipped_or_missing_case():
    expected=list(probe.LIFECYCLE_TESTS)
    result=probe.validate_lifecycle_report(expected,tests=3,failures=0,errors=0,skipped=0,returncode=0)
    assert result["status"]=="passed" and result["executed"]==3
    for kwargs in ({"skipped":1},{"tests":2},{"failures":1},{"errors":1},{"returncode":1}):
        values=dict(tests=3,failures=0,errors=0,skipped=0,returncode=0);values.update(kwargs)
        assert probe.validate_lifecycle_report(expected,**values)["status"]!="passed"
    with pytest.raises(probe.ProbeRefused):
        probe.validate_lifecycle_report(["test_foreign"],tests=1,failures=0,errors=0,skipped=0,returncode=0)

def test_actual_tensor_device_is_checked_before_work(monkeypatch):
    rt=HealthRuntime()
    original=rt.empty
    def wrong(*args,**kwargs):
        tensor=original(*args,**kwargs); tensor.device=SimpleNamespace(type="npu",index=8)
        return tensor
    rt.empty=wrong
    result=probe.health(rt,device=2,byte_limit=4096,timeout_s=1,clock=Clock())
    assert result["status"]=="failed" and not result["readback_matches"]

def test_health_records_device_and_stream_context_restoration():
    rt=HealthRuntime()
    result=probe.health(rt,device=2,byte_limit=4096,timeout_s=1,clock=Clock())
    assert result["status"]=="passed" and result["context_restored"] is True

def test_metadata_output_preserves_reviewable_relevant_excerpts(tmp_path,monkeypatch):
    library=tmp_path/"lib.so";library.write_bytes(b"fixture")
    header=tmp_path/"acl.h";header.write_text("typedef int aclError;\naclError aclInit(const char*);\n")
    monkeypatch.setattr(probe.shutil,"which",lambda name:"/fixture/readelf")
    monkeypatch.setattr(probe,"bounded_symbols",lambda argv:"1 0 1 FUNC GLOBAL DEFAULT 1 aclInit\n")
    result=probe.collect_abi_evidence([str(header)],str(library),str(tmp_path/"output"))
    assert "typedef int aclError" in result["header_excerpts"][str(header.resolve())]
    assert result["symbol_lines"]==["1 0 1 FUNC GLOBAL DEFAULT 1 aclInit"]


def test_symbol_output_limit_caps_actual_retained_tempfile_bytes(monkeypatch):
    import os
    import tempfile
    original=tempfile.TemporaryFile
    sizes=[]
    class ObservedFile:
        def __init__(self):self.raw=original()
        def __getattr__(self,name):return getattr(self.raw,name)
        def __enter__(self):return self
        def __exit__(self,*args):
            sizes.append(os.fstat(self.raw.fileno()).st_size)
            self.raw.close()
    monkeypatch.setattr(tempfile,"TemporaryFile",ObservedFile)
    with pytest.raises(probe.ProbeRefused):
        probe.bounded_symbols([sys.executable,"-c","import os; os.write(1,b'x'*(16<<20))"],timeout_s=3,limit=1024)
    assert sizes and max(sizes)<=1024


def test_symbol_capture_keeps_exact_limit_and_nonzero_exit_semantics():
    assert probe.bounded_symbols([sys.executable,"-c","import os; os.write(1,b'x'*1024)"],limit=1024,timeout_s=3)=="x"*1024
    with pytest.raises(probe.ProbeRefused,match="reader failed"):
        probe.bounded_symbols([sys.executable,"-c","raise SystemExit(7)"],limit=1024,timeout_s=3)
