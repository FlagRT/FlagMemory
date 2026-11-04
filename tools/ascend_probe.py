#!/usr/bin/env python3
"""Small, fail-closed Ascend preparation probes. No framework is imported at collection."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMAGE_ID = "sha256:e13a15d0d02b86e9cd7b1656df3c6b52838c258f270b6b894a91c2b074da5f15"
SIGNATURES = {
    "aclInit": "int32(char*)",
    "aclFinalize": "int32()",
    "aclrtGetDeviceCount": "int32(uint32*)",
    "aclrtGetLogicDevIdByUserDevId": "int32(int32,int32*)",
    "aclrtGetPhyDevIdByLogicDevId": "int32(int32,int32*)",
}
VISIBLE = ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES")
MAX_STATIC = 10 << 20
TORCH_NPU_COMMIT = "94f8a8e6b523d7ba553e1b80d5b5248478391526"
LIFECYCLE_TESTS = (
    "test_actual_tensor_and_cross_stream_dependency",
    "test_completed_reuse_rejects_old_lease",
    "test_pending_borrower_protection_when_observed",
)
_CUSTODY = []  # Failed device work stays strongly held until this process exits.

class ProbeRefused(RuntimeError):
    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence or {}

class ProbeFailure(RuntimeError):
    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence or {}

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()

def absolute_file(value):
    path = Path(value)
    if not path.is_absolute() or not path.is_file():
        raise ProbeRefused("an existing absolute file path is required")
    return path.resolve()

def read_json(path):
    try:
        file = absolute_file(path)
        if file.stat().st_size > MAX_STATIC:
            raise ProbeRefused("evidence exceeds the size limit")
        result = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("object required")
        return result
    except (OSError, ValueError, TypeError) as exc:
        raise ProbeRefused("invalid evidence: " + str(exc)) from exc

def check_hashes(files):
    if not isinstance(files, dict) or not files:
        raise ProbeRefused("nonempty source file hashes required")
    for value, expected in files.items():
        path = absolute_file(value)
        if not isinstance(expected, str) or sha256(path) != expected:
            raise ProbeRefused("evidence file hash mismatch: " + str(path))

def bounded_symbols(argv, timeout_s=30, limit=MAX_STATIC):
    # The shared reader uses a bounded pipe queue and checks before every disk write.
    from tools.run_ascend_bounded import OutputLimit, capture_bounded
    try:
        code, result = capture_bounded(argv, timeout_s=timeout_s, limit=limit)
    except OutputLimit as exc:
        raise ProbeRefused("symbol output limit") from exc
    except TimeoutError as exc:
        raise ProbeRefused("symbol collection deadline") from exc
    if code != 0:
        raise ProbeRefused("symbol reader failed")
    return result.decode("utf-8", errors="replace")

def collect_abi_evidence(header_paths, library_path, output_dir):
    library = absolute_file(library_path)
    if not header_paths or len(header_paths) > 20:
        raise ProbeRefused("one to twenty explicit header paths required")
    headers = [absolute_file(value) for value in header_paths]
    if sum(path.stat().st_size for path in headers) > MAX_STATIC:
        raise ProbeRefused("header input limit")
    target = Path(output_dir)
    if not target.is_absolute() or target.exists():
        raise ProbeRefused("a new absolute output directory is required")
    reader = shutil.which("readelf") or shutil.which("nm")
    if not reader:
        raise ProbeRefused("readelf or nm is required; nothing will be installed")
    argv = [reader, "-Ws" if Path(reader).name.startswith("readelf") else "-D", str(library)]
    symbols = bounded_symbols(argv)
    target.mkdir(parents=False)
    (target / "symbols.txt").write_text(symbols, encoding="utf-8")
    excerpts = {}
    for path in headers:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        selected = set()
        for i, line in enumerate(lines):
            if any(name in line for name in SIGNATURES) or ("typedef" in line and "aclError" in line):
                selected.update(range(max(0, i - 2), min(len(lines), i + 3)))
        excerpts[str(path)] = "\n".join(str(i + 1) + ": " + lines[i] for i in sorted(selected))[:16384]
    (target / "header-excerpts.json").write_text(json.dumps(excerpts, indent=2), encoding="utf-8")
    result = {
        "schema": 1, "stage": "metadata", "status": "uncovered",
        "review_status": "unreviewed", "image_id": IMAGE_ID,
        "library_path": str(Path(library_path)), "library_realpath": str(library),
        "library_sha256": sha256(library),
        "header_sha256s": {str(path): sha256(path) for path in headers},
        "exports": [name for name in SIGNATURES if any(
            line.split() and line.split()[-1].split("@")[0] == name and " UND " not in line
            for line in symbols.splitlines())],
        "header_excerpts": excerpts,
        "symbol_lines": [line for line in symbols.splitlines() if any(
            line.split() and line.split()[-1].split("@")[0] == name for name in SIGNATURES)][:100],
        "platform": platform.platform(), "python_c_int_bytes": ctypes.sizeof(ctypes.c_int),
        "reason": "static evidence only; independent ABI and runtime review required",
    }
    (target / "collected.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

class _ReviewedAcl:
    physical_argument_semantics = "identity-only-user-equals-logic"
    def __init__(self, library):
        self.library = library
        ptr_i = ctypes.POINTER(ctypes.c_int32)
        for name, args in (
            ("aclInit", [ctypes.c_char_p]), ("aclFinalize", []),
            ("aclrtGetDeviceCount", [ctypes.POINTER(ctypes.c_uint32)]),
            ("aclrtGetLogicDevIdByUserDevId", [ctypes.c_int32, ptr_i]),
            ("aclrtGetPhyDevIdByLogicDevId", [ctypes.c_int32, ptr_i]),
        ):
            function = getattr(library, name)
            function.argtypes, function.restype = args, ctypes.c_int32
    def init(self): return self.library.aclInit(None)
    def finalize(self): return self.library.aclFinalize()
    def get_count(self):
        value = ctypes.c_uint32(0xFFFFFFFF)
        code = self.library.aclrtGetDeviceCount(ctypes.byref(value))
        return code, value.value
    def get_logic(self, user):
        value = ctypes.c_int32(-1)
        code = self.library.aclrtGetLogicDevIdByUserDevId(user, ctypes.byref(value))
        return code, value.value
    def get_physical(self, user):
        value = ctypes.c_int32(-1)
        code = self.library.aclrtGetPhyDevIdByLogicDevId(user, ctypes.byref(value))
        return code, value.value

def load_reviewed_acl(evidence_path, *, cdll_factory=ctypes.CDLL):
    data = read_json(evidence_path)
    required = {"schema","image_id","library_realpath","library_sha256","header_sha256s",
                "signatures","init_required","init_argument","physical_argument_semantics",
                "reviewer","reviewed_at","review_status","exports"}
    if not required <= data.keys():
        raise ProbeRefused("ABI review fields are missing")
    if (data["schema"] != 1 or data["image_id"] != IMAGE_ID
        or data["review_status"] != "accepted" or not data["reviewer"] or not data["reviewed_at"]
        or data["signatures"] != SIGNATURES or data["init_required"] is not True
        or data["init_argument"] is not None or data["physical_argument_semantics"] != "identity-only-user-equals-logic"
        or not set(SIGNATURES) <= set(data["exports"]) or ctypes.sizeof(ctypes.c_int32) != 4):
        raise ProbeRefused("unsupported or unreviewed ABI")
    library = absolute_file(data["library_realpath"])
    if str(library) != data["library_realpath"]:
        raise ProbeRefused("library path must be the resolved runtime library")
    check_hashes({str(library):data["library_sha256"]})
    check_hashes(data["header_sha256s"])
    if any("devlib" in part for part in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)):
        raise ProbeRefused("development stub search path is not allowed")
    try:
        bound = _ReviewedAcl(cdll_factory(str(library)))
        if cdll_factory is ctypes.CDLL and Path("/proc/self/maps").is_file():
            loaded = [line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                      if "libascendcl.so" in line]
            if not loaded or any(Path(path).resolve() != library for path in loaded):
                raise ProbeFailure("actual loaded ACL path differs from reviewed runtime library")
        return bound
    except (OSError, AttributeError, TypeError) as exc:
        raise ProbeFailure("reviewed library could not be bound: " + str(exc)) from exc

def map_device(acl, *, expected_physical, environment):
    evidence = {"raw_return_codes": {}, "cleanup": {"finalized": False}}
    if type(expected_physical) is not int or expected_physical < 0:
        raise ProbeRefused("nonnegative expected physical device required", evidence)
    if any(environment.get(key) for key in VISIBLE):
        raise ProbeRefused("visibility reordering is not supported", evidence)
    if getattr(acl, "physical_argument_semantics", None) != "identity-only-user-equals-logic":
        raise ProbeRefused("reviewed physical numbering semantics required", evidence)
    initialized = False
    failure = None
    def check(name, code):
        evidence["raw_return_codes"][name] = code
        if type(code) is not int or code != 0:
            raise ProbeFailure(name + " returned an error", evidence)
    try:
        check("init", acl.init())
        initialized = True
        code, count = acl.get_count(); check("count", code)
        if type(count) is not int or count != 1:
            raise ProbeRefused("exactly one user device required", evidence)
        code, logical = acl.get_logic(0); check("logic", code)
        evidence.update(user_device=0, logical_device=logical)
        # CANN 9.0 calls the old argument logic; 9.2 beta1 corrects it to user.
        # With identity numbering both meanings agree. Never chain a remapped logic ID.
        if type(logical) is not int or logical != 0:
            raise ProbeRefused("nonidentity numbering is ambiguous for the old interface", evidence)
        code, physical = acl.get_physical(0); check("physical", code)
        evidence["physical_device"] = physical
        if type(physical) is not int or physical != expected_physical:
            raise ProbeRefused("physical identity does not match the approved target", evidence)
    except Exception as exc:
        failure = exc
    finally:
        if initialized:
            try:
                code = acl.finalize()
                evidence["raw_return_codes"]["finalize"] = code
                evidence["cleanup"]["finalized"] = code == 0
                if code != 0:
                    raise ProbeFailure("finalize returned an error", evidence)
            except Exception as exc:
                evidence["cleanup"]["error"] = str(exc)
                if failure is None:
                    failure = exc
                else:
                    evidence["primary_error"] = str(failure)
                    failure = ProbeFailure(str(failure) + "; finalize failed", evidence)
    if failure:
        if isinstance(failure, (ProbeRefused, ProbeFailure)):
            failure.evidence = evidence
            raise failure
        raise ProbeFailure(str(failure), evidence) from failure
    return dict(evidence, status="passed")

def _adapter():
    location = Path(__file__).resolve().parents[1] / "tests" / "pool" / "ascend_test_adapter.py"
    spec = importlib.util.spec_from_file_location("_ascend_probe_adapter", location)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def health(runtime, *, device, byte_limit, timeout_s, clock=time.monotonic):
    if (type(device) is not int or device < 0 or type(byte_limit) is not int
        or not 4096 <= byte_limit <= 16 << 20 or isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int,float)) or not math.isfinite(timeout_s)
        or not 0 < timeout_s <= 30):
        raise ProbeRefused("invalid device, health budget or deadline")
    adapter = _adapter()
    budget = adapter.AllocationBudget(byte_limit)
    allocator = adapter.AscendTensorAllocator(runtime, device, budget)
    backend = adapter.AscendTestBackend(runtime, device)
    result = {"allocated_bytes": 0, "release_calls": 0, "readback_matches": False,
              "cleanup": {"released": False, "completion_confirmed": False}, "query_samples_s": []}
    tensor = None
    complete = False
    release_attempted = False
    started = clock()
    try:
        prior_device = runtime.npu.current_device()
        prior_stream = runtime.npu.current_stream(device)
        with runtime.npu.device(device):
            stream = runtime.npu.Stream(device=device)
        backend.validate_stream_device(stream, device)
        with runtime.npu.device(device), runtime.npu.stream(stream):
            current = backend.current_stream(device)
            if not backend.streams_equal(current, stream):
                raise ProbeRefused("stream context did not select the requested stream")
            tensor = allocator.allocate(4096, device, stream=stream)
            actual_device = getattr(tensor, "device", None)
            if (getattr(actual_device, "type", None) != "npu"
                or getattr(actual_device, "index", None) != device
                or tensor.dtype != runtime.uint8 or tensor.numel() != 4096):
                raise ProbeFailure("allocated tensor identity, dtype or size mismatch")
            tensor.fill_(23)
            event = backend.create_event(device)
            backend.record_event(event, stream)
        result["context_restored"] = (
            runtime.npu.current_device() == prior_device
            and backend.streams_equal(runtime.npu.current_stream(device), prior_stream))
        if not result["context_restored"]:
            raise ProbeFailure("device or stream context was not restored")
        while True:
            before = clock()
            done = backend.query_event(event)
            after = clock()
            if len(result["query_samples_s"]) < 16:
                result["query_samples_s"].append(after-before)
            if after-started >= timeout_s:
                raise ProbeFailure("health deadline")
            if done:
                complete = True
                break
            time.sleep(0.001)
        result["cleanup"]["completion_confirmed"] = True
        result["readback_matches"] = tensor.cpu().tolist() == [23] * 4096
        if not result["readback_matches"]:
            raise ProbeFailure("health readback mismatch")
        release_attempted = True
        allocator.release(tensor)
        tensor = None
        result["cleanup"]["released"] = True
        result.update(status="passed", stream_device=device)
    except Exception as exc:
        # Release only after an actual completed event. Unknown work remains held.
        if tensor is not None and complete and not release_attempted:
            try:
                allocator.release(tensor)
                tensor = None
                result["cleanup"]["released"] = True
            except Exception as cleanup_exc:
                result["cleanup"]["error"] = str(cleanup_exc)
        if tensor is not None:
            _CUSTODY.append((runtime, allocator, backend, tensor))
        result.update(status="failed", reason=str(exc))
    result.update(allocated_bytes=allocator.allocated_bytes, release_calls=allocator.release_calls)
    return result

def validate_runtime_evidence(path):
    data = read_json(path)
    prefix = "https://github.com/Ascend/pytorch/blob/" + TORCH_NPU_COMMIT + "/"
    sources = data.get("contract_source", [])
    if (data.get("schema") != 1 or data.get("image_id") != IMAGE_ID
        or data.get("review_status") != "accepted" or not data.get("reviewer")
        or not data.get("reviewed_at") or data.get("torch_npu_git_version") != TORCH_NPU_COMMIT
        or data.get("event_query_contract") != "polls-status-without-waiting-for-device-completion"
        or data.get("stream_contract") != "explicit-device-and-restoring-stream-context"
        or not isinstance(sources, list) or not sources
        or not all(isinstance(source, str) and source.startswith(prefix) for source in sources)):
        raise ProbeRefused("reviewed exact-source runtime contract required")
    # Installed version.py/headers bind the trusted package to the reviewed official source.
    # A prior successful health probe is deliberately not a prerequisite for initial health.
    check_hashes(data.get("source_sha256s"))
    return data

def validate_lifecycle_report(names, *, tests, failures, errors, skipped, returncode):
    if sorted(names) != sorted(LIFECYCLE_TESTS):
        raise ProbeRefused("the exact three Ascend lifecycle cases are required")
    status = "passed"
    if returncode or tests != 3 or failures or errors:
        status = "failed"
    elif skipped:
        status = "uncovered"
    return {"status": status, "executed": tests, "failures": failures,
            "errors": errors, "skipped": skipped}

def validate_prior(path, stage):
    data = read_json(path)
    if (data.get("status") != "passed" or data.get("stage") != stage
        or data.get("image_id") != IMAGE_ID
        or not data.get("run_id") or not data.get("source_sha256")
        or data.get("source_sha256") != os.environ.get("FLAGMEMORY_SOURCE_SHA256")):
        raise ProbeRefused("matching successful prior stage evidence required")
    return data

def lifecycle(mapping_path, runtime_path, health_path):
    mapping = validate_prior(mapping_path, "mapping")
    validate_prior(health_path, "health")
    validate_runtime_evidence(runtime_path)
    if (mapping.get("user_device"), mapping.get("logical_device"), mapping.get("physical_device")) != (0, 0, 15):
        raise ProbeRefused("identity mapping for physical target 15 required")
    selected = str(Path(__file__).resolve().parents[1] / "tests/pool/test_pool_ascend_realcard.py")
    common = [sys.executable, "-B", "-m", "pytest", selected, "-q", "-o", "addopts=",
              "-m", "realcard", "-p", "no:cacheprovider"]
    collected = bounded_symbols(common + ["--collect-only"], timeout_s=30, limit=1 << 20)
    names = [line.split("::")[-1].strip() for line in collected.splitlines() if "::test_" in line]
    if sorted(names) != sorted(LIFECYCLE_TESTS):
        raise ProbeRefused("lifecycle collection differs from the three approved cases")
    env = dict(os.environ)
    env.update(FLAGMEMORY_ASCEND_ENABLE="approved-mapping-and-bounds",
               FLAGMEMORY_ASCEND_DEVICE=str(mapping["user_device"]),
               FLAGMEMORY_ASCEND_BYTE_LIMIT=str(64 << 20),
               FLAGMEMORY_ASCEND_QUERY_NONBLOCKING="verified")
    with tempfile.TemporaryDirectory(prefix="flagmemory-lifecycle-") as temp:
        report_path = Path(temp) / "report.xml"
        # Output is inherited into the host executor's bounded stream.
        process = subprocess.Popen(common + ["--junitxml", str(report_path)],
                                   stdin=subprocess.DEVNULL, env=env, shell=False)
        try:
            code = process.wait(timeout=240)
        except subprocess.TimeoutExpired:
            process.kill()
            raise ProbeFailure("lifecycle child deadline")
        if not report_path.is_file() or report_path.stat().st_size > 1 << 20:
            raise ProbeFailure("bounded lifecycle report missing")
        root = ET.parse(report_path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
        totals = {key: sum(int(suite.get(key, "0")) for suite in suites)
                  for key in ("tests", "failures", "errors", "skipped")}
        cases = [case.get("name") for case in root.iter("testcase")]
        return validate_lifecycle_report(cases, returncode=code, **totals)


BRIDGE_FILE = "tests/pool/test_pool_device_context_ascend_realcard.py"
BRIDGE_NODEIDS = tuple(BRIDGE_FILE+"::"+name for name in LIFECYCLE_TESTS)


def bridge_helper():
    root=Path(__file__).resolve().parents[1]/"tests/pool"
    sys.path.insert(0,str(root))
    import device_context_ascend_case
    return device_context_ascend_case


def validate_bridge_lifecycle_report(nodeids, *, tests, failures, errors, skipped, returncode):
    if sorted(nodeids) != sorted(BRIDGE_NODEIDS):
        raise ProbeRefused("exact new bridge lifecycle nodeids required")
    result=validate_lifecycle_report([name.split("::")[-1] for name in nodeids],
        tests=tests,failures=failures,errors=errors,skipped=skipped,returncode=returncode)
    result.update(adapter_kind="device-context-ascend-profile",nodeids=list(nodeids))
    return result


def bridge_lifecycle(mapping_path, runtime_path, health_path, device_source_path):
    mapping=validate_prior(mapping_path,"mapping")
    validate_prior(health_path,"health")
    validate_runtime_evidence(runtime_path)
    if (mapping.get("user_device"),mapping.get("logical_device"),mapping.get("physical_device")) != (0,0,15):
        raise ProbeRefused("identity mapping for physical target 15 required")
    helper=bridge_helper()
    try:
        digest=os.environ.get("FLAGMEMORY_DEVICE_MANIFEST_SHA256")
        manifest=helper.load_manifest(Path(device_source_path),digest)
        helper.verify_runtime_origins(Path(device_source_path).parent)
    except ValueError as exc: raise ProbeRefused(str(exc)) from exc
    selected=str(Path(__file__).resolve().parents[1]/BRIDGE_FILE)
    common=[sys.executable,"-B","-m","pytest",selected,"-q","-s","-o","addopts=","-m","realcard","-p","no:cacheprovider"]
    collected=bounded_symbols(common+["--collect-only"],timeout_s=30,limit=1<<20)
    nodeids=[line.strip() for line in collected.splitlines() if "::test_" in line]
    if sorted(nodeids) != sorted(BRIDGE_NODEIDS):
        raise ProbeRefused("new bridge collection differs from exact approved nodeids")
    env=dict(os.environ)
    env.update(FLAGMEMORY_ASCEND_ENABLE="approved-bridge-lifecycle",FLAGMEMORY_ASCEND_DEVICE="0",
        FLAGMEMORY_ASCEND_BYTE_LIMIT=str(64<<20),FLAGMEMORY_ASCEND_QUERY_NONBLOCKING="verified",
        FLAGMEMORY_DEVICE_SOURCE=str(Path(device_source_path).resolve()))
    with tempfile.TemporaryDirectory(prefix="flagmemory-bridge-lifecycle-") as temp:
        report_path=Path(temp)/"report.xml"
        process=subprocess.Popen(common+["--junitxml",str(report_path)],stdin=subprocess.DEVNULL,env=env,shell=False)
        try: code=process.wait(timeout=240)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
            raise ProbeFailure("bridge lifecycle child deadline")
        if not report_path.is_file() or report_path.stat().st_size > 1<<20:
            raise ProbeFailure("bounded bridge lifecycle report missing")
        root=ET.parse(report_path).getroot()
        suites=[root] if root.tag=="testsuite" else list(root.findall("testsuite"))
        totals={key:sum(int(suite.get(key,"0")) for suite in suites) for key in ("tests","failures","errors","skipped")}
        ids=[case.get("classname","").replace(".","/")+".py::"+case.get("name","") for case in root.iter("testcase")]
        result=validate_bridge_lifecycle_report(ids,returncode=code,**totals)
        result["device_runtime"]={key:manifest[key] for key in ("source_repository","source_commit","runtime_tree")}
        result["device_runtime"]["manifest_sha256"]=digest
        return result

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="stage",required=True)
    static=sub.add_parser("collect-abi")
    static.add_argument("--header",action="append",required=True)
    static.add_argument("--library",required=True)
    static.add_argument("--output-dir",required=True)
    mapping=sub.add_parser("mapping")
    mapping.add_argument("--evidence",required=True)
    mapping.add_argument("--expected-physical",required=True,type=int)
    check=sub.add_parser("health")
    check.add_argument("--mapping",required=True)
    check.add_argument("--runtime-evidence",required=True)
    check.add_argument("--byte-limit",required=True,type=int)
    check.add_argument("--timeout-s",required=True,type=float)
    life=sub.add_parser("lifecycle")
    life.add_argument("--mapping",required=True)
    life.add_argument("--runtime-evidence",required=True)
    life.add_argument("--health",required=True)
    bridge=sub.add_parser("bridge-lifecycle")
    bridge.add_argument("--mapping",required=True)
    bridge.add_argument("--runtime-evidence",required=True)
    bridge.add_argument("--health",required=True)
    bridge.add_argument("--device-source",required=True)
    args=parser.parse_args(argv)
    started=time.monotonic()
    result={"schema":1,"stage":args.stage,"run_id":os.environ.get("FLAGMEMORY_RUN_ID"),
            "image_id":IMAGE_ID,"source_sha256":os.environ.get("FLAGMEMORY_SOURCE_SHA256"),
            "started_at":datetime.now(timezone.utc).isoformat(),"cleanup":{}}
    try:
        if args.stage=="collect-abi":
            result.update(collect_abi_evidence(args.header,args.library,args.output_dir))
        elif args.stage=="mapping":
            result.update(map_device(load_reviewed_acl(args.evidence),
                          expected_physical=args.expected_physical,environment=dict(os.environ)))
        elif args.stage=="bridge-lifecycle":
            result.update(bridge_lifecycle(args.mapping,args.runtime_evidence,args.health,args.device_source))
        elif args.stage=="lifecycle":
            result.update(lifecycle(args.mapping,args.runtime_evidence,args.health))
        else:
            mapping=validate_prior(args.mapping,"mapping")
            if (mapping.get("status")!="passed" or mapping.get("image_id")!=IMAGE_ID
                or mapping.get("physical_device")!=15 or mapping.get("user_device")!=0
                or mapping.get("logical_device")!=0):
                raise ProbeRefused("successful identity mapping for this image required")
            validate_runtime_evidence(args.runtime_evidence)
            # Validate numeric bounds before importing any framework.
            if not 4096 <= args.byte_limit <= 16<<20 or not 0 < args.timeout_s <=30:
                raise ProbeRefused("invalid health bounds")
            import torch
            import torch_npu  # noqa: F401: registration only after the gates
            result.update(health(torch,device=mapping["user_device"],
                                 byte_limit=args.byte_limit,timeout_s=args.timeout_s))
        code=0 if result["status"] in ("passed","uncovered") else 1
    except (ProbeRefused,ProbeFailure) as exc:
        result.update(exc.evidence)
        result.update(status="refused" if isinstance(exc,ProbeRefused) else "failed",reason=str(exc))
        code=2 if isinstance(exc,ProbeRefused) else 1
    except Exception as exc:
        result.update(status="failed",reason=str(exc)); code=1
    result.update(elapsed_s=time.monotonic()-started,exit_code=code)
    print(json.dumps(result,sort_keys=True))
    return code

if __name__=="__main__":
    raise SystemExit(main())
