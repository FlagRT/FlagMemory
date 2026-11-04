"""All Docker and host-device calls here are injected; no daemon is contacted."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from datetime import datetime,timezone
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("run_ascend_bounded",ROOT/"tools"/"run_ascend_bounded.py")
runner=importlib.util.module_from_spec(spec); sys.modules[spec.name]=runner; spec.loader.exec_module(runner)
CID="a"*64

class FakeProcess:
    def __init__(self,docker): self.docker=docker; self.sent=False; self.stopped=False
    def read(self,max_bytes,timeout_s):
        self.docker.now+=.1
        if self.docker.mode=="never_exits": self.docker.now+=50; return b""
        if self.sent: return b""
        self.sent=True
        if self.docker.mode=="output_exceeds_limit": return b"x"*max_bytes
        return json.dumps({"schema":1,"stage":"mapping","status":"passed","image_id":runner.IMAGE_ID,
                           "run_id":self.docker.run_id,"source_sha256":self.docker.source_sha,
                           "user_device":0,"logical_device":0,"physical_device":15}).encode()+b"\n"
    def poll(self):
        if self.docker.mode=="never_exits": return None
        return (1 if self.docker.mode=="nonzero" else 0) if self.sent else None
    def terminate(self,timeout_s): self.stopped=True

class FakeDocker:
    def __init__(self,mode=None):
        self.mode=mode; self.now=0; self.calls=[]; self.exists=False; self.removed_ids=[]
        self.created_id=CID; self.run_id=None; self.source_sha=None; self.inspects=0
        self.current=None
    def clock(self): return self.now
    def call(self,argv,timeout_s):
        assert 0 < timeout_s <= 10
        self.calls.append(argv); self.now+=.01
        if self.mode=="unreachable": raise OSError("Docker unavailable")
        if argv[:2]==["npu-smi","info"]: return 0,SAMPLE,""
        action=argv[1]
        if action=="ps":
            if "--filter" in argv: return 0,("b"*64 if self.mode=="same_name" else ""),""
            if self.mode=="overlap": return 0,"b"*64,""
            if self.mode=="nonoverlap": return 0,"b"*64,""
            return 0,"",""
        if action=="create":
            self.exists=True; self.run_id=argv[argv.index("--label")+1].split("=",1)[1]
            envs=dict(argv[i+1].split("=",1) for i,a in enumerate(argv) if a=="--env")
            self.source_sha=envs["FLAGMEMORY_SOURCE_SHA256"]
            self.current={"Id":CID,"Image":runner.IMAGE_ID,
              "Config":{"Labels":{"flagmemory.run_id":self.run_id}},
              "HostConfig":{"Privileged":False,"Runtime":"runc","CapDrop":["ALL"],
                "SecurityOpt":["no-new-privileges"],"NetworkMode":"none","NanoCpus":2000000000,
                "Memory":4294967296,"IpcMode":"private","PidMode":"","RestartPolicy":{"Name":"no"},
                "LogConfig":{"Type":"none"},"DeviceCgroupRules":[],"DeviceRequests":[],
                "Devices":[{"PathOnHost":"/dev/"+n,"PathInContainer":"/dev/"+n,"CgroupPermissions":"rw"}
                           for n in ("davinci15","davinci_manager","devmm_svm","hisi_hdc")]},
              "Mounts":[{"Source":"/usr/local/Ascend/driver","Destination":"/usr/local/Ascend/driver","RW":False},
                {"Source":self.source_dir,"Destination":"/work","RW":False},
                {"Source":self.input_dir,"Destination":"/input","RW":False}]}
            return 0,("bad" if self.mode=="bad_id" else CID),""
        if action=="inspect":
            if not self.exists and argv[-1]==CID: return 1,"","No such object"
            if argv[-1]=="b"*64:
                item={"Id":"b"*64,"HostConfig":{"Privileged":False,"Devices":[
                    {"PathOnHost":"/dev/davinci0" if self.mode=="nonoverlap" else "/dev/davinci15"}],
                    "DeviceCgroupRules":[],"DeviceRequests":[]},"Mounts":[]}
                return 0,json.dumps([item]),""
            self.inspects+=1
            if self.mode=="inspect_timeout": raise TimeoutError("inspect deadline")
            item=json.loads(json.dumps(self.current))
            if self.mode=="wrong_image": item["Image"]="sha256:"+"b"*64
            if self.mode=="privileged": item["HostConfig"]["Privileged"]=True
            if self.mode=="broad": item["HostConfig"]["DeviceCgroupRules"]=["c *:* rw"]
            if self.mode=="extra_device": item["HostConfig"]["Devices"].append({"PathOnHost":"/dev/davinci0"})
            if self.mode=="writable_driver":
                next(m for m in item["Mounts"] if m["Destination"]=="/usr/local/Ascend/driver")["RW"]=True
            if self.mode=="foreign_label_on_cleanup" and self.inspects>=2:
                item["Config"]["Labels"]["flagmemory.run_id"]="foreign"
            return 0,json.dumps([item]),""
        if action in ("stop","kill"):
            if self.mode=="stop_timeout": raise TimeoutError("stop deadline")
            return 0,"",""
        if action=="rm":
            if self.mode=="rm_timeout": raise TimeoutError("remove deadline")
            self.removed_ids.append(argv[-1]); self.exists=False; return 0,"",""
        if action=="info": return 0,"{}",""
        raise AssertionError(argv)
    def start_attached(self,cid): assert cid==CID; return FakeProcess(self)

@pytest.fixture
def valid_run(tmp_path):
    source=tmp_path/"source"; source.mkdir(); (source/"file.py").write_text("# frozen fixture\n")
    proof=tmp_path/"filter.txt"; proof.write_text("CPU fixture: effective whitelist")
    reviewed=tmp_path/"reviewed.json"; reviewed.write_text("{}")
    data={"schema":1,"image_id":runner.IMAGE_ID,"source_sha256":runner.source_digest(source),
          "collected_at":datetime.now(timezone.utc).isoformat(),"window_confirmed":True,
          "window_mode":"shared_candidate","target":{"physical":15,"major":235,"minor":15,
          "healthy":True,"process_count":0},"target_snapshot":dict(FACTS),"hbm_tolerance_mb":128,
          "filter":{"version":"v2","review_status":"accepted","reviewer":"CPU fixture",
                    "evidence_path":str(proof),"sha256":hashlib.sha256(proof.read_bytes()).hexdigest(),
                    "allowed_compute":[15],"controls":["davinci_manager","devmm_svm","hisi_hdc"]},
          "inputs":{"reviewed-abi.json":str(reviewed)}}
    path=tmp_path/"preflight.json"; path.write_text(json.dumps(data))
    return dict(stage="mapping",run_dir=str(tmp_path/"run"),source_dir=str(source),
                image=runner.IMAGE_ID,preflight_path=str(path))

def run(mode,valid_run,**extra):
    docker=FakeDocker(mode)
    docker.source_dir=valid_run["source_dir"]; docker.input_dir=str(Path(valid_run["run_dir"])/"input")
    result=runner.run_stage(**valid_run,docker=docker,clock=docker.clock,**extra)
    return docker,result

def test_shared_nonoverlapping_candidate_runs_and_cleans_exact_id(valid_run):
    docker,result=run("nonoverlap",valid_run)
    assert result["status"]=="passed" and result["cleanup"]["removed"]
    assert docker.removed_ids==[CID]
    assert all(a[-1]==CID for a in docker.calls if a[1] in ("stop","kill","rm"))

@pytest.mark.parametrize("mode",["never_exits","nonzero","output_exceeds_limit"])
def test_work_failure_never_accepts_stdout_claim(mode,valid_run,monkeypatch):
    monkeypatch.setattr(runner,"OUTPUT_LIMIT",2048)
    docker,result=run(mode,valid_run)
    assert result["status"]=="failed"
    assert result["reason"]=={"never_exits":"deadline","nonzero":"container_exit",
                              "output_exceeds_limit":"output_limit"}[mode]
    assert docker.removed_ids==[CID] and result["captured_bytes"]<=2048

@pytest.mark.parametrize("mode",["same_name","bad_id","wrong_image","privileged","broad",
                                "extra_device","writable_driver","overlap","unreachable"])
def test_bad_environment_never_starts(mode,valid_run):
    docker,result=run(mode,valid_run)
    assert result["status"] in ("refused","failed")
    assert not any(a[1]=="start" for a in docker.calls)
    if mode in ("same_name","bad_id","overlap","unreachable"): assert docker.removed_ids==[]

def test_foreign_label_is_never_removed(valid_run):
    docker,result=run("foreign_label_on_cleanup",valid_run)
    assert docker.removed_ids==[]
    assert result["status"]=="failed" and result["cleanup"]["ownership_verified"] is False

@pytest.mark.parametrize("mode",["inspect_timeout","stop_timeout","rm_timeout"])
def test_cleanup_errors_remain_visible(mode,valid_run):
    docker,result=run(mode,valid_run)
    assert result["status"]=="failed"
    assert result["cleanup"].get("errors")

@pytest.mark.parametrize("field",["window","stale","filter","source"])
def test_bad_preflight_refuses_before_create(field,valid_run):
    path=Path(valid_run["preflight_path"]); data=json.loads(path.read_text())
    if field=="window": data["window_confirmed"]=False
    if field=="stale": data["collected_at"]="2000-01-01T00:00:00+00:00"
    if field=="filter": data["filter"]["sha256"]="0"*64
    if field=="source": data["source_sha256"]="0"*64
    path.write_text(json.dumps(data))
    docker,result=run(None,valid_run)
    assert result["status"]=="refused" and not any(a[1]=="create" for a in docker.calls)

def test_existing_run_evidence_is_not_overwritten(valid_run):
    Path(valid_run["run_dir"]).mkdir()
    sentinel=Path(valid_run["run_dir"])/"sentinel"; sentinel.write_text("keep")
    docker,result=run(None,valid_run)
    assert result["status"]=="refused" and sentinel.read_text()=="keep" and not docker.calls

def test_metadata_configuration_has_no_devices_or_driver(tmp_path):
    argv=runner.build_create_argv(image=runner.IMAGE_ID,name="fm-test",run_id="test",
                                 source_dir=str(tmp_path),stage="metadata")
    assert "--device" not in argv and "/usr/local/Ascend/driver" not in " ".join(argv)


def test_success_preserves_postcheck_and_a_reusable_stage_evidence_file(valid_run):
    docker,result=run(None,valid_run)
    root=Path(valid_run["run_dir"])
    assert result["status"]=="passed"
    assert (root/"post-npu-smi.txt").read_text()==SAMPLE
    evidence=json.loads((root/"stage-evidence.json").read_text())
    assert evidence["run_id"]==result["run_id"] and evidence["physical_device"]==15

def test_v1_effective_filter_is_accepted_without_v2_specific_files(valid_run):
    path=Path(valid_run["preflight_path"]); data=json.loads(path.read_text())
    data["filter"]["version"]="v1";path.write_text(json.dumps(data))
    docker,result=run(None,valid_run)
    assert result["status"]=="passed"

def test_preflight_health_requires_matching_successful_mapping(valid_run):
    valid_run["stage"]="health"
    path=Path(valid_run["preflight_path"]); data=json.loads(path.read_text())
    mapping=path.parent/"mapping.json";mapping.write_text(json.dumps({"status":"failed"}))
    runtime=path.parent/"runtime.json";runtime.write_text("{}")
    data["inputs"]={"mapping.json":str(mapping),"runtime.json":str(runtime)}
    path.write_text(json.dumps(data))
    docker,result=run(None,valid_run)
    assert result["status"]=="refused" and not any(a[1]=="create" for a in docker.calls)

def test_docker_unqueryable_after_cleanup_is_failure(valid_run):
    docker=FakeDocker()
    docker.source_dir=valid_run["source_dir"];docker.input_dir=str(Path(valid_run["run_dir"])/"input")
    original=docker.call
    def call(argv,timeout_s):
        if argv[:2]==["docker","info"]:return 1,"","daemon lost"
        return original(argv,timeout_s)
    docker.call=call
    result=runner.run_stage(**valid_run,docker=docker,clock=docker.clock)
    assert result["status"]=="failed" and result["cleanup"]["docker_queryable"] is False

def test_changed_postcheck_prevents_reusing_stage_evidence(valid_run):
    docker=FakeDocker()
    docker.source_dir=valid_run["source_dir"];docker.input_dir=str(Path(valid_run["run_dir"])/"input")
    original=docker.call
    count=0
    def call(argv,timeout_s):
        nonlocal count
        if argv[:2]==["npu-smi","info"]:
            count+=1
            if count==3:return 0,SAMPLE.replace("2880 /","4500 /"),""
        return original(argv,timeout_s)
    docker.call=call
    result=runner.run_stage(**valid_run,docker=docker,clock=docker.clock)
    assert result["status"]=="uncovered" and result["cleanup"]["postcheck_review_required"]
    assert not (Path(valid_run["run_dir"])/"stage-evidence.json").exists()

def test_local_control_wrapper_bounds_cpu_child_output_and_deadline(monkeypatch):
    local=runner.LocalDocker()
    code,out,err=local.call([sys.executable,"-c","print('cpu-only'); raise SystemExit(7)"],timeout_s=2)
    assert code==7 and "cpu-only" in out
    monkeypatch.setattr(runner,"CONTROL_LIMIT",128)
    with pytest.raises(runner.Refused):
        local.call([sys.executable,"-c","print('x'*10000)"],timeout_s=2)
    with pytest.raises(TimeoutError):
        local.call([sys.executable,"-c","import time; time.sleep(1)"],timeout_s=.03)

def test_real_cpu_pipe_reader_preserves_all_chunks():
    import subprocess
    process=subprocess.Popen([sys.executable,"-c","import sys; sys.stdout.buffer.write(b'x'*200000)"],
                             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    attached=runner.Attached(process)
    data=bytearray()
    try:
        while True:
            data.extend(attached.read(65536,.2))
            if attached.poll() is not None:break
        assert len(data)==200000
    finally:attached.terminate(.5)

SAMPLE = """timestamp: synthetic CPU fixture
| NPU Name | Health | Power(W) Temp(C) Hugepages-Usage(page) |
| Chip Phy-ID | Bus-Id | AICore(%) Memory-Usage(MB) HBM-Usage(MB) |
| 0 AscendDevice | OK | 120 30 0 / 0 |
| 0 0 | 0000:bb:00.0 | 60 0 / 0 9000 / 65536 |
| 7 AscendDevice | OK | - 33 0 / 0 |
| 1 15 | 0000:aa:00.0 | 0 0 / 0 2880 / 65536 |
| NPU Chip | Process id | Process name | Process memory(MB) | Process id in container |
| 0 0 | 4567 | worker | 6000 | NA |
| No running processes found in NPU 7 |
"""
FACTS={"physical":15,"npu":7,"chip":1,"bus":"0000:aa:00.0","health":"OK",
       "hbm_used_mb":2880,"hbm_total_mb":65536,"pids":[],"aicore":0}

def test_fixed_host_snapshot_parser_selects_only_target_facts():
    assert runner.parse_target_snapshot(SAMPLE)==FACTS
    noisy=SAMPLE.replace("synthetic CPU fixture","different timestamp").replace("120 30","250 50").replace("9000 /","15000 /")
    assert runner.parse_target_snapshot(noisy)==FACTS
    runner.compare_target_snapshot(FACTS,runner.parse_target_snapshot(noisy),tolerance_mb=128)

@pytest.mark.parametrize("change",["pid","health","memory","malformed"])
def test_target_changes_or_unknown_shape_are_refused(change):
    sample=SAMPLE
    if change=="pid":sample=sample.replace("| No running processes found in NPU 7 |","| 7 1 | 9999 | worker | 800 | NA |")
    if change=="health":sample=sample.replace("| 7 AscendDevice | OK","| 7 AscendDevice | Warning")
    if change=="memory":sample=sample.replace("2880 /","4500 /")
    if change=="malformed":sample=sample.replace("HBM-Usage(MB)","unknown format")
    with pytest.raises(runner.Refused):
        runner.compare_target_snapshot(FACTS,runner.parse_target_snapshot(sample),tolerance_mb=128)


@pytest.mark.parametrize("device_path", ["/dev", "/dev/", "/dev/davinci", "/dev/davinci15-alias", "/unknown/device"])
def test_unknown_or_directory_device_authorization_is_not_an_empty_safe_set(device_path,valid_run):
    docker=FakeDocker("nonoverlap")
    docker.source_dir=valid_run["source_dir"];docker.input_dir=str(Path(valid_run["run_dir"])/"input")
    original=docker.call
    def call(argv,timeout_s):
        code,out,err=original(argv,timeout_s)
        if argv[:2]==["docker","inspect"] and argv[-1]=="b"*64:
            item=json.loads(out)[0]
            item["HostConfig"]["Devices"]=[{"PathOnHost":device_path}]
            return code,json.dumps([item]),err
        return code,out,err
    docker.call=call
    result=runner.run_stage(**valid_run,docker=docker,clock=docker.clock)
    assert result["status"]=="refused"
    assert not any(argv[1]=="create" for argv in docker.calls)
    assert docker.current is None

def test_other_chip_valid_process_does_not_hide_unknown_target_process_row():
    sample=SAMPLE.replace("| No running processes found in NPU 7 |",
        "| 7 0 | 8888 | worker | 500 | NA |\n| 7 1 unknown | 9999 | worker | 800 | NA |")
    with pytest.raises(runner.Refused):
        runner.compare_target_snapshot(FACTS,runner.parse_target_snapshot(sample),tolerance_mb=128)

def test_control_output_limit_caps_actual_retained_tempfile_bytes(monkeypatch):
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
    monkeypatch.setattr(runner,"CONTROL_LIMIT",1024)
    with pytest.raises(runner.Refused):
        runner.LocalDocker().call([sys.executable,"-c","import os; os.write(1,b'x'*(16<<20))"],timeout_s=3)
    assert sizes and max(sizes)<=1024

@pytest.mark.parametrize("failure", [None,"nonzero","cleanup"])
def test_lifecycle_uncovered_preserves_counts_without_passed_evidence(valid_run,failure):
    valid_run["stage"]="lifecycle"
    path=Path(valid_run["preflight_path"]);data=json.loads(path.read_text())
    common={"status":"passed","image_id":runner.IMAGE_ID,"source_sha256":data["source_sha256"],"run_id":"prior"}
    mapping=path.parent/"mapping.json"
    mapping.write_text(json.dumps(dict(common,stage="mapping",user_device=0,logical_device=0,physical_device=15)))
    health=path.parent/"health.json";health.write_text(json.dumps(dict(common,stage="health")))
    runtime=path.parent/"runtime.json";runtime.write_text("{}")
    data["inputs"]={"mapping.json":str(mapping),"runtime.json":str(runtime),"health.json":str(health)}
    path.write_text(json.dumps(data))
    docker=FakeDocker("rm_timeout" if failure=="cleanup" else None)
    docker.source_dir=valid_run["source_dir"];docker.input_dir=str(Path(valid_run["run_dir"])/"input")
    class LifecycleProcess:
        sent=False
        def read(self,max_bytes,timeout_s):
            if self.sent:return b""
            self.sent=True
            return json.dumps({"schema":1,"stage":"lifecycle","status":"uncovered","image_id":runner.IMAGE_ID,
                "run_id":docker.run_id,"source_sha256":docker.source_sha,"executed":3,"skipped":1}).encode()+b"\n"
        def poll(self):return (1 if failure=="nonzero" else 0) if self.sent else None
        def terminate(self,timeout_s):pass
    docker.start_attached=lambda cid:LifecycleProcess()
    result=runner.run_stage(**valid_run,docker=docker,clock=docker.clock)
    if failure is None:
        assert result["status"]=="uncovered" and result["exit_code"]==0
        assert result["probe"]["executed"]==3 and result["probe"]["skipped"]==1
    else:assert result["status"]=="failed"
    assert not (Path(valid_run["run_dir"])/"stage-evidence.json").exists()


def test_unknown_device_appearing_before_start_refuses_and_cleans_own_container(valid_run):
    docker=FakeDocker("nonoverlap")
    docker.source_dir=valid_run["source_dir"];docker.input_dir=str(Path(valid_run["run_dir"])/"input")
    original=docker.call
    inventories=0;started=[]
    def call(argv,timeout_s):
        nonlocal inventories
        code,out,err=original(argv,timeout_s)
        if argv[:2]==["docker","inspect"] and argv[-1]=="b"*64:
            inventories+=1
            if inventories==2:
                item=json.loads(out)[0];item["HostConfig"]["Devices"]=[{"PathOnHost":"/dev"}]
                return code,json.dumps([item]),err
        return code,out,err
    docker.call=call
    docker.start_attached=lambda cid:started.append(cid)
    result=runner.run_stage(**valid_run,docker=docker,clock=docker.clock)
    assert result["status"]=="refused" and not started
    assert docker.removed_ids==[CID] and result["cleanup"]["removed"]


def test_source_digest_uses_case_sensitive_relative_posix_order_across_parent_paths(tmp_path):
    # Fixed serialized order is Z.txt, a.txt, a/b.txt, independent of Path comparison.
    records=[("Z.txt",b"upper\n"),("a.txt",b"lower\n"),("a/b.txt",b"nested\n")]
    roots=[tmp_path/"parent-one"/"Checkout",tmp_path/"different"/"deep-parent"/"Source"]
    for root,entries in zip(roots,(records,list(reversed(records)))):
        for relative,content in entries:
            path=root/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(content)
    # Independently fixed from UTF-8 relative name + NUL + raw SHA256(content) records.
    expected="2b01815d6312cb15f7cdb0cec90fce46ef640c7d9225c5fe7e9e0f1cf5e7ff02"
    assert [runner.source_digest(root) for root in roots]==[expected,expected]

def test_bridge_stage_is_separate_fixed_command():
    argv=runner.build_create_argv(image=runner.IMAGE_ID,name='bridge-case',run_id='a',source_dir=str(ROOT),stage='bridge-lifecycle')
    assert 'bridge-lifecycle' in argv and '--device-source' in argv
    assert argv[argv.index('--device-source')+1] == '/work/device-runtime/manifest.json'
    assert not any('privileged' in a or 'network=host' in a for a in argv)

def test_bridge_stage_refuses_unbound_export_before_create(valid_run):
    valid_run['stage']='bridge-lifecycle'
    docker,result=run(None,valid_run)
    assert result['status']=='refused'
    assert not any(call[:2]==['docker','create'] for call in docker.calls)


def test_old_adapter_payload_cannot_satisfy_bridge_result():
    validate=getattr(runner,'validate_bridge_result',None)
    assert callable(validate),'bridge payload identity validation required'
    payload={'status':'passed','executed':3,'failures':0,'errors':0,'skipped':0,'adapter_kind':'old-direct-adapter'}
    with pytest.raises(runner.Refused): validate(payload,{'manifest_sha256':'a'*64})


@pytest.fixture
def bridge_run(valid_run,monkeypatch):
    sys.path.insert(0,str(ROOT/'tests/pool'))
    import device_context_ascend_case as helper
    source=Path(valid_run['source_dir']); export=source/'device-runtime'; package=export/'runtime'; package.mkdir(parents=True)
    raw=b'# synthetic host export\n'; (package/'__init__.py').write_bytes(raw)
    blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    entry=b'100644 __init__.py\0'+bytes.fromhex(blob)
    tree=hashlib.sha1(b'tree '+str(len(entry)).encode()+b'\0'+entry).hexdigest()
    monkeypatch.setattr(helper,'RUNTIME_TREE',tree)
    manifest={'source_repository':'https://github.com/FlagRT/runtime-team','source_commit':helper.RUNTIME_COMMIT,
        'runtime_tree':tree,'count':1,'total_bytes':len(raw),'files':[{'path':'__init__.py','bytes':len(raw),
        'git_blob':blob,'sha256':hashlib.sha256(raw).hexdigest(),'mode':'100644'}]}
    file=export/'manifest.json'; file.write_text(json.dumps(manifest))
    identity={key:manifest[key] for key in ('source_repository','source_commit','runtime_tree')}
    identity['manifest_sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
    path=Path(valid_run['preflight_path']); data=json.loads(path.read_text())
    data['device_runtime']=identity; data['source_sha256']=runner.source_digest(source)
    common={'status':'passed','image_id':runner.IMAGE_ID,'source_sha256':data['source_sha256'],'run_id':'prior'}
    data['inputs']={}
    for name in ('mapping','runtime','health'):
        prior=dict(common,stage=name)
        if name=='mapping': prior.update(user_device=0,logical_device=0,physical_device=15)
        f=path.parent/(name+'.json'); f.write_text(json.dumps(prior)); data['inputs'][name+'.json']=str(f)
    path.write_text(json.dumps(data)); valid_run['stage']='bridge-lifecycle'
    return valid_run,identity


class BridgeDocker(FakeDocker):
    def call(self,argv,timeout_s):
        result=super().call(argv,timeout_s)
        if argv[:2]==['docker','create']:
            self.current['Config'].update(Entrypoint=['python3'],Cmd=argv[argv.index(runner.IMAGE_ID)+1:],
                Env=[argv[i+1] for i,x in enumerate(argv) if x=='--env'])
            if self.mode=='wrong_command': self.current['Config']['Cmd'][2]='lifecycle'
        return result


@pytest.mark.parametrize('mode',[None,'skip','old_adapter','wrong_nodeids','wrong_runtime','wrong_command','never_exits','nonzero','rm_timeout'])
def test_new_bridge_result_and_cleanup_are_bound_to_exact_stage(bridge_run,mode):
    args,identity=bridge_run; docker=BridgeDocker(mode)
    docker.source_dir=args['source_dir']; docker.input_dir=str(Path(args['run_dir'])/'input')
    class Process(FakeProcess):
        def read(self,max_bytes,timeout_s):
            if mode=='never_exits': docker.now+=10; return b''
            if self.sent:return b''
            self.sent=True
            payload=dict(schema=1,stage='bridge-lifecycle',status='uncovered' if mode=='skip' else 'passed',
                image_id=runner.IMAGE_ID,run_id=docker.run_id,source_sha256=docker.source_sha,
                adapter_kind='device-context-ascend-profile',nodeids=list(runner.BRIDGE_NODEIDS),
                device_runtime=dict(identity),executed=3,failures=0,errors=0,skipped=1 if mode=='skip' else 0)
            if mode=='old_adapter':payload['adapter_kind']='old-direct-adapter'
            if mode=='wrong_nodeids':payload['nodeids'][0]='tests/pool/test_pool_ascend_realcard.py::'+runner.EXPECTED_TESTS[0]
            if mode=='wrong_runtime':payload['device_runtime']['manifest_sha256']='0'*64
            return json.dumps(payload).encode()+b'\n'
    started=[]
    def start(cid):started.append(cid);return Process(docker)
    docker.start_attached=start
    result=runner.run_stage(**args,docker=docker,clock=docker.clock)
    assert result['status']==('passed' if mode is None else 'uncovered' if mode=='skip' else 'refused' if mode in ('old_adapter','wrong_nodeids','wrong_runtime','wrong_command') else 'failed')
    if mode=='wrong_command':assert not started
    if mode!='rm_timeout':assert docker.removed_ids==[CID]
    assert (Path(args['run_dir'])/'stage-evidence.json').exists() == (mode is None)
