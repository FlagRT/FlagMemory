#!/usr/bin/env python3
"""Fixed local Ascend stages with external deadlines and exact-ID ownership cleanup."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid

IMAGE_ID="sha256:e13a15d0d02b86e9cd7b1656df3c6b52838c258f270b6b894a91c2b074da5f15"
NODES=("davinci15","davinci_manager","devmm_svm","hisi_hdc")
OUTPUT_LIMIT=100<<20
CONTROL_LIMIT=1<<20
STAGES=("metadata","mapping","health","lifecycle")
ID=re.compile(r"^[0-9a-f]{64}$")
EXPECTED_TESTS=(
    "test_actual_tensor_and_cross_stream_dependency",
    "test_completed_reuse_rejects_old_lease",
    "test_pending_borrower_protection_when_observed",
)

class Refused(RuntimeError): pass

def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(65536),b""): h.update(block)
    return h.hexdigest()

def source_digest(path):
    root=Path(path).resolve()
    if not root.is_dir() or (root/".git").exists():
        raise Refused("source must be an independent frozen directory without .git")
    h=hashlib.sha256()
    for file in sorted(root.rglob("*"), key=lambda file:file.relative_to(root).as_posix()):
        if file.is_symlink(): raise Refused("source symlinks are not allowed")
        if file.is_file() and not any(part in ("__pycache__",".pytest_cache") for part in file.parts):
            h.update(file.relative_to(root).as_posix().encode()); h.update(b"\0")
            h.update(bytes.fromhex(digest_file(file)))
    return h.hexdigest()

def load_json(path):
    file=Path(path)
    if not file.is_absolute() or not file.is_file() or file.stat().st_size>CONTROL_LIMIT:
        raise Refused("absolute bounded JSON evidence file required")
    try:
        value=json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(value,dict): raise ValueError("object required")
        return value
    except (ValueError,OSError) as exc: raise Refused("invalid evidence: "+str(exc)) from exc

def build_create_argv(*,image,name,run_id,source_dir,stage):
    if image!=IMAGE_ID or stage not in STAGES or not re.fullmatch(r"[a-zA-Z0-9_.-]+",name):
        raise Refused("fixed image, stage and safe unique name required")
    source=Path(source_dir)
    if not source.is_absolute() or "," in str(source) or "\n" in str(source):
        raise Refused("absolute mount source without Docker mount separators required")
    argv=["docker","create","--name",name,"--label","flagmemory.run_id="+run_id,
          "--runtime=runc","--user=0:0","--cap-drop=ALL","--security-opt=no-new-privileges",
          "--network=none","--pull=never","--cpus=2","--memory=4g","--ipc=private",
          "--restart=no","--log-driver=none","--entrypoint=python3","--workdir=/work",
          "--env","FLAGMEMORY_RUN_ID="+run_id,
          "--mount","type=bind,src="+str(source)+",dst=/work,readonly"]
    if stage!="metadata":
        for node in NODES: argv+=["--device","/dev/"+node+":/dev/"+node+":rw"]
        argv+=["--mount","type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver,readonly"]
    command=["-B","/work/tools/ascend_probe.py"]
    if stage=="metadata":
        command+=["collect-abi","--output-dir","/tmp/acl-evidence"]
    elif stage=="mapping":
        command+=["mapping","--evidence","/input/reviewed-abi.json","--expected-physical","15"]
    elif stage=="health":
        command+=["health","--mapping","/input/mapping.json","--runtime-evidence","/input/runtime.json",
                  "--byte-limit",str(16<<20),"--timeout-s","30"]
    else:
        command+=["lifecycle","--mapping","/input/mapping.json","--runtime-evidence","/input/runtime.json",
                  "--health","/input/health.json"]
    return argv+[image]+command

class Attached:
    """Bounded reader queue; subprocess pipes cannot deadlock or grow unbounded."""
    def __init__(self,process):
        self.process=process; self.parts=queue.Queue(maxsize=4); self.eof=False
        self.cancel=threading.Event()
        def reader():
            try:
                while not self.cancel.is_set():
                    data=process.stdout.read(65536)
                    if not data: break
                    while not self.cancel.is_set():
                        try: self.parts.put(data,timeout=.1); break
                        except queue.Full: pass
            finally:
                while not self.cancel.is_set():
                    try: self.parts.put(None,timeout=.1); break
                    except queue.Full: pass
        self.thread=threading.Thread(target=reader,daemon=True); self.thread.start()
    def read(self,max_bytes,timeout_s):
        if self.eof:return b""
        try: data=self.parts.get(timeout=timeout_s)
        except queue.Empty:return b""
        if data is None:self.eof=True; return b""
        # Runner requests >=64KiB while work remains. Never silently truncate a block.
        return data
    def poll(self):
        return self.process.poll() if self.eof else None
    def terminate(self,timeout_s):
        self.cancel.set()
        if self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=max(.01,timeout_s/2))
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait(timeout=max(.01,timeout_s/2))
        self.process.stdout.close()
        self.thread.join(timeout=min(.2,timeout_s))

class OutputLimit(RuntimeError): pass

def capture_bounded(argv,*,timeout_s,limit):
    """Backpressure the child and cap every parent write, including temporary output."""
    deadline=time.monotonic()+timeout_s
    with tempfile.TemporaryFile() as output:
        process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,shell=False,bufsize=0)
        attached=Attached(process)
        captured=0
        try:
            while True:
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError("control command deadline")
                block=attached.read(65536,timeout_s=min(.1,remaining))
                available=limit-captured
                if block:
                    kept=block[:available]
                    output.write(kept)
                    captured+=len(kept)
                    if len(block)>available:raise OutputLimit("output limit")
                code=attached.poll()
                if code is not None:break
            output.seek(0)
            return code,output.read(limit)
        finally:
            attached.terminate(timeout_s=min(2,max(.1,timeout_s)))

class LocalDocker:
    def call(self,argv,timeout_s):
        try:
            code,output=capture_bounded(argv,timeout_s=timeout_s,limit=CONTROL_LIMIT)
        except OutputLimit as exc:
            raise Refused("control output limit") from exc
        return code,output.decode("utf-8",errors="replace"),""
    def start_attached(self,container_id):
        return Attached(subprocess.Popen(["docker","start","--attach",container_id],
                        stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                        shell=False,bufsize=0))

def parse_target_snapshot(text):
    """Select physical 15 from the fixed paired-row npu-smi info format."""
    if "Phy-ID" not in text or "HBM-Usage(MB)" not in text:
        raise Refused("unrecognized device table")
    # Locate the process header independently of whitespace alignment.
    lines=text.splitlines()
    process_at=next((i for i,line in enumerate(lines) if "Process id" in line and "Process name" in line),None)
    if process_at is None:raise Refused("process table missing")
    candidates=[]; previous=None
    for line in lines[:process_at]:
        if not line.strip().startswith("|"):continue
        cells=[part.strip() for part in line.strip().strip("|").split("|")]
        if len(cells)<3:continue
        words=cells[0].split()
        if len(words)==2 and words[0].isdigit() and not words[1].isdigit():
            previous=(int(words[0]),cells[1])
        elif len(words)==2 and all(word.isdigit() for word in words):
            chip,physical=map(int,words)
            if physical!=15:continue
            if previous is None:raise Refused("target row has no matching NPU header")
            npu,health=previous
            matches=re.findall(r"(\d+)\s*/\s*(\d+)",cells[-1])
            core=re.match(r"(\d+)\s+",cells[-1])
            if len(matches)!=2 or core is None or not re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]",cells[1]):
                raise Refused("unrecognized target metrics")
            used,total=map(int,matches[-1])
            candidates.append({"physical":physical,"npu":npu,"chip":chip,"bus":cells[1].lower(),
                               "health":health,"hbm_used_mb":used,"hbm_total_mb":total,
                               "aicore":int(core.group(1)),"pids":[]})
    if len(candidates)!=1:raise Refused("target physical identity is absent or ambiguous")
    result=candidates[0]; observed=False; no_process=False
    for line in lines[process_at+1:]:
        empty=re.search(r"No running processes found in NPU\s+(\d+)",line)
        if empty and int(empty.group(1))==result["npu"]:
            observed=True;no_process=True
        if not line.strip().startswith("|"):continue
        cells=[part.strip() for part in line.strip().strip("|").split("|")]
        words=cells[0].split()
        if (words and words[0]==str(result["npu"])
            and not (len(words)==2 and all(word.isdigit() for word in words))):
            raise Refused("unrecognized target NPU process row")
        if len(words)==2 and all(word.isdigit() for word in words):
            npu,chip=map(int,words)
            if len(cells)!=5 or not cells[1].isdigit() or not cells[3].isdigit():
                raise Refused("unrecognized process row")
            if npu==result["npu"]:
                observed=True
                if no_process:raise Refused("contradictory target process rows")
                if chip==result["chip"]:result["pids"].append(int(cells[1]))
    if not observed:raise Refused("target process inventory is missing")
    result["pids"].sort()
    return result

def compare_target_snapshot(reference,current,*,tolerance_mb):
    if type(tolerance_mb) is not int or not 0<=tolerance_mb<=1024:
        raise Refused("explicit target HBM observation tolerance within 1 GiB required")
    keys=("physical","npu","chip","bus","hbm_total_mb")
    if not isinstance(reference,dict) or any(reference.get(key)!=current.get(key) for key in keys):
        raise Refused("target identity or memory capacity changed")
    if reference.get("physical")!=15 or current.get("health")!="OK" or reference.get("health")!="OK":
        raise Refused("target health is not OK")
    if current.get("pids") or reference.get("pids") or current.get("aicore")!=0:
        raise Refused("target has a compute process or nonzero utilization")
    used=current.get("hbm_used_mb"); baseline=reference.get("hbm_used_mb"); total=current.get("hbm_total_mb")
    if (type(used) is not int or type(baseline) is not int or type(total) is not int
        or not 0<=used<=total or not 0<=baseline<=total or abs(used-baseline)>tolerance_mb):
        raise Refused("target HBM changed beyond the reviewed observation tolerance")

def validate_preflight(data,stage,source_sha,now):
    if data.get("schema")!=1 or data.get("image_id")!=IMAGE_ID or data.get("source_sha256")!=source_sha:
        raise Refused("preflight image or frozen source mismatch")
    if stage=="metadata":return
    try:
        age=(now-datetime.fromisoformat(data["collected_at"])).total_seconds()
    except (KeyError,ValueError,TypeError):raise Refused("dated preflight required")
    if not 0<=age<=300 or data.get("window_confirmed") is not True:
        raise Refused("a confirmed preflight from the last five minutes is required")
    if data.get("window_mode") not in ("exclusive","shared_candidate"):
        raise Refused("explicit exclusive or shared_candidate window required")
    if data.get("target")!={"physical":15,"major":235,"minor":15,"healthy":True,"process_count":0}:
        raise Refused("approved healthy unoccupied physical target required")
    proof=data.get("filter",{})
    if (proof.get("version") not in ("v1","v2") or proof.get("review_status")!="accepted"
        or not proof.get("reviewer") or proof.get("allowed_compute")!=[15]
        or proof.get("controls")!=list(NODES[1:])):
        raise Refused("reviewed effective device filter evidence required")
    proof_file=Path(proof.get("evidence_path",""))
    if not proof_file.is_absolute() or not proof_file.is_file() or digest_file(proof_file)!=proof.get("sha256"):
        raise Refused("effective filter evidence hash mismatch")
    compare_target_snapshot(data.get("target_snapshot"),data.get("target_snapshot",{}),
                            tolerance_mb=data.get("hbm_tolerance_mb"))

def no_conflicting_container(item,mode):
    host=item.get("HostConfig",{})
    if host.get("Privileged") or host.get("DeviceCgroupRules") or host.get("DeviceRequests"):
        raise Refused("another container has broad device access")
    if any(m.get("Source") in ("/","/dev") for m in item.get("Mounts",[])):
        raise Refused("another container has a broad host mount")
    devices=host.get("Devices") or []
    if not isinstance(devices,list):
        raise Refused("another container has unrecognized device authorization")
    computes=[]
    controls={"/dev/"+node for node in NODES[1:]}
    for device in devices:
        path=device.get("PathOnHost") if isinstance(device,dict) else None
        if not isinstance(path,str):
            raise Refused("another container has unrecognized device authorization")
        if re.fullmatch(r"/dev/davinci(?:[0-9]|1[0-5])",path):
            computes.append(path)
        elif path not in controls:
            # Docker can recursively authorize directories; unknown paths are not safe emptiness.
            raise Refused("another container has directory, alias or unknown device authorization")
    if "/dev/davinci15" in computes or (mode=="exclusive" and computes):
        raise Refused("overlapping compute authorization or exclusive window conflict")

def verify_created(item,run_id,source,input_dir,stage):
    if item.get("Image")!=IMAGE_ID or item.get("Config",{}).get("Labels",{}).get("flagmemory.run_id")!=run_id:
        raise Refused("created container identity or owner label mismatch")
    host=item.get("HostConfig",{})
    if (host.get("Privileged") or host.get("Runtime")!="runc" or host.get("NetworkMode")!="none"
        or host.get("NanoCpus")!=2000000000 or host.get("Memory")!=4<<30
        or host.get("IpcMode")!="private" or host.get("PidMode") not in ("",None,"private")
        or host.get("RestartPolicy",{}).get("Name")!="no"
        or host.get("LogConfig",{}).get("Type")!="none"
        or {s.lower() for s in host.get("CapDrop",[])}!={"all"}
        or "no-new-privileges" not in host.get("SecurityOpt",[])
        or host.get("DeviceCgroupRules") or host.get("DeviceRequests")):
        raise Refused("actual container security or resource configuration differs")
    expected=[] if stage=="metadata" else [("/dev/"+n,"/dev/"+n,"rw") for n in NODES]
    actual=[(d.get("PathOnHost"),d.get("PathInContainer"),d.get("CgroupPermissions"))
            for d in (host.get("Devices") or [])]
    if sorted(actual)!=sorted(expected):raise Refused("actual device whitelist differs")
    mounts=[(m.get("Source"),m.get("Destination"),m.get("RW")) for m in item.get("Mounts",[])]
    wanted=[(str(source),"/work",False),(str(input_dir),"/input",False)]
    if stage!="metadata":wanted.append(("/usr/local/Ascend/driver","/usr/local/Ascend/driver",False))
    if sorted(mounts)!=sorted(wanted):raise Refused("actual read-only mounts differ")

def run_stage(*,stage,run_dir,source_dir,image,preflight_path,docker,clock=time.monotonic):
    start=clock(); total=300 if stage=="lifecycle" else 60
    deadline=start+total; work_deadline=deadline-(20 if stage=="lifecycle" else 15)
    run_id=uuid.uuid4().hex; cid=None; attached=None; owner=False; root=None
    result={"schema":1,"stage":stage,"status":"refused","run_id":run_id,"image_id":image,
            "started_at":datetime.now(timezone.utc).isoformat(),"source_sha256":None,
            "captured_bytes":0,"cleanup":{"ownership_verified":False,"removed":False,"errors":[]},
            "target_postcheck_required":stage!="metadata"}
    def call(argv):
        remaining=deadline-clock()
        if remaining<=0:raise TimeoutError("total deadline exhausted")
        code,out,err=docker.call(argv,timeout_s=min(10,remaining))
        if len(out.encode())+len(err.encode())>CONTROL_LIMIT:raise Refused("control output limit")
        return code,out,err
    def inspect(container):
        code,out,err=call(["docker","inspect",container])
        if code:raise RuntimeError("inspect failed: "+err+out)
        values=json.loads(out)
        if len(values)!=1 or values[0].get("Id")!=container:raise Refused("inspect ID mismatch")
        return values[0]
    refresh_count=0
    def refresh(data):
        nonlocal refresh_count
        code,out,err=call(["docker","ps","--quiet","--no-trunc"])
        if code:raise RuntimeError("container inventory failed")
        for other in out.split():
            if other==cid:continue
            if not ID.fullmatch(other):raise Refused("inventory did not return complete IDs")
            no_conflicting_container(inspect(other),data["window_mode"])
        code,out,err=call(["npu-smi","info"])
        if code:raise Refused("host target snapshot unavailable")
        refresh_count+=1
        (root/("pre-create-npu-smi.txt" if refresh_count==1 else "pre-start-npu-smi.txt")).write_text(out,encoding="utf-8")
        facts=parse_target_snapshot(out)
        compare_target_snapshot(data["target_snapshot"],facts,tolerance_mb=data["hbm_tolerance_mb"])
    try:
        if image!=IMAGE_ID or stage not in STAGES:raise Refused("fixed image and stage required")
        source=Path(source_dir); target=Path(run_dir)
        if not source.is_absolute() or not target.is_absolute() or "," in str(target):raise Refused("absolute directories without mount separators required")
        source=source.resolve(); target=target.resolve()
        if target.exists() or source==target or source in target.parents or target in source.parents:
            raise Refused("new run directory must not overlap frozen source")
        source_sha=source_digest(source); result["source_sha256"]=source_sha
        data=load_json(preflight_path)
        validate_preflight(data,stage,source_sha,datetime.now(timezone.utc))
        target.mkdir(parents=False); root=target
        if hasattr(os,"getuid") and target.stat().st_uid!=os.getuid():raise Refused("run directory owner mismatch")
        input_dir=target/"input"; input_dir.mkdir()
        inputs=data.get("inputs",{})
        required={"metadata":set(),"mapping":{"reviewed-abi.json"},
                  "health":{"mapping.json","runtime.json"},
                  "lifecycle":{"mapping.json","runtime.json","health.json"}}[stage]
        if not required<=inputs.keys():raise Refused("stage evidence bundle is incomplete")
        allowed={"reviewed-abi.json","mapping.json","runtime.json","health.json"}
        copied=0
        for name,value in inputs.items():
            if name not in allowed:raise Refused("unknown input evidence filename")
            file=Path(value)
            if not file.is_absolute() or not file.is_file():raise Refused("absolute evidence input required")
            copied+=file.stat().st_size
            if copied>CONTROL_LIMIT:raise Refused("input evidence exceeds limit")
            shutil.copyfile(file,input_dir/name)
        for name in required & {"mapping.json","health.json"}:
            prior=load_json(str(input_dir/name))
            if (prior.get("status")!="passed" or prior.get("image_id")!=IMAGE_ID
                or prior.get("source_sha256")!=source_sha or not prior.get("run_id")):
                raise Refused("prior stage did not pass for this image and source")
            if name=="mapping.json" and (prior.get("physical_device")!=15 or prior.get("user_device")!=0
                                        or prior.get("logical_device")!=0):
                raise Refused("prior mapping does not identify the target")
        name="flagmemory-"+stage+"-"+run_id
        code,out,err=call(["docker","ps","--all","--quiet","--no-trunc","--filter","name=^/"+name+"$"])
        if code or out.strip():raise Refused("container name is already present or unavailable")
        if stage!="metadata":refresh(data)
        argv=build_create_argv(image=image,name=name,run_id=run_id,source_dir=str(source),stage=stage)
        index=argv.index(image)
        extra=["--mount","type=bind,src="+str(input_dir)+",dst=/input,readonly",
               "--env","FLAGMEMORY_SOURCE_SHA256="+source_sha]
        argv[index:index]=extra
        if stage=="metadata":
            static=data.get("metadata",{})
            headers=static.get("headers",[]); library=static.get("library")
            if not headers or not library or not all(isinstance(p,str) and p.startswith("/") for p in [*headers,library]):
                raise Refused("explicit absolute target header/library paths required")
            for header in headers:argv+=["--header",header]
            argv+=["--library",library]
        code,out,err=call(argv)
        if code:raise RuntimeError("create failed: "+err+out)
        candidate=out.strip()
        if not ID.fullmatch(candidate):raise Refused("create did not return a complete ID; manual check required")
        cid=candidate; result["container_id"]=cid
        item=inspect(cid)
        owner=item.get("Config",{}).get("Labels",{}).get("flagmemory.run_id")==run_id
        if not owner:raise Refused("created owner label mismatch")
        verify_created(item,run_id,source,input_dir,stage)
        (target/"inspect.json").write_text(json.dumps(item,indent=2),encoding="utf-8")
        if stage!="metadata":refresh(data)
        # No arbitrary shell, command, Docker option, device, image or retry is accepted.
        attached=docker.start_attached(cid)
        log_limit=OUTPUT_LIMIT-min(8<<20,OUTPUT_LIMIT//4)
        with (target/"output.log").open("xb") as log:
            while True:
                if clock()>=work_deadline:raise TimeoutError("deadline")
                block=attached.read(65536,timeout_s=min(.1,max(.001,work_deadline-clock())))
                available=log_limit-result["captured_bytes"]
                log.write(block[:max(0,available)])
                result["captured_bytes"]+=min(len(block),max(0,available))
                if len(block)>=available:raise RuntimeError("output_limit")
                exit_code=attached.poll()
                if exit_code is not None:break
        if exit_code!=0:raise RuntimeError("container_exit")
        text=(target/"output.log").read_bytes()[-CONTROL_LIMIT:].decode("utf-8",errors="replace")
        payload=None
        for line in reversed(text.splitlines()):
            try:
                candidate=json.loads(line)
                if isinstance(candidate,dict):payload=candidate;break
            except ValueError:pass
        allowed_status=(("uncovered",) if stage=="metadata" else
                        ("passed","uncovered") if stage=="lifecycle" else ("passed",))
        expected_stage="metadata" if stage=="metadata" else stage
        if (not payload or payload.get("status") not in allowed_status or payload.get("stage")!=expected_stage
            or payload.get("run_id")!=run_id or payload.get("image_id")!=image
            or payload.get("source_sha256")!=source_sha):
            raise RuntimeError("valid matching stage result missing")
        result["probe"]=payload; result["status"]=payload["status"]; result["reason"]="stage completed"
    except Refused as exc:
        result.update(status="refused",reason=str(exc))
    except Exception as exc:
        reason="deadline" if isinstance(exc,TimeoutError) else str(exc)
        result.update(status="failed",reason=reason)
    finally:
        cleanup=result["cleanup"]
        if attached is not None:
            try:attached.terminate(timeout_s=max(.01,min(2,deadline-clock())))
            except Exception as exc:cleanup["errors"].append("attached CLI: "+str(exc))
        if cid is not None:
            try:
                current=inspect(cid)
                owner=current.get("Config",{}).get("Labels",{}).get("flagmemory.run_id")==run_id
                cleanup["ownership_verified"]=owner
                if not owner:raise Refused("ownership changed; no cleanup attempted")
                code,out,err=call(["docker","stop","--time","1",cid])
                if code:
                    cleanup["errors"].append("stop: "+err+out)
                    code,out,err=call(["docker","kill",cid])
                    if code:cleanup["errors"].append("kill: "+err+out)
                code,out,err=call(["docker","rm","--force",cid])
                if code:raise RuntimeError("remove failed: "+err+out)
                code,out,err=call(["docker","inspect",cid])
                if code!=0 and ("No such" in out+err or "not found" in out+err):
                    cleanup["removed"]=True
                else:raise RuntimeError("container removal is not confirmed")
            except Exception as exc:cleanup["errors"].append(str(exc))
        if cid is not None and not cleanup["removed"]:
            result["status"]="failed"
        if cleanup["errors"]:result["status"]="failed"
        if root is not None:
            try:
                code,out,err=call(["docker","info","--format","{{json .ServerVersion}}"])
                cleanup["docker_queryable"]=code==0
                if code:raise RuntimeError("Docker is not queryable after cleanup")
            except Exception as exc:
                cleanup["docker_queryable"]=False;cleanup["errors"].append(str(exc));result["status"]="failed"
            if stage!="metadata" and cid is not None:
                try:
                    code,out,err=call(["npu-smi","info"])
                    if code:raise RuntimeError("target postcheck unavailable")
                    (root/"post-npu-smi.txt").write_text(out,encoding="utf-8")
                    cleanup["post_snapshot_sha256"]=hashlib.sha256(out.encode()).hexdigest()
                    facts=parse_target_snapshot(out)
                    cleanup["target_facts"]=facts
                    try:
                        compare_target_snapshot(data["target_snapshot"],facts,tolerance_mb=data["hbm_tolerance_mb"])
                        cleanup["target_facts_match"]=True
                        cleanup["postcheck_review_required"]=False
                    except Refused as exc:
                        cleanup["target_facts_match"]=False
                        cleanup["postcheck_review_required"]=True
                        if result["status"]=="passed":
                            result.update(status="uncovered",reason="target_postcheck_requires_review: "+str(exc))
                except Exception as exc:
                    cleanup["errors"].append("target postcheck: "+str(exc));result["status"]="failed"
            if result["status"]=="passed" and cleanup["removed"] and cleanup.get("docker_queryable"):
                (root/"stage-evidence.json").write_text(json.dumps(result["probe"],indent=2),encoding="utf-8")
            result.update(elapsed_s=clock()-start,
                          exit_code=0 if result["status"] in ("passed","uncovered") else (2 if result["status"]=="refused" else 1))
            (root/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        else:
            result.update(elapsed_s=clock()-start,exit_code=2 if result["status"]=="refused" else 1)
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",required=True,choices=STAGES)
    parser.add_argument("--run-dir",required=True)
    parser.add_argument("--source-dir",required=True)
    parser.add_argument("--preflight",required=True)
    args=parser.parse_args(argv)
    result=run_stage(stage=args.stage,run_dir=args.run_dir,source_dir=args.source_dir,
                     image=IMAGE_ID,preflight_path=args.preflight,docker=LocalDocker())
    print(json.dumps(result,sort_keys=True))
    return result["exit_code"]

if __name__=="__main__":raise SystemExit(main())
