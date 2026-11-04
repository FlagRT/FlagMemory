# Bounded Ascend preparation

Run fixed stages within an authorized scope and a confirmed window. A shared
candidate permits other containers with disjoint compute-device authorization.
Overlapping or broad device access, busy initialization, unknown mapping, or
unconfirmed cleanup stops progression. Preserve other users' containers and work.

The executable configuration is in [the host executor](../../tools/run_ascend_bounded.py).
It fixes the image, one compute node plus required control nodes, ordinary
runtime, private namespaces, read-only driver/source/input mounts, no network,
no added capabilities, two CPUs and four GiB of host memory. No arbitrary command,
extra Docker option, device reset, installation, image pull, remote operation,
automatic retry or automatic next stage is provided.

## Stages and results

1. metadata reads explicit headers and library symbols without loading ACL,
   importing a framework or initializing a device. It reports unreviewed static
   material; static collection is never mapping approval.
2. mapping validates accepted ABI evidence and current file hashes before loading
   the library. A separate short process initializes, checks the unique user
   device, checks user equals logic, queries the target physical identity and
   finalizes. Health uses a new process.
3. health validates matching mapping and reviewed exact-source event/stream
   evidence, allocates 4096 uint8 bytes on the explicit stream, checks the actual
   tensor identity, writes and reads, polls a recorded event to a deadline and
   releases after confirmed completion. Device and stream context restoration is
   checked. Uncertain completion keeps the allocation held until process teardown.
4. lifecycle validates mapping, health and runtime evidence, collects the exact
   three existing cases below and runs only their standalone Ascend file:

- test_actual_tensor_and_cross_stream_dependency
- test_completed_reuse_rejects_old_lease
- test_pending_borrower_protection_when_observed

A skipped pending observation is uncovered; the host retains executed/skipped
counts and does not produce passed stage evidence. Missing cases, nonzero child exit,
mismatched evidence or cleanup failure cannot pass. The legacy CUDA file and
broad real-card selection are excluded.

Health and lifecycle explicit cumulative allocation budgets are 16 MiB and
64 MiB, totaling 80 MiB. Release does not refund a cumulative budget. Framework
caches/workspaces are observed separately. Host deadlines are 60 seconds for
metadata/mapping/health and 300 seconds for lifecycle, with reserved cleanup time.
A Python polling deadline cannot interrupt a stuck C call.

The host terminates only its attached CLI and verified full container ID. It
rechecks ownership, bounds stop/remove calls, confirms absence, queries Docker
and records the target postcheck. Unconfirmed cleanup remains a failure. No
prefix/name cleanup or automatic reset is provided. Output and evidence total
at most 100 MiB; control commands and streamed output are independently bounded. Control and symbol
subprocesses write to a bounded pipe; only the parent writes retained temporary
output after deducting its byte budget, so the file cannot overshoot the limit.

## Preflight and input bundle

Use an independent frozen source directory without Git metadata or symlinks.
Keep it and the new evidence directory outside shared checkouts and disjoint.
The source digest is source_digest(source_dir) in the executor.

Preflight JSON fields are:

| Field | Meaning |
|---|---|
| schema, image_id, source_sha256 | Schema 1; matching fixed executable image and frozen source |
| collected_at, window_confirmed, window_mode | Timezone-aware collection within five minutes, explicit confirmation, exclusive or shared_candidate |
| target | Reviewed physical number, node major/minor, healthy state and zero target process count |
| target_snapshot | Normalized physical/npu/chip/bus identity, health, HBM used/total MB, target pids and aicore, obtained from the fixed host table |
| hbm_tolerance_mb | Reviewed target HBM observation tolerance, from zero through 1024 MiB; this is not a hard framework allocation quota |
| filter | cgroup version v1/v2, accepted review_status, named reviewer, absolute evidence_path, its sha256, exact allowed_compute and controls |
| inputs | Stage-specific evidence file paths under the fixed names below |
| metadata | Metadata only: explicit absolute target-image headers list and library path |

Before create/start, the executor refreshes actual container authorization and
target device facts. Directory, alias and unknown device authorization paths
in another container are refused rather than treated as an empty safe set.
Raw tables are retained. Unrelated devices, timestamps,
temperature and power changes do not reject the target. Target identity/health,
new target PIDs, nonzero target compute utilization or HBM beyond the reviewed
tolerance stop progression. Unknown table formats are refused. The postcheck
uses the same target facts. A relevant change is uncovered pending review;
a failed command or unparseable postcheck is a failure.

For cgroup v1 retain the effective device allow-list; for v2 retain the effective
device-filter/OCI enforcement basis. Absence of a v1 devices.list on v2 is not
itself a failure. Enumeration or an owner label alone is insufficient. Preflight
is reviewed evidence, not a reservation or adversarial driver isolation proof.

The inputs object allows only reviewed-abi.json, runtime.json, mapping.json and
health.json. Mapping requires the ABI file; health requires mapping and runtime;
lifecycle requires mapping, runtime and health. Copy successful stage-evidence.json
to the next bundle's mapping/health name. Stage evidence is emitted only after
matching result identity, confirmed self-cleanup and a successful target postcheck.
Read result.json, inspect.json, output.log and raw host snapshots before proceeding.

ABI evidence contains schema/image, resolved runtime library path/hash, absolute
header hashes, the probe's exact SIGNATURES table, the five exported names,
init_required=true, init_argument=null, named reviewer/time and accepted status.
The physical argument policy is identity-only-user-equals-logic. Official
descriptions of the old physical-query argument differ; equality avoids choosing
between them. Remapped numbering is refused before physical query. Development
stub libraries/search paths are refused. Never invent an accepted review.

Runtime evidence contains schema/image, named reviewer/time, accepted status,
installed metadata/header hashes in source_sha256s, torch_npu_git_version and
official fixed-commit contract_source links. Its event_query_contract is
polls-status-without-waiting-for-device-completion; stream_contract is
explicit-device-and-restoring-stream-context. Query acceptance is a source
contract, not wait-free or hard real-time assurance. Keep runtime_query_observed
and realcard_passed separate. Initial health does not require a previous health
success.

## Local usage

Help requires no Docker/device:

~~~text
python -B tools/ascend_probe.py --help
python -B tools/run_ascend_bounded.py --help
~~~

The executor requires --stage, --run-dir, --source-dir and --preflight with
reviewed local paths. The run directory must not exist or overlap source.
It executes locally on the selected host and provides no SSH wrapper.

Standalone static collection uses collect-abi --header (repeatable), --library
and --output-dir. All paths are absolute; output must be new. Existing readelf
or nm is required; missing tools are refused rather than installed.

Framework-free collection of the existing real cases is:

~~~text
python -B -m pytest tests/pool/test_pool_ascend_realcard.py --collect-only -q -o addopts= -m realcard
~~~

Collection must list exactly the three names above without importing torch,
torch_npu or vLLM. CPU tests use injected Docker/device behavior. Their success
does not establish actual driver mapping, device behavior or real lifecycle
coverage.
