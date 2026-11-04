"""Explicitly gated test-only OneRec input consumption; never a production adapter."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import pytest

MODEL = 'OpenOneRec/OneRec-8B'
REVISION = '29f95b3da63a4bbc00a1cd3aeee948efb2ff40ff'
pytestmark = pytest.mark.realcard


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _check_readonly(path):
    try:
        flags = os.statvfs(path).f_flag
        readonly = os.ST_RDONLY
    except (AttributeError, OSError) as exc:
        raise ValueError('read-only model mount cannot be verified') from exc
    if not flags & readonly:
        raise ValueError('read-only model mount required')


def _validate_stage(item, stage, config):
    path = Path(item.get('path', ''))
    if not path.is_file() or path.stat().st_size > 1048576 or _hash(path) != item.get('sha256'):
        raise ValueError(stage + ' evidence identity mismatch')
    try:
        data = json.loads(path.read_text(encoding='utf8'))
    except (ValueError, OSError) as exc:
        raise ValueError(stage + ' evidence invalid') from exc
    if (data.get('stage') != stage or data.get('status') != 'passed'
            or data.get('image_id') != config['image_id']
            or data.get('source_sha256') != config['probe_source_sha256']
            or not data.get('run_id')):
        raise ValueError(stage + ' evidence does not match')
    if stage == 'mapping' and (data.get('user_device') != config['device']
            or data.get('logical_device') != config['device']
            or data.get('physical_device') != config['physical_device']):
        raise ValueError('mapping target mismatch')
    if stage == 'health' and (data.get('stream_device') != config['device'] or data.get('readback_matches') is not True or data.get('cleanup', {}).get('completion_confirmed') is not True or data.get('cleanup', {}).get('released') is not True):
        raise ValueError('health evidence incomplete')
    return data


def validate_model_gate(config):
    if config.get('weights_confirmed') is not True:
        raise ValueError('weight identity is not confirmed')
    if config.get('model_id') != MODEL or config.get('revision') != REVISION:
        raise ValueError('model identity mismatch')
    if config.get('resource_scope') != 'onerec-single-forward' or config.get('resource_approved') is not True:
        raise ValueError('resource approval missing')
    resources = ('host_memory_bytes', 'device_observation_limit_bytes', 'timeout_s')
    for key in resources:
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('invalid resource ' + key)
        if config.get('approved_resources', {}).get(key) != value:
            raise ValueError('resource differs from approval')
    if config['host_memory_bytes'] <= 4 << 30:
        raise ValueError('resource host limit too small')
    for key in ('atol', 'rtol'):
        v = config.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError('invalid tolerance')
    try:
        expires = datetime.datetime.fromisoformat(config['expires_at'].replace('Z', '+00:00'))
        if expires.tzinfo is None or expires <= datetime.datetime.now(datetime.timezone.utc):
            raise ValueError('window expired')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('window missing') from exc
    if not isinstance(config.get('device'), int) or isinstance(config['device'], bool) or config['device'] < 0:
        raise ValueError('invalid mapping device')
    if config.get('model_mount_read_only') is not True or config.get('external_watchdog_confirmed') is not True:
        raise ValueError('resource execution controls missing')
    for key in ('image_id', 'source_commit', 'torch_version', 'torch_npu_version', 'transformers_version'):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError('missing identity ' + key)
    if (not isinstance(config.get('physical_device'), int) or isinstance(config['physical_device'], bool)
            or config['physical_device'] < 0 or not config.get('probe_source_sha256')):
        raise ValueError('mapping identity missing')
    for key in ('mapping', 'health'):
        _validate_stage(config.get(key, {}), key, config)
    root = Path(config.get('model_path', ''))
    files = config.get('files')
    if not root.is_dir() or not isinstance(files, dict) or not files:
        raise ValueError('weight path or identity manifest missing')
    required = {'config.json', 'tokenizer.json', 'model.safetensors.index.json'}
    if not required.issubset(files) or not any(name.endswith('.safetensors') for name in files):
        raise ValueError('weight manifest incomplete')
    root = root.resolve()
    _check_readonly(root)
    present = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('weight symlinks are not permitted')
        if path.is_file():
            present.add(path.relative_to(root).as_posix())
    if present != set(files):
        raise ValueError('weight manifest does not cover the model directory')
    for name, digest in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or _hash(path) != digest:
            raise ValueError('weight identity mismatch: ' + name)
    index = json.loads((root / 'model.safetensors.index.json').read_text(encoding='utf8'))
    shards = set(index.get('weight_map', {}).values())
    if not shards or not shards.issubset(files):
        raise ValueError('weight shards missing from manifest')
    if expires <= datetime.datetime.now(datetime.timezone.utc):
        raise ValueError('window expired while validating model files')
    return config


def compare_forward(baseline, pooled, *, atol, rtol):
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0 for x in (atol, rtol)):
        raise ValueError('invalid tolerance')
    a, b = list(baseline), list(pooled)
    if not a or len(a) != len(b) or any(not math.isfinite(float(x)) for x in a + b):
        raise ValueError('invalid logits')
    if max(range(len(a)), key=a.__getitem__) != max(range(len(b)), key=b.__getitem__):
        raise ValueError('argmax mismatch')
    differences = [abs(float(x)-float(y)) for x, y in zip(a, b)]
    if any(d > atol + rtol * abs(float(x)) for d, x in zip(differences, a)):
        raise ValueError('logits mismatch')
    return {'argmax': max(range(len(a)), key=a.__getitem__), 'max_abs_diff': max(differences)}


@pytest.fixture
def model_gate():
    if os.environ.get('FLAGMEMORY_ONEREC_ENABLE') != 'approved-single-forward':
        pytest.fail('explicit model execution enable required')
    path = os.environ.get('FLAGMEMORY_ONEREC_GATE')
    if not path:
        pytest.fail('model gate evidence required')
    try:
        return validate_model_gate(json.loads(Path(path).read_text(encoding='utf8')))
    except (ValueError, OSError, TypeError) as exc:
        pytest.fail('invalid model execution gate: ' + str(exc))


def test_onerec_forward_consumes_pool_input_ids(model_gate):
    # All expensive imports occur only after the explicit gate succeeds.
    import torch
    import torch_npu  # noqa: F401
    import transformers
    from ascend_test_adapter import AscendTestBackend, AscendTensorAllocator
    from flagmemory.pool import PoolForm, ReusePolicy, TierSpec, create_pool
    from pooled_model_inputs import InputLeaseError, lease_input_ids, assert_consumed_input, release_input_ids, confirm_completion

    device = model_gate['device']
    if torch.__version__.split('+')[0] != model_gate.get('torch_version') or torch_npu.__version__.split('+')[0] != model_gate.get('torch_npu_version') or transformers.__version__ != model_gate.get('transformers_version'):
        raise ValueError('framework versions differ from approved evidence')
    torch.npu.set_device(device)
    backend = AscendTestBackend(torch, device)
    allocator = AscendTensorAllocator(torch, device, 16 << 20)
    pool = create_pool(backend, device_index=device, form=PoolForm.TENSOR,
                       base_allocator=allocator, tier_spec=TierSpec(8 << 20, 8 << 20, 2),
                       reuse=ReusePolicy(opportunistic=False), barrier_timeout_s=10)
    lease = None
    succeeded = False
    hook = None
    output = last = None
    try:
        import resource
        free, total = torch.npu.mem_get_info(device)
        hbm_before = int(total - free)
        print({'hbm_before_bytes': hbm_before, 'hbm_observation_limit_bytes': model_gate['device_observation_limit_bytes']})
        if hbm_before >= model_gate['device_observation_limit_bytes']:
            raise RuntimeError('device observation limit reached')
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_gate['model_path'], local_files_only=True, trust_remote_code=False)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_gate['model_path'], local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation='eager')
        if getattr(model, 'hf_device_map', None):
            raise ValueError('device_map is not permitted')
        model.to(f'npu:{device}').eval()
        if any(p.device.type != 'npu' or p.device.index != device for p in model.parameters()):
            raise ValueError('model parameters are not all on target device')
        messages = [{'role': 'user', 'content': 'Explain this video: <|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|>'}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors='pt', truncation=False)
        if set(inputs) != {'input_ids', 'attention_mask'}:
            raise ValueError('unexpected tokenizer keys')
        ids, mask = inputs['input_ids'], inputs['attention_mask']
        if ids.dtype != torch.int64 or ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= 128 or mask.shape != ids.shape or mask.dtype != torch.int64 or mask.device.type != 'cpu' or not mask.is_contiguous() or not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError('invalid tokenizer inputs')
        stream = torch.npu.Stream(device=device)
        sid = pool.register_stream(stream)
        with torch.inference_mode(), torch.npu.stream(stream):
            ordinary = ids.to(f'npu:{device}')
            mask = mask.to(f'npu:{device}')
            output = model(input_ids=ordinary, attention_mask=mask, use_cache=False, return_dict=True)
            if output.logits.device.type != 'npu' or output.logits.device.index != device:
                raise ValueError('baseline output device mismatch')
            baseline = output.logits[:, -1, :].float().cpu().flatten().tolist()
            del output, ordinary
            output = None
            try:
                lease = lease_input_ids(pool, torch, ids, stream=stream, stream_id=sid, device=device)
            except InputLeaseError as exc:
                lease = exc.lease
                raise
            if not torch.equal(lease.tensor.cpu(), ids):
                raise ValueError('pooled input differs from baseline')
            seen = []
            def before_embedding(module, args):
                if len(args) != 1:
                    raise ValueError('unexpected embedding arguments')
                seen.append(assert_consumed_input(lease, args[0], device=device))
            hook = model.get_input_embeddings().register_forward_pre_hook(before_embedding)
            try:
                output = model(input_ids=lease.tensor, attention_mask=mask, use_cache=False, return_dict=True)
                if output.logits.device.type != 'npu' or output.logits.device.index != device:
                    raise ValueError('pooled output device mismatch')
                last = output.logits[:, -1, :]
                pooled = last.float().cpu().flatten().tolist()
            finally:
                hook.remove()
                hook = None
            if len(seen) != 1:
                raise ValueError('embedding hook must run exactly once')
            result = compare_forward(baseline, pooled, atol=model_gate['atol'], rtol=model_gate['rtol'])
            free, total = torch.npu.mem_get_info(device)
            hbm_after = int(total - free)
            rss_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
            print({'hbm_after_bytes': hbm_after, 'rss_peak_bytes': rss_peak})
            if hbm_after >= model_gate['device_observation_limit_bytes'] or rss_peak >= model_gate['host_memory_bytes']:
                raise RuntimeError('approved resource observation limit reached')
            succeeded = True
            print({'consumer': seen[0], 'comparison': result, 'input_ids_sha256': hashlib.sha256(str(ids.tolist()).encode()).hexdigest(), 'input_shape': list(ids.shape), 'purpose': 'activation', 'allocated_bytes': lease.nbytes})
    finally:
        if hook is not None:
            hook.remove()
        try:
            if lease is not None and not lease.closed and not lease.return_attempted:
                confirm_completion(backend, stream=stream, timeout_s=10)
                output = last = None
                release_input_ids(lease, pool, backend, stream=stream, timeout_s=10)
                if succeeded:
                    previous = lease
                    lease = lease_input_ids(pool, torch, ids, stream=stream, stream_id=sid, device=device)
                    if lease.storage_id != previous.storage_id or lease.block_id == previous.block_id:
                        raise AssertionError('completed pool storage was not reused as a new lease')
                    print({'reused_storage_id': lease.storage_id, 'new_block_id': lease.block_id})
                    release_input_ids(lease, pool, backend, stream=stream, timeout_s=10)
        finally:
            report = pool.destroy_pool(timeout_s=10)
            print({'destroy': str(report), 'live_bytes': allocator.live_bytes, 'allocations': allocator.allocate_calls, 'releases': allocator.release_calls})
            assert report.completed and not report.pending and not report.quarantined and not report.leaked
            assert allocator.live_bytes == 0 and allocator.release_calls == allocator.allocate_calls
