"""Gated unified-bridge OneRec sequence; collection imports no vendor framework."""
import json
import os
from pathlib import Path
import pytest
import test_onerec_inputs_ascend as legacy
from device_context_ascend_case import (ADAPTER_KIND,RUNTIME_COMMIT,RUNTIME_TREE,
    load_manifest,load_runtime_types,make_case_profile,verify_runtime_origins)
from device_context_onerec_sequence import run_sequence,sequence_lengths

pytestmark=pytest.mark.realcard
LIFECYCLE_COMMIT='a51ec5c00505eb6a816fcdeb57b8ccdb8c3357c5'
LIFECYCLE_SOURCE='0448781653cde6261fbf1ecc7c229f5cb601b004131406e0b3fb77d0057795cc'
LIFECYCLE_NODEIDS=[
    'tests/pool/test_pool_device_context_ascend_realcard.py::test_actual_tensor_and_cross_stream_dependency',
    'tests/pool/test_pool_device_context_ascend_realcard.py::test_completed_reuse_rejects_old_lease',
    'tests/pool/test_pool_device_context_ascend_realcard.py::test_pending_borrower_protection_when_observed']


def validate_sequence_gate(config):
    if config.get('resource_scope')!='onerec-bridge-sequence': raise ValueError('new sequence resource scope required')
    for key,value in {'host_memory_bytes':32<<30,'device_observation_limit_bytes':32<<30,'timeout_s':600,'atol':.001,'rtol':.001,'device':0,'physical_device':15}.items():
        if config.get(key)!=value or isinstance(config.get(key),bool): raise ValueError('fixed sequence resource required: '+key)
    if (config.get('lifecycle_source_commit')!=LIFECYCLE_COMMIT
        or config.get('lifecycle_source_sha256')!=LIFECYCLE_SOURCE
        or config.get('lifecycle_independent_review_accepted') is not True
        or config.get('source_ancestry_verified') is not True
        or LIFECYCLE_COMMIT not in config.get('accepted_ancestors',[])
        or config.get('source_commit')!=config.get('model_source_commit')
        or not config.get('model_source_commit')):
        raise ValueError('accepted lifecycle source ancestry required')
    item=config.get('lifecycle',{}); path=Path(item.get('path',''))
    if not path.is_file() or path.stat().st_size>1<<20 or legacy._hash(path)!=item.get('sha256'): raise ValueError('lifecycle evidence identity')
    evidence=json.loads(path.read_text(encoding='utf8'))
    if (evidence.get('stage')!='bridge-lifecycle' or evidence.get('status')!='passed'
        or evidence.get('source_sha256')!=LIFECYCLE_SOURCE or evidence.get('image_id')!=config.get('image_id')
        or evidence.get('adapter_kind')!=ADAPTER_KIND or evidence.get('nodeids')!=LIFECYCLE_NODEIDS
        or evidence.get('executed')!=3 or evidence.get('skipped')!=0 or evidence.get('errors')!=0
        or evidence.get('failures')!=0 or evidence.get('exit_code')!=0 or not evidence.get('run_id')):
        raise ValueError('accepted three-pass lifecycle required')
    expected={'source_repository':'https://github.com/FlagRT/runtime-team','source_commit':RUNTIME_COMMIT,
              'runtime_tree':RUNTIME_TREE,'manifest_sha256':config.get('device_manifest_sha256')}
    if evidence.get('device_runtime')!=expected: raise ValueError('public runtime identity differs')
    manifest=load_manifest(Path(config.get('device_manifest_path','')),config.get('device_manifest_sha256'))
    if manifest.get('source_commit')!=RUNTIME_COMMIT: raise ValueError('device source identity')
    old=dict(config,resource_scope='onerec-single-forward')
    legacy.validate_model_gate(old)
    return config


def load_model_pair(transformers,model_path,dtype):
    tokenizer=transformers.AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    model=transformers.AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,
        trust_remote_code=False,torch_dtype=dtype,attn_implementation='eager')
    return tokenizer,model


@pytest.fixture
def sequence_gate():
    if os.environ.get('FLAGMEMORY_ONEREC_ENABLE')!='approved-bridge-sequence': pytest.fail('new model sequence enable required')
    path=os.environ.get('FLAGMEMORY_ONEREC_GATE')
    if not path: pytest.fail('sequence gate required')
    return validate_sequence_gate(json.loads(Path(path).read_text(encoding='utf8')))


def test_onerec_continuous_inputs_reuse_and_reclaim(sequence_gate):
    gate=sequence_gate
    # Every source/model/execution gate runs before these expensive imports.
    manifest_path=Path(gate['device_manifest_path'])
    manifest=load_manifest(manifest_path,gate['device_manifest_sha256'])
    backend_type,stream_type,event_type=load_runtime_types(manifest_path.parent,manifest)
    import torch
    import torch_npu
    import transformers
    from flagmemory.pool import PoolForm,ReusePolicy,TierSpec,create_pool
    if (torch.__version__.split('+')[0]!=gate['torch_version']
        or torch_npu.__version__.split('+')[0]!=gate['torch_npu_version']
        or transformers.__version__!=gate['transformers_version']): raise ValueError('framework identity differs')
    device=gate['device']; torch.npu.set_device(device); torch.npu.current_stream(device)
    if not torch.npu.is_initialized(): raise ValueError('caller framework context uninitialized')
    profile=make_case_profile(torch,backend=backend_type(),stream_type=stream_type,event_type=event_type,
                              device_index=device,byte_limit=16<<20)
    def observe():
        value=legacy._sample_resources(torch,device)
        if ('hbm_used_bytes' not in value or 'rss_peak_bytes' not in value
            or value['hbm_used_bytes']>=32<<30 or value['rss_peak_bytes']>=32<<30):
            raise RuntimeError('resource observation unavailable or limit reached')
        return value
    record={'model_loads':0,'input_rule':'19-token seed repeated then narrowed; mask all ones',
            'adapter_kind':ADAPTER_KIND,'device_runtime':{'source_repository':'https://github.com/FlagRT/runtime-team','source_commit':RUNTIME_COMMIT,'runtime_tree':RUNTIME_TREE,'manifest_sha256':gate['device_manifest_sha256']},
            'accepted_lifecycle_commit':gate['lifecycle_source_commit'],
            'model_source_commit':gate['model_source_commit'],'observations':{}}
    pool=None; primary=None; sequence_owns_pool=False
    try:
        record['observations']['before_model']=observe()
        tokenizer,model=load_model_pair(transformers,gate['model_path'],torch.bfloat16); record['model_loads']=1
        if getattr(model,'hf_device_map',None): raise ValueError('device_map not permitted')
        model.to(f'npu:{device}').eval()
        if any(p.device.type!='npu' or p.device.index!=device for p in model.parameters()): raise ValueError('model target mismatch')
        record['observations']['after_model']=observe()
        messages=[{'role':'user','content':'Explain this video: <|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|>'}]
        prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        tokens=tokenizer(prompt,return_tensors='pt',truncation=False)
        if set(tokens)!={'input_ids','attention_mask'}: raise ValueError('unexpected tokenizer keys')
        seed=tokens['input_ids']
        seed_mask=tokens['attention_mask']
        if (tuple(seed_mask.shape)!=(1,19) or seed_mask.dtype!=torch.int64
            or seed_mask.device.type!='cpu' or not seed_mask.is_contiguous()
            or not bool((seed_mask==1).all())): raise ValueError('invalid seed mask')
        if (seed.dtype!=torch.int64 or tuple(seed.shape)!=(1,19) or seed.device.type!='cpu'
            or not seed.is_contiguous() or int(seed.min())<0
            or int(seed.max())>=model.get_input_embeddings().num_embeddings): raise ValueError('invalid fixed seed token range')
        cpu_inputs=[]
        for length in sequence_lengths():
            ids=seed.repeat(1,(length+18)//19).narrow(1,0,length).contiguous()
            cpu_inputs.append((ids,torch.ones_like(ids)))
        stream=profile.new_stream()
        pool=create_pool(profile.bridge,device_index=device,form=PoolForm.TENSOR,base_allocator=profile.allocator,
            tier_spec=TierSpec(8<<20,8<<20,2),reuse=ReusePolicy(opportunistic=False),barrier_timeout_s=10)
        sid=pool.register_stream(stream)
        # run_sequence owns the sole destroy call, including its failure path.
        sequence_owns_pool=True
        record['sequence']=run_sequence(model=model,runtime=torch,pool=pool,backend=profile.bridge,
            stream=stream,stream_id=sid,device=device,cpu_inputs=cpu_inputs,atol=gate['atol'],rtol=gate['rtol'],observe=observe)
        if (profile.allocator.allocate_calls!=1 or profile.allocator.allocated_bytes!=8<<20
            or profile.allocator.release_calls!=1 or profile.allocator.live_bytes): raise ValueError('allocator lifecycle differs')
        record['origins']=verify_runtime_origins(manifest_path.parent)
    except BaseException as exc:
        primary=exc
        if hasattr(exc,'sequence_record'): record['sequence']=exc.sequence_record
        raise
    finally:
        if pool is not None and not sequence_owns_pool:
            try:
                report=pool.destroy_pool(timeout_s=10)
                record['pre_sequence_destroy']=legacy._plain(report)
                if not report.completed or report.pending or report.quarantined or report.leaked: raise RuntimeError('pre-sequence destroy incomplete')
            except BaseException as cleanup:
                record['pre_sequence_cleanup_error']=repr(cleanup)
                if primary is not None:
                    primary.pre_sequence_cleanup_error=cleanup
                    primary.pre_sequence_destroy_report=getattr(cleanup,'report',None)
                else: raise
        record['observations']['after_destroy']=legacy._sample_resources(torch,device)
        record['primary_error']=repr(primary) if primary is not None else None
        record['allocator']={'allocate_calls':profile.allocator.allocate_calls,'release_calls':profile.allocator.release_calls,
            'allocated_bytes':profile.allocator.allocated_bytes,'live_bytes':profile.allocator.live_bytes}
        print('ONEREC_SEQUENCE_EVIDENCE '+json.dumps(legacy._plain(record),sort_keys=True))
