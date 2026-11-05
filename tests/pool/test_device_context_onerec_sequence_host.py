import pytest

def test_sequence_is_exactly_sixteen_pairs():
    from device_context_onerec_sequence import sequence_lengths
    assert sequence_lengths() == (19,64,128,19)*4


from contextlib import nullcontext
from types import SimpleNamespace
from test_pooled_model_inputs_host import FakeTensor, FakePool, FakeBackend


class Tensor(FakeTensor):
    def to(self, device):
        return Tensor(buf=bytearray(self.buf), shape=self.shape, dtype=self.dtype, device='npu', index=2)
    def cpu(self): return self
    def tolist(self): return [self.values()]


class Logits:
    device = SimpleNamespace(type='npu', index=2)
    def __getitem__(self, key): return self
    def float(self): return self
    def cpu(self): return self
    def flatten(self): return self
    def tolist(self): return [1., 2., 3.]


class Model:
    def __init__(self, fail_at=None): self.calls=0; self.hook=None; self.removed=0; self.fail_at=fail_at; self.error=RuntimeError('forward'); self.error.__cause__=ValueError('cause')
    def get_input_embeddings(self): return self
    def register_forward_pre_hook(self, callback):
        self.hook=callback
        def remove(): self.hook=None; self.removed+=1
        return SimpleNamespace(remove=remove)
    def __call__(self, *, input_ids, attention_mask, use_cache, return_dict):
        assert use_cache is False and return_dict
        self.calls+=1
        if self.hook: self.hook(self, (input_ids,))
        if self.calls == self.fail_at: raise self.error
        return SimpleNamespace(logits=Logits())


def sequence_case():
    from device_context_onerec_sequence import sequence_lengths
    pool=FakePool(); pool.destroy_calls=0
    def destroy(**kw):
        pool.destroy_calls+=1
        return SimpleNamespace(completed=True,pending=0,quarantined=0,leaked=0)
    pool.destroy_pool=destroy
    runtime=SimpleNamespace(int64='int64',uint8='uint8',inference_mode=nullcontext,equal=lambda a,b:a.values()==b.values())
    stream=SimpleNamespace(context=nullcontext)
    inputs=[(Tensor(list(range(n)),shape=(1,n)),Tensor([1]*n,shape=(1,n))) for n in sequence_lengths()]
    return dict(model=Model(),runtime=runtime,pool=pool,backend=FakeBackend(),stream=stream,stream_id=3,device=2,cpu_inputs=inputs,atol=.001,rtol=.001,observe=lambda:{'hbm_used_bytes':1,'rss_peak_bytes':1})


def test_sequence_all_pairs_return_before_next_and_destroy_once():
    from device_context_onerec_sequence import run_sequence
    kw=sequence_case(); result=run_sequence(**kw)
    assert kw['model'].calls==32 and kw['model'].removed==16
    assert kw['pool'].free_calls==16 and kw['pool'].destroy_calls==1
    assert len(result['groups'])==16 and result['completed']==16
    assert len({g['consumer']['block_id'] for g in result['groups']})==16
    assert {g['consumer']['storage_id'] for g in result['groups']}=={1}
    assert all(g['returned']['completed'] for g in result['groups'])


@pytest.mark.parametrize('fail_at',[1,2,16,31,32])
def test_sequence_stops_on_original_error_and_keeps_cause(fail_at):
    from device_context_onerec_sequence import run_sequence
    kw=sequence_case(); kw['model']=Model(fail_at)
    with pytest.raises(RuntimeError) as caught: run_sequence(**kw)
    assert caught.value is kw['model'].error and isinstance(caught.value.__cause__,ValueError)
    assert kw['model'].calls==fail_at and kw['model'].hook is None
    assert kw['pool'].destroy_calls==1
    assert caught.value.sequence_record['completed']==(fail_at-1)//2


def test_sequence_retains_forward_free_and_destroy_errors():
    from device_context_onerec_sequence import run_sequence
    kw=sequence_case(); kw['model']=Model(2); kw['pool'].fail_free=True
    destroy_error=RuntimeError('destroy')
    def destroy(**kw): raise destroy_error
    kw['pool'].destroy_pool=destroy
    with pytest.raises(RuntimeError) as caught: run_sequence(**kw)
    assert caught.value is kw['model'].error
    assert len(caught.value.sequence_cleanup_errors)==2
    assert caught.value.sequence_cleanup_errors[-1] is destroy_error
    assert kw['pool'].free_calls==1


def test_sequence_rejects_wrong_lengths_before_forward():
    from device_context_onerec_sequence import run_sequence
    kw=sequence_case(); kw['cpu_inputs']=kw['cpu_inputs'][:-1]
    with pytest.raises(ValueError): run_sequence(**kw)
    assert kw['model'].calls==0


@pytest.fixture(autouse=True)
def tensor_readback(monkeypatch):
    monkeypatch.setattr(FakeTensor, 'cpu', lambda self:self, raising=False)
    monkeypatch.setattr(FakeTensor, 'tolist', lambda self:[self.values()], raising=False)


@pytest.mark.parametrize('config',[{}, {'resource_scope':'onerec-single-forward'}, {'resource_scope':'onerec-bridge-sequence'}])
def test_sequence_gate_rejects_old_or_missing_lifecycle_before_framework(config):
    from test_onerec_device_context_ascend import validate_sequence_gate
    with pytest.raises(ValueError): validate_sequence_gate(config)


def test_sequence_entry_collection_has_no_vendor_imports():
    import ast
    from pathlib import Path
    tree=ast.parse(Path(__file__).with_name('test_onerec_device_context_ascend.py').read_text())
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names=[a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else [node.module.split('.')[0]]
            assert not set(names)&{'torch','torch_npu','transformers','runtime'}



def test_load_pair_loads_tokenizer_and_model_once():
    from test_onerec_device_context_ascend import load_model_pair
    calls=[]
    class Loader:
        def __init__(self,name): self.name=name
        def from_pretrained(self,*a,**kw): calls.append((self.name,kw)); return self
    transformers=SimpleNamespace(AutoTokenizer=Loader('tokenizer'),AutoModelForCausalLM=Loader('model'))
    load_model_pair(transformers,'model','bf16')
    assert [name for name,_ in calls]==['tokenizer','model']
    assert calls[-1][1]['attn_implementation']=='eager'


def test_forward_error_survives_hook_remove_failure():
    from device_context_onerec_sequence import run_sequence
    kw=sequence_case(); kw['model']=Model(2); error=RuntimeError('remove')
    def register(callback):
        kw['model'].hook=callback
        def remove(): raise error
        return SimpleNamespace(remove=remove)
    kw['model'].register_forward_pre_hook=register
    with pytest.raises(RuntimeError) as caught: run_sequence(**kw)
    assert caught.value is kw['model'].error
    assert error in caught.value.sequence_cleanup_errors


def good_gate(tmp_path,monkeypatch):
    import json,hashlib
    import test_onerec_device_context_ascend as entry
    lifecycle={'stage':'bridge-lifecycle','status':'passed','source_sha256':entry.LIFECYCLE_SOURCE,
      'image_id':'image','adapter_kind':entry.ADAPTER_KIND,'nodeids':entry.LIFECYCLE_NODEIDS,
      'executed':3,'skipped':0,'errors':0,'failures':0,'exit_code':0,'run_id':'run',
      'device_runtime':{'source_repository':'https://github.com/FlagRT/runtime-team','source_commit':entry.RUNTIME_COMMIT,'runtime_tree':entry.RUNTIME_TREE,'manifest_sha256':'manifest'}}
    path=tmp_path/'lifecycle.json';path.write_text(json.dumps(lifecycle))
    gate=dict(resource_scope='onerec-bridge-sequence',host_memory_bytes=32<<30,device_observation_limit_bytes=32<<30,
      timeout_s=600,atol=.001,rtol=.001,device=0,physical_device=15,lifecycle_source_commit=entry.LIFECYCLE_COMMIT,
      lifecycle_source_sha256=entry.LIFECYCLE_SOURCE,lifecycle_independent_review_accepted=True,source_ancestry_verified=True,
      accepted_ancestors=[entry.LIFECYCLE_COMMIT],source_commit='new-source',model_source_commit='new-source',image_id='image',
      lifecycle={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},device_manifest_sha256='manifest',device_manifest_path=str(tmp_path/'manifest.json'))
    monkeypatch.setattr(entry,'load_manifest',lambda *a:{'source_commit':entry.RUNTIME_COMMIT})
    monkeypatch.setattr(entry.legacy,'validate_model_gate',lambda config:config)
    return entry,gate,lifecycle,path


def test_new_gate_accepts_separate_model_source_and_accepted_lifecycle(tmp_path,monkeypatch):
    entry,gate,_,_=good_gate(tmp_path,monkeypatch)
    assert entry.validate_sequence_gate(gate) is gate


@pytest.mark.parametrize('fault',['source','adapter','skipped','nodeids','runtime','sha','review','ancestry'])
def test_new_gate_rejects_lifecycle_or_source_mismatch(tmp_path,monkeypatch,fault):
    import hashlib,json
    entry,gate,data,path=good_gate(tmp_path,monkeypatch)
    if fault=='source': data['source_sha256']='foreign'
    if fault=='adapter':data['adapter_kind']='direct'
    if fault=='skipped':data['skipped']=1
    if fault=='nodeids':data['nodeids']=[]
    if fault=='runtime':data['device_runtime']['runtime_tree']='foreign'
    if fault=='review':gate['lifecycle_independent_review_accepted']=False
    if fault=='ancestry':gate['accepted_ancestors']=[]
    path.write_text(json.dumps(data));gate['lifecycle']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    if fault=='sha':gate['lifecycle']['sha256']='foreign'
    with pytest.raises(ValueError):entry.validate_sequence_gate(gate)
