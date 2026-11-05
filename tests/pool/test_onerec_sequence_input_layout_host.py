"""Singleton-stride model cross-checked against fixed real CPU Torch evidence."""
from types import SimpleNamespace
import pytest
from pooled_model_inputs import lease_input_ids

class AllocationIntercept(RuntimeError): pass
class NoAllocation:
    def alloc_on_stream(self,*args,**kwargs): raise AllocationIntercept

class CpuLayout:
    # Semantics independently observed with Torch: narrowing size-one batches
    # preserves leading stride; contiguous() does not normalize it.
    dtype='int64'
    device=SimpleNamespace(type='cpu',index=None)
    def __init__(self,values,shape,stride): self.values=list(values);self.shape=shape;self.layout=stride
    def stride(self):return self.layout
    def is_contiguous(self):return self.layout[-1]==1 and self.shape[0]==1
    def repeat(self,batch,count):
        assert batch==1
        return CpuLayout(self.values*count,(1,len(self.values)*count),(len(self.values)*count,1))
    def narrow(self,dim,start,length):
        assert dim==1 and start==0
        return CpuLayout(self.values[:length],(1,length),self.layout)
    def contiguous(self):return self
    def flatten(self):return CpuLayout(self.values,(len(self.values),),(1,))
    def clone(self):return CpuLayout(self.values,self.shape,self.layout)
    def view(self,batch,length):
        assert batch==1 and len(self.shape)==1 and self.layout==(1,)
        return CpuLayout(self.values,(1,length),(length,1))


def test_original_singleton_strides_are_contiguous_but_guard_rejects():
    seed=CpuLayout(range(19),(1,19),(19,1))
    for length,leading in ((64,76),(128,133)):
        ids=seed.repeat(1,(length+18)//19).narrow(1,0,length).contiguous()
        assert ids.is_contiguous() and ids.stride()==(leading,1)
        with pytest.raises(ValueError):lease_input_ids(NoAllocation(),SimpleNamespace(int64='int64'),ids,stream=object(),stream_id=1,device=0)


def test_input_builder_normalizes_all_ids_and_masks_preserving_bytes():
    from test_onerec_device_context_ascend import build_cpu_inputs
    seed=CpuLayout(range(19),(1,19),(19,1))
    runtime=SimpleNamespace(ones_like=lambda ids:CpuLayout([1]*ids.shape[1],ids.shape,ids.stride()))
    inputs=build_cpu_inputs(runtime,seed)
    assert len(inputs)==16
    for (ids,mask),length in zip(inputs,(19,64,128,19)*4):
        assert ids.shape==mask.shape==(1,length)
        assert ids.stride()==mask.stride()==(length,1)
        assert ids.values==[i%19 for i in range(length)] and mask.values==[1]*length
        with pytest.raises(AllocationIntercept):
            lease_input_ids(NoAllocation(),SimpleNamespace(int64='int64'),ids,stream=object(),stream_id=1,device=0)
