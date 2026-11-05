"""Test-only finite paired forwards using one explicit unified stream."""
import hashlib
import sys
from pooled_model_inputs import InputLeaseError, lease_input_ids, assert_consumed_input, release_input_ids, confirm_completion
from test_onerec_inputs_ascend import compare_forward, _plain
_CUSTODY = []

def sequence_lengths(): return (19,64,128,19)*4

def _digest(ids): return hashlib.sha256(str(ids.tolist()).encode()).hexdigest()

def run_sequence(*,model,runtime,pool,backend,stream,stream_id,device,cpu_inputs,atol,rtol,observe):
    record={'groups':[],'completed':0,'cleanup_errors':[],'destroy':None}
    lease=hook=output=last=ordinary=mask=None
    errors=[]; primary=None; previous=None; stage='validate_inputs'
    try:
        if len(cpu_inputs)!=16: raise ValueError('exactly sixteen inputs required')
        for (ids,cpu_mask),length in zip(cpu_inputs,sequence_lengths()):
            if (tuple(ids.shape)!=(1,length) or tuple(cpu_mask.shape)!=(1,length)
                or ids.dtype!=runtime.int64 or cpu_mask.dtype!=runtime.int64
                or ids.device.type!='cpu' or cpu_mask.device.type!='cpu'
                or not ids.is_contiguous() or not cpu_mask.is_contiguous()
                or any(v!=1 for row in cpu_mask.tolist() for v in row)):
                raise ValueError('invalid sequence input or mask')
        for index,(ids,cpu_mask) in enumerate(cpu_inputs):
            group={'index':index,'length':sequence_lengths()[index],'input_sha256':_digest(ids),'shape':list(ids.shape)}
            record['groups'].append(group); stage='baseline'
            with runtime.inference_mode(),stream.context():
                ordinary=ids.to(f'npu:{device}'); mask=cpu_mask.to(f'npu:{device}')
                output=model(input_ids=ordinary,attention_mask=mask,use_cache=False,return_dict=True)
                if output.logits.device.type!='npu' or output.logits.device.index!=device: raise ValueError('baseline output device mismatch')
                baseline=output.logits[:,-1,:].float().cpu().flatten().tolist()
                output=ordinary=None; stage='lease'
                try: lease=lease_input_ids(pool,runtime,ids,stream=stream,stream_id=stream_id,device=device,stream_context=stream.context)
                except InputLeaseError as exc: lease=exc.lease; raise
                copied=lease.tensor.cpu(); group['input_equal']=bool(runtime.equal(copied,ids)); group['pooled_sha256']=_digest(copied); copied=None
                if not group['input_equal'] or group['input_sha256']!=group['pooled_sha256']: raise ValueError('pooled input differs')
                seen=[]
                def before_embedding(module,args):
                    if len(args)!=1: raise ValueError('unexpected embedding arguments')
                    seen.append(assert_consumed_input(lease,args[0],device=device))
                hook=model.get_input_embeddings().register_forward_pre_hook(before_embedding); stage='pooled_forward'
                try:
                    output=model(input_ids=lease.tensor,attention_mask=mask,use_cache=False,return_dict=True)
                    if output.logits.device.type!='npu' or output.logits.device.index!=device: raise ValueError('pooled output device mismatch')
                    last=output.logits[:,-1,:]; pooled=last.float().cpu().flatten().tolist()
                finally:
                    forward_error=sys.exc_info()[1]
                    try: hook.remove()
                    except BaseException as exc:
                        errors.append(exc)
                        if forward_error is None: raise
                    finally: hook=None
                if len(seen)!=1: raise ValueError('embedding must observe exactly one pooled input')
                group['consumer']=seen[0]; group['comparison']=compare_forward(baseline,pooled,atol=atol,rtol=rtol)
                identity=(lease.storage_id,seen[0]['storage_ptr'])
                if previous is not None and (identity!=previous[:2] or lease.block_id==previous[2]): raise ValueError('completed storage not reused')
                previous=(*identity,lease.block_id); stage='completion'
                confirm_completion(backend,stream=stream,timeout_s=10,device_index=device)
                output=last=mask=None; stage='return'
                group['returned']=release_input_ids(lease,pool,backend,stream=stream,timeout_s=10,device_index=device,evidence=group.setdefault('return_evidence',{})); lease=None
            stage='observe'; group['observation']=observe(); record['completed']+=1
    except BaseException as exc:
        primary=exc; record['failed_stage']=stage; record['primary_error']=repr(exc)
    finally:
        if hook is not None:
            try: hook.remove()
            except BaseException as exc: errors.append(exc)
        if primary is not None and lease is None:
            try:
                confirm_completion(backend,stream=stream,timeout_s=10,device_index=device)
                output=last=mask=ordinary=None
            except BaseException as exc: errors.append(exc)
        if lease is not None and not lease.closed and not lease.return_attempted:
            try:
                confirm_completion(backend,stream=stream,timeout_s=10,device_index=device)
                output=last=mask=ordinary=None
                release_input_ids(lease,pool,backend,stream=stream,timeout_s=10,device_index=device,evidence=record.setdefault('failure_return',{}))
            except BaseException as exc: errors.append(exc)
        try:
            report=pool.destroy_pool(timeout_s=10); record['destroy']=_plain(report)
            if not report.completed or report.pending or report.quarantined or report.leaked: raise RuntimeError('sequence pool destroy incomplete')
        except BaseException as exc:
            errors.append(exc)
            if hasattr(exc,'report'): record['destroy']=_plain(exc.report)
        record['cleanup_errors']=[repr(exc) for exc in errors]
        if primary is not None or errors or (lease is not None and not lease.closed): _CUSTODY.append((primary,model,runtime,pool,lease,output,last,mask,ordinary))
    if primary is not None:
        primary.sequence_record=record; primary.sequence_cleanup_errors=tuple(errors); raise primary
    if errors:
        errors[0].sequence_record=record; errors[0].sequence_cleanup_errors=tuple(errors); raise errors[0]
    return record
