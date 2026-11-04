"""Test-only pooled integer inputs; no framework imports."""
from dataclasses import dataclass
import time
import math
from flagmemory.pool import Purpose


@dataclass
class InputLease:
    block_id: int
    storage_id: int
    tensor: object
    base: object
    nbytes: int
    closed: bool = False
    return_attempted: bool = False


class InputLeaseError(RuntimeError):
    def __init__(self, message, lease):
        super().__init__(message)
        self.lease = lease


def _device(tensor, kind, index=None):
    d = tensor.device
    return d.type == kind and (index is None or d.index == index)


def _identity(tensor):
    storage = tensor.untyped_storage()
    offset = tensor.storage_offset() * tensor.element_size()
    size = tensor.numel() * tensor.element_size()
    return storage.data_ptr(), offset, size, tensor.data_ptr()


def lease_input_ids(pool, runtime, cpu_ids, *, stream, stream_id, device):
    shape = tuple(cpu_ids.shape)
    if (len(shape) != 2 or shape[0] != 1 or not 1 <= shape[1] <= 128
            or cpu_ids.dtype != runtime.int64 or not _device(cpu_ids, 'cpu')
            or not cpu_ids.is_contiguous() or tuple(cpu_ids.stride()) != (shape[1], 1)):
        raise ValueError('expected contiguous CPU int64 [1,L], 1 <= L <= 128')
    nbytes = shape[1] * 8
    block = pool.alloc_on_stream(nbytes, stream_id=stream_id, purpose=Purpose.ACTIVATION)
    lease = InputLease(block.block_id, block.storage_id, None, None, nbytes)
    try:
        base = pool.access_block(block.block_id)
        lease.base = base
        if (not _device(base, 'npu', device) or base.dtype != runtime.uint8
                or not base.is_contiguous() or base.numel() < nbytes
                or base.storage_offset() % 8):
            raise ValueError('invalid pool backing')
        lease.tensor = base.narrow(0, 0, nbytes).view(runtime.int64).view(1, shape[1])
        assert_consumed_input(lease, lease.tensor, device=device)
        with runtime.npu.device(device), runtime.npu.stream(stream):
            lease.tensor.copy_(cpu_ids, non_blocking=False)
        return lease
    except Exception as exc:
        raise InputLeaseError('input construction failed; lease requires safe cleanup', lease) from exc


def assert_consumed_input(lease, actual, *, device):
    if lease.closed or lease.return_attempted or lease.tensor is None or lease.base is None:
        raise ValueError('lease unavailable')
    expected = lease.tensor
    if (not _device(actual, 'npu', device) or actual.dtype != expected.dtype
            or tuple(actual.shape) != tuple(expected.shape) or not actual.is_contiguous()
            or tuple(actual.stride()) != tuple(expected.stride())):
        raise ValueError('consumer input metadata mismatch')
    base_id, base_offset, base_size, base_ptr = _identity(lease.base)
    sid, offset, size, ptr = _identity(actual)
    expected_id = _identity(expected)
    if (sid != base_id or (sid, offset, size, ptr) != expected_id
            or size != lease.nbytes or offset != base_offset
            or ptr != base_ptr or offset + size > base_offset + base_size):
        raise ValueError('consumer input does not share expected pool range')
    return {'block_id': lease.block_id, 'storage_id': lease.storage_id,
            'storage_ptr': sid, 'offset_bytes': offset, 'data_ptr': ptr, 'nbytes': size}


def confirm_completion(backend, *, stream, timeout_s):
    if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s < 0:
        raise ValueError('invalid timeout')
    event = backend.create_event(device_index=stream.device.index)
    backend.record_event(event, stream)
    deadline = time.monotonic() + timeout_s
    while not backend.query_event(event):
        if time.monotonic() >= deadline:
            raise TimeoutError('input completion not confirmed')
        time.sleep(min(.001, max(0, deadline-time.monotonic())))
    return deadline


def release_input_ids(lease, pool, backend, *, stream, timeout_s, evidence=None):
    if lease.closed or lease.return_attempted:
        raise ValueError('lease already returned or return outcome uncertain')
    evidence = evidence if evidence is not None else {}
    evidence.update(completion={'status': 'not_attempted'}, free={'status': 'not_attempted'}, drain={'status': 'not_attempted'})
    try:
        deadline = confirm_completion(backend, stream=stream, timeout_s=timeout_s)
    except Exception as exc:
        evidence['completion'] = {'status': 'failed', 'error': repr(exc)}
        raise
    evidence['completion'] = {'status': 'confirmed'}
    lease.tensor = None
    lease.base = None
    lease.return_attempted = True
    evidence['free'] = {'status': 'attempted'}
    try:
        returned = pool.free_block(lease.block_id)
    except Exception as exc:
        evidence['free'] = {'status': 'failed_uncertain', 'error': repr(exc)}
        raise
    evidence['free'] = {'status': 'completed', 'block_id': lease.block_id, 'state': str(getattr(returned, 'state', 'unknown'))}
    remaining = max(0, deadline-time.monotonic())
    try:
        result = pool.drain_stream(timeout_s=remaining)
    except Exception as exc:
        evidence['drain'] = {'status': 'failed', 'error': repr(exc)}
        raise
    evidence['drain'] = {'status': 'completed' if not result.recoverable_timeout and not result.still_pending and not result.quarantined_blocks else 'incomplete', 'still_pending': result.still_pending, 'recoverable_timeout': result.recoverable_timeout, 'quarantined_blocks': list(result.quarantined_blocks), 'promoted_blocks': getattr(result, 'promoted_blocks', None), 'promoted_bytes': getattr(result, 'promoted_bytes', None)}
    if result.recoverable_timeout or result.still_pending or result.quarantined_blocks:
        raise RuntimeError('pool drain did not complete')
    lease.closed = True
    return {'completed': True, 'block_id': lease.block_id}
