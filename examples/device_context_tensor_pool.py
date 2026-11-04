"""Minimal tensor consumer; supply already configured dependencies.

The consumer must not retain the base tensor or views. Data dependencies across
streams are the caller's responsibility. An incomplete report retains custody.
"""
from typing import Any, Callable
from flagmemory.pool import PoolForm, Purpose, TierSpec, create_pool
from flagmemory.pool.types import DestroyReport
from flagmemory.pool.base import BaseAllocator
from flagmemory.adapters.device_context import DeviceContextBridge


def consume_once(bridge: DeviceContextBridge, allocator: BaseAllocator, *,
                 tier_spec: TierSpec, nbytes: int,
                 consume: Callable[[Any], None], device_index: int = 0) -> DestroyReport:
    pool = create_pool(bridge, device_index=device_index, form=PoolForm.TENSOR,
                       base_allocator=allocator, tier_spec=tier_spec)
    lease = None
    error = None
    try:
        stream = bridge.current_stream(device_index)
        lease = pool.alloc_on_stream(nbytes, purpose=Purpose.SCRATCH)
        ref = pool.access_block(lease.block_id)
        try:
            with stream.context():
                bridge.validate_stream_device(stream, device_index)
                consume(ref)
        finally:
            del ref
    except BaseException as exc:
        error = exc
    cleanup_error = None
    if lease is not None:
        try:
            pool.free_block(lease.block_id)
        except BaseException as exc:
            cleanup_error = exc
    try:
        report = pool.destroy_pool(timeout_s=0)
    except BaseException as exc:
        report = getattr(exc, "report", None)
        if cleanup_error is None:
            cleanup_error = exc
    if error is not None:
        if report is not None:
            try:
                error.destroy_report = report
            except (AttributeError, TypeError):
                wrapper = RuntimeError("consumer failed; cleanup report attached")
                wrapper.destroy_report = report
                raise wrapper from error
        if cleanup_error is not None:
            raise error from cleanup_error
        raise error
    if cleanup_error is not None:
        if report is not None:
            cleanup_error.destroy_report = report
        raise cleanup_error
    return report
