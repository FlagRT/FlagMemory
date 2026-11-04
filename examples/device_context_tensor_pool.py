"""Minimal tensor consumer; supply already configured dependencies.

The consumer must not retain the base tensor or views. Data dependencies across
streams are the caller's responsibility. An incomplete report retains custody.
"""
from typing import Any, Callable
from flagmemory.pool import PoolForm, Purpose, TierSpec, create_pool
from flagmemory.pool.types import DestroyReport
from flagmemory.pool.base import BaseAllocator
from flagmemory.adapters.device_context import DeviceContextBridge


class CleanupFailures(Exception):
    """Structured cleanup outcomes, attached to the primary exception."""

    def __init__(self, free_error: BaseException | None,
                 destroy_error: BaseException | None,
                 destroy_report: DestroyReport | None) -> None:
        super().__init__('cleanup failed or report could not be attached')
        self.free_error = free_error
        self.destroy_error = destroy_error
        self.destroy_report = destroy_report


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
    free_error = None
    destroy_error = None
    if lease is not None:
        try:
            pool.free_block(lease.block_id)
        except BaseException as exc:
            free_error = exc
    try:
        report = pool.destroy_pool(timeout_s=0)
    except BaseException as exc:
        destroy_error = exc
        report = getattr(exc, "report", None)
    primary = error if error is not None else (free_error if free_error is not None else destroy_error)
    if primary is not None:
        report_attached = False
        if report is not None:
            try:
                primary.destroy_report = report
                report_attached = True
            except (AttributeError, TypeError):
                pass
        if free_error is not None or destroy_error is not None or (report is not None and not report_attached):
            cleanup = CleanupFailures(free_error, destroy_error, report)
            if primary.__cause__ is not None:
                BaseException.__setattr__(primary, "cleanup_failures", cleanup)
                raise primary
            raise primary from cleanup
        raise primary
    return report
