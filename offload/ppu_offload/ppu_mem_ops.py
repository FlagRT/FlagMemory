# SPDX-License-Identifier: Apache-2.0
# Vendored from vLLM 0.24 v1/simple_kv_offload/cuda_mem_ops.py (FlagRT memory-manager).
# PPU port (P-2): PPU HGGC driver does not implement cuMemcpyBatchAsync
# ("hgapiMemcpyBatchAsync: function not implemented", err=998), so copy_blocks
# falls back to per-block tensor.copy_ on the given stream. Pinning via
# cudaHostRegister is kept with a pin_memory() fallback (P-3).
"""Low-level PPU memory helpers: pinning and per-block DMA transfers."""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Kept for signature compatibility with the vendor copy_backend, which passes
# them to build_params; the PPU transfer path does not use access orders.
CU_MEMCPY_SRC_ACCESS_ORDER_STREAM = 1
CU_MEMCPY_SRC_ACCESS_ORDER_ANY = 3

# DMA diagnostic counters (w2 tradition). Printed to stderr because vllm's
# logging config only surfaces vllm.* loggers, not offload.*.
import sys as _sys
import time as _time

_dma_stats = {"store_calls": 0, "store_blocks": 0, "load_calls": 0, "load_blocks": 0}


def _dma_diag(direction: str, n_blocks: int, num_layers: int) -> None:
    st = _dma_stats
    st[f"{direction}_calls"] += 1
    st[f"{direction}_blocks"] += n_blocks
    print(
        f"[PpuOffload-DMA] {direction} blocks={n_blocks} layers={num_layers} "
        f"total_{direction}_calls={st[f'{direction}_calls']} "
        f"total_{direction}_blocks={st[f'{direction}_blocks']} "
        f"t={_time.strftime('%H:%M:%S')}",
        file=_sys.stderr,
        flush=True,
    )


def pin_tensor(tensor: torch.Tensor) -> None:
    """Pin a CPU tensor via cudaHostRegister (bypasses power-of-2 rounding)."""
    err = torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), tensor.nbytes, 0)
    if err.value != 0:
        raise RuntimeError(f"cudaHostRegister failed: {err}")


def pin_tensor_or_fallback(tensor: torch.Tensor) -> torch.Tensor:
    """Pin in place when possible; otherwise return a pin_memory() copy.

    Returns the tensor that should be used (in-place pinning keeps identity).
    """
    try:
        pin_tensor(tensor)
        return tensor
    except Exception as e:
        logger.warning(
            "PpuOffload: cudaHostRegister failed (%s); "
            "falling back to tensor.pin_memory().",
            e,
        )
        try:
            return tensor.pin_memory()
        except Exception as e2:
            logger.warning(
                "PpuOffload: pin_memory() also failed (%s); leaving unpinned.", e2
            )
            return tensor


class BatchMemcpyParams:
    """Per-direction copy plan: block geometry + tensor references + stream.

    Field names src_bases/dst_bases/bpb/num_layers kept for log compatibility
    with the vendor implementation; the actual transfer path uses tensors.
    """

    def __init__(
        self,
        src_caches: dict[str, torch.Tensor],
        dst_caches: dict[str, torch.Tensor],
        stream: torch.cuda.Stream,
    ):
        assert list(src_caches.keys()) == list(dst_caches.keys())
        src_tensors = list(src_caches.values())
        dst_tensors = list(dst_caches.values())
        bpb = []
        for s, d in zip(src_tensors, dst_tensors):
            s_bpb = s.stride(0) * s.element_size()
            assert s_bpb == d.stride(0) * d.element_size()
            bpb.append(s_bpb)
        self.src_bases = [s.data_ptr() for s in src_tensors]
        self.dst_bases = [d.data_ptr() for d in dst_tensors]
        self.bpb = bpb
        self.num_layers = len(src_tensors)
        self.src_tensors = src_tensors
        self.dst_tensors = dst_tensors
        self.stream = stream


def build_params(
    src_caches: dict[str, torch.Tensor],
    dst_caches: dict[str, torch.Tensor],
    stream: torch.cuda.Stream,
    src_access_order: int = 0,
) -> BatchMemcpyParams:
    """Build the copy plan for one direction. src_access_order is accepted
    for signature compatibility with the vendor backend and ignored."""
    return BatchMemcpyParams(src_caches, dst_caches, stream)


def copy_blocks(
    src_block_ids: list[int],
    dst_block_ids: list[int],
    params: BatchMemcpyParams,
    direction: str = "store",
) -> None:
    """Copy blocks one by one with tensor.copy_ on the direction stream.

    PPU fallback for the unavailable batch memcpy API. Each copy_ is a real
    DMA (DtoH on store / HtoD on load); non_blocking is honored for pinned
    CPU-side memory.
    """
    n = len(src_block_ids)
    if n == 0:
        return
    _dma_diag(direction, n, params.num_layers)
    with torch.cuda.stream(params.stream):
        for s, d in zip(params.src_tensors, params.dst_tensors):
            for k in range(n):
                d[dst_block_ids[k]].copy_(s[src_block_ids[k]], non_blocking=True)
