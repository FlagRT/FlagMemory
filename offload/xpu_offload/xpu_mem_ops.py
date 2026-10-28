# SPDX-License-Identifier: Apache-2.0
"""Host-memory pinning helper for the XPU (kunlun) offload connector.

Port of w2_npu_offload.npu_mem_ops minus the stream-priority helper: on this
stack Stream.priority_range must never be called (no priority semantics on
XMLIR — stock death point P-1), so the worker creates plain streams directly.

Replaces vllm/v1/simple_kv_offload/cuda_mem_ops.py pin_tensor
(cudaHostRegister + cuda-python), which is unreachable here (P-2).
Tensor.pin_memory() is platform-agnostic torch API and W1 verified it pins
successfully on this stack (is_pinned -> True). If it ever fails we fall back
to pageable host memory — correctness is unaffected, copies just stop being
truly asynchronous (non_blocking degrades to a staging copy).
"""

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_fallback_logged = False


def pin_or_fallback(tensor: torch.Tensor) -> tuple[torch.Tensor, str]:
    """Pin a CPU tensor if the platform supports it, else return it unchanged.

    Returns (tensor, mode) with mode in {"pinned", "pageable"}. Note
    Tensor.pin_memory() returns a new tensor (and torch's host allocator may
    round sizes up to powers of two — accepted here, same economics upstream
    sought to avoid via cudaHostRegister).
    """
    global _fallback_logged
    try:
        return tensor.pin_memory(), "pinned"
    except Exception as e:
        if not _fallback_logged:
            _fallback_logged = True
            logger.warning(
                "pin_memory() unavailable (%s); falling back to pageable host "
                "memory. Offload works but transfers will be slower.", e
            )
        return tensor, "pageable"
