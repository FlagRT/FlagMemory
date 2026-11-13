# SPDX-License-Identifier: Apache-2.0
"""Host-memory pinning helpers for the Cambricon MLU offload connector.

Port of the npu/xpu mem_ops. Replaces
vllm/v1/simple_kv_offload/cuda_mem_ops.py pin_tensor (cudaHostRegister via
torch.cuda.cudart()), which has no MLU equivalent exposed to Python (and
torch.cuda is not a compatibility face on this stack). Tensor.pin_memory()
is the platform-agnostic route and was verified to pin successfully on
MLU590 (is_pinned -> True, 2026-11-13). If it ever fails we fall back to
pageable host memory — correctness is unaffected, copies just stop being
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


def stream_priority_range() -> int:
    """Lowest priority (highest value) for transfer streams; MLU mirrors the
    CUDA convention where lower number = higher priority. Probed range on
    MLU590: (0, -3). Values outside the range are silently accepted by
    torch_mlu (stack.lock.cambricon.v1 known issue C-5), so this helper only
    ever returns a value reported by the platform itself."""
    try:
        low_pri, _ = torch.mlu.Stream.priority_range()
        return low_pri
    except (AttributeError, RuntimeError):
        return 0
