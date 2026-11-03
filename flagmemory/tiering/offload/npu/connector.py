# SPDX-License-Identifier: Apache-2.0
"""NPU adaptation of SimpleCPUOffloadConnector (vLLM 0.20.2).

Scheduler side is inherited unchanged (pure scheduling logic, no device calls).
Worker side is swapped for NpuSimpleCPUOffloadWorker: vllm-ascend hands
register_kv_caches per-layer (k, v) tuples and the upstream worker assumes
CUDA streams + cuMemcpyBatchAsync.
"""

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
)

from .npu_copy_backend import NpuDmaCopyBackend
from .npu_worker import NpuSimpleCPUOffloadWorker


class NpuSimpleCPUOffloadConnector(SimpleCPUOffloadConnector):
    """SimpleCPUOffloadConnector with the worker handler replaced by the NPU one."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config=None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        # The parent already created a CUDA-bound SimpleCPUOffloadWorker (its
        # __init__ touches no device APIs, so creating it is harmless on NPU);
        # we reuse its resolved cpu_capacity_bytes and swap in the NPU worker.
        # If prefix caching was disabled the parent leaves worker_handler None
        # and offload is off — keep that behavior.
        if role == KVConnectorRole.WORKER and self.worker_handler is not None:
            capacity = self.worker_handler.cpu_capacity_bytes
            self.worker_handler = NpuSimpleCPUOffloadWorker(
                vllm_config, kv_cache_config, capacity
            )
            with open("/tmp/w2_marker_swap.txt", "w") as f:
                import os
                f.write(f"swap done pid={os.getpid()} handler={type(self.worker_handler).__name__}\n")

    def _diag(self, msg: str) -> None:
        try:
            with open("/tmp/w2_marker_diag.txt", "a") as f:
                f.write(msg + "\n")
        except OSError:
            pass

    def get_num_new_matched_tokens(self, request, num_computed_tokens: int):
        n_hashes = len(getattr(request, "block_hashes", None) or [])
        out = super().get_num_new_matched_tokens(request, num_computed_tokens)
        try:
            self._diag(
                f"cpu_lookup req={request.request_id} gpu_hit={num_computed_tokens}"
                f" req_hashes={n_hashes} -> ext={out[0]} async={out[1]}"
            )
        except Exception:
            pass
        return out

    def update_connector_output(self, connector_output) -> None:
        meta = getattr(connector_output, "kv_connector_worker_meta", None)
        if meta is not None:
            self._diag(f"worker_meta received completed_store={dict(meta.completed_store_events)}")
        super().update_connector_output(connector_output)

    def update_state_after_alloc(self, request, blocks, num_external_tokens: int) -> None:
        super().update_state_after_alloc(request, blocks, num_external_tokens)
        if num_external_tokens:
            mgr = self.scheduler_manager
            st = mgr._reqs_to_load.get(request.request_id) if mgr else None
            self._diag(
                f"usa req={request.request_id} ext={num_external_tokens}"
                f" registered_load={st is not None}"
                f" pend_loads={len(mgr._reqs_to_load) if mgr else -1}"
            )

    _meta_log_i = 0

    def build_connector_meta(self, scheduler_output):
        meta = super().build_connector_meta(scheduler_output)
        try:
            if meta.load_event >= 0:
                self._diag(
                    f"meta load_ev={meta.load_event}"
                    f" n_load={len(meta.load_gpu_blocks)} load_cpu={meta.load_cpu_blocks[:4]}"
                )
            elif meta.store_event >= 0:
                self._meta_log_i += 1
                if self._meta_log_i % 200 == 1:
                    self._diag(
                        f"meta store_ev={meta.store_event}"
                        f" n_store={len(meta.store_gpu_blocks)} (sampled)"
                    )
        except Exception:
            pass
        return meta
