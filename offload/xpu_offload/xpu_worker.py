# SPDX-License-Identifier: Apache-2.0
"""Worker-side handler for the XPU (kunlun) offload connector.

Port of w2_npu_offload.npu_worker. register_kv_caches follows
vllm/v1/simple_kv_offload/worker.py (Apache-2.0, Copyright vLLM contributors)
with the platform-specific pieces replaced:

1. Input normalization (kept from the NPU port as a defensive superset):
   per-layer (k, v) tuples are split into {name}.0/{name}.1 entries. W1 on
   this stack showed the stock registration path passes its checks, but the
   normalizer is harmless for plain tensors and keeps one code path across
   both platforms.
2. torch.cuda streams + XpuCopyBackend instead of DmaCopyBackend
   (cuMemcpyBatchAsync via cuda-python). Streams are created WITHOUT
   priority: XMLIR implements Stream but has no priority semantics, and
   querying priority_range is exactly stock death point P-1.
"""

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

from .xpu_copy_backend import XpuCopyBackend
from .xpu_mem_ops import pin_or_fallback

logger = init_logger(__name__)


def normalize_kv_caches(kv_caches) -> dict[str, tuple[torch.Tensor, int]]:
    """dict[str, Tensor | tuple | list] -> dict[str, (Tensor, sibling_count)].

    sibling_count = number of tensors sharing the layer's spec page_size_bytes
    (2 for a (k, v) tuple, 1 for a plain tensor) — needed because the spec
    page covers the layer's whole allocation.
    """
    out: dict[str, tuple[torch.Tensor, int]] = {}
    for name, v in kv_caches.items():
        if isinstance(v, torch.Tensor):
            out[name] = (v, 1)
        elif isinstance(v, (tuple, list)):
            if not v:
                raise TypeError(f"kv cache entry {name} is an empty {type(v).__name__}")
            for i, t in enumerate(v):
                if not isinstance(t, torch.Tensor):
                    raise TypeError(
                        f"kv cache entry {name}.{i} is {type(t)}, expected Tensor"
                    )
                out[f"{name}.{i}"] = (t, len(v))
        else:
            raise TypeError(
                f"kv cache entry {name} is {type(v)}, expected Tensor|tuple|list"
            )
    return out


def _spec_page_sizes(kv_cache_config) -> dict[str, int]:
    """layer name -> spec page_size_bytes (covers the layer's full allocation)."""
    page: dict[str, int] = {}
    for g in kv_cache_config.kv_cache_groups:
        for ln in g.layer_names:
            page[ln] = g.kv_cache_spec.page_size_bytes
    return page


class XpuSimpleCPUOffloadWorker(SimpleCPUOffloadWorker):
    """SimpleCPUOffloadWorker with XPU input normalization and XPU copy backend."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config,
        cpu_capacity_bytes: int,
    ):
        super().__init__(vllm_config, kv_cache_config, cpu_capacity_bytes)
        # Replace the CUDA DMA backend before any registration can use it.
        self._backend = XpuCopyBackend()

    def register_kv_caches(self, kv_caches: dict) -> None:
        import os
        import traceback as _tb

        def _mark(msg: str) -> None:
            with open("/tmp/w2x_marker_register.txt", "a") as f:
                f.write(msg + "\n")

        _mark(f"register entered pid={os.getpid()} n_entries={len(kv_caches) if kv_caches else 0}")
        try:
            self._register_impl(kv_caches, _mark)
            _mark("register COMPLETED")
        except Exception:
            _mark("register RAISED:\n" + _tb.format_exc())
            raise

    def _register_impl(self, kv_caches: dict, _mark) -> None:
        kv_caches = normalize_kv_caches(kv_caches)

        if not kv_caches:
            logger.warning("No KV caches to offload.")
            return

        self.device = next(iter(kv_caches.values()))[0].device
        assert self.kv_cache_config is not None
        spec_page = _spec_page_sizes(self.kv_cache_config)

        # Dedup by backing storage (upstream worker.py:97-133). Separate k/v
        # allocations yield up to two entries per layer.
        seen_ptrs: dict[int, tuple[str, torch.Tensor, int]] = {}
        for name, (tensor, siblings) in kv_caches.items():
            ptr = tensor.untyped_storage().data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs[ptr] = (name, tensor, siblings)

        # Padded raw allocations (seen on Ascend: actual blocks >
        # kv_cache_config.num_blocks) break the upstream assumption
        # storage.nbytes() == num_blocks * page_size; derive the real block
        # count per tensor from the spec page size instead.
        unique_gpu_caches: dict[str, torch.Tensor] = {}
        for name, tensor, siblings in seen_ptrs.values():
            storage = tensor.untyped_storage()
            raw = (
                torch.empty(0, dtype=torch.int8, device=self.device)
                .set_(storage, 0, (storage.nbytes(),))
            )
            el = tensor.element_size()
            base = name if name in spec_page else name.rsplit(".", 1)[0]
            if base not in spec_page:
                raise KeyError(f"no KVCacheSpec page size found for layer {name}")
            page_size_bytes = spec_page[base] // siblings
            if storage.nbytes() % page_size_bytes:
                raise RuntimeError(
                    f"layer {name}: storage {storage.nbytes()} bytes is not a "
                    f"multiple of derived page {page_size_bytes} "
                    f"(spec {spec_page[base]} / {siblings})"
                )
            actual_blocks = storage.nbytes() // page_size_bytes
            outer_dims = [
                d for d in range(tensor.ndim) if tensor.stride(d) * el > page_size_bytes
            ]
            if not outer_dims:
                unique_gpu_caches[name] = raw.view(actual_blocks, -1)
            else:
                seg_stride = tensor.stride(outer_dims[0]) * el
                for idx in range(tensor.shape[outer_dims[0]]):
                    offset = idx * seg_stride
                    chunk = raw[offset : offset + seg_stride]
                    unique_gpu_caches[f"{name}.{idx}"] = chunk.view(actual_blocks, -1)
            logger.info(
                "XpuSimpleCPUOffloadWorker: layer %s page=%d B, blocks=%d "
                "(kv_cache_config.num_blocks=%d)",
                name, page_size_bytes, actual_blocks,
                self.kv_cache_config.num_blocks,
            )

        per_tensor_bpb = [t.stride(0) * t.element_size() for t in unique_gpu_caches.values()]
        total_bytes_per_block = sum(per_tensor_bpb)

        self.num_cpu_blocks = max(1, self.cpu_capacity_bytes // total_bytes_per_block)
        _mark(f"dedup done n_unique={len(unique_gpu_caches)} total_bpb={total_bytes_per_block} cpu_blocks={self.num_cpu_blocks}")

        logger.info(
            "XpuSimpleCPUOffloadWorker: %d unique GPU KV tensors, "
            "allocating %d CPU blocks (%.2f GB), per-layer page %.2f MiB",
            len(unique_gpu_caches),
            self.num_cpu_blocks,
            (self.num_cpu_blocks * total_bytes_per_block) / (1024**3),
            total_bytes_per_block / (1024**2),
        )

        self.gpu_kv_caches = unique_gpu_caches
        self.cpu_kv_caches = {}
        pin_modes: set[str] = set()
        for name, gpu_tensor in unique_gpu_caches.items():
            cpu_shape = (self.num_cpu_blocks,) + gpu_tensor.shape[1:]
            tensor = torch.zeros(cpu_shape, dtype=gpu_tensor.dtype, device="cpu")
            tensor, mode = pin_or_fallback(tensor)
            pin_modes.add(mode)
            self.cpu_kv_caches[name] = tensor
        logger.info(
            "XpuSimpleCPUOffloadWorker: host memory mode: %s",
            ", ".join(sorted(pin_modes)),
        )

        # Plain streams: XMLIR has no priority semantics (P-1), and priority 0
        # is the default anyway — nothing is lost by not asking.
        self.load_stream = torch.cuda.Stream()
        self.store_stream = torch.cuda.Stream()

        self._backend.init(
            self.gpu_kv_caches,
            self.cpu_kv_caches,
            self.device,
            self.load_stream,
            self.store_stream,
        )
        _mark(f"backend init done pin_modes={sorted(pin_modes)}")

    def get_finished(self, finished_req_ids: set) -> tuple:
        out = super().get_finished(finished_req_ids)
        try:
            if (self._pending_store_event_indices or
                    self._pending_load_event_indices or out[1]):
                with open("/tmp/w2x_marker_diag.txt", "a") as f:
                    f.write(
                        f"get_finished pend_store={len(self._pending_store_event_indices)}"
                        f" pend_load={len(self._pending_load_event_indices)}"
                        f" store_hwm={self._store_hwm} load_hwm={self._load_hwm}"
                        f" store_ev_list={len(self._store_events)}"
                        f" load_ev_list={len(self._load_events)}"
                        f" fin_recving={len(out[1]) if out[1] else 0}\n"
                    )
        except OSError:
            pass
        return out
