# SPDX-License-Identifier: Apache-2.0
"""NPU copy backend replacing DmaCopyBackend + cuda_mem_ops (cuMemcpyBatchAsync).

Structure mirrors vllm/v1/simple_kv_offload/copy_backend.py (Apache-2.0,
Copyright vLLM contributors): a daemon thread consumes a work queue so the
main thread never blocks on transfers, and completion is reported via events
polled by the upstream worker's get_finished().

Device differences:
- Streams/events are torch.npu.* instead of torch.cuda.*.
- Ascend exposes no batch-memcpy driver API from Python, so each block is
  copied with Tensor.copy_(non_blocking=True) on the transfer stream. This is
  correctness-first; per-step copy count is num_blocks x num_tensors, so
  bandwidth tuning (gather/scatter through a staging buffer) is deferred.
"""

import queue
import threading

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


class NpuDmaCopyBackend:
    """Background-thread copy backend for NPU<->host block transfers."""

    def __init__(self) -> None:
        self._store_layers: list = []  # (gpu_tensor, cpu_tensor) pairs
        self._load_layers: list = []  # (cpu_tensor, gpu_tensor) pairs
        self._queue: queue.SimpleQueue | None = None
        self._thread: threading.Thread | None = None
        self._shutdown: bool = False

    def init(
        self,
        gpu_caches: dict,
        cpu_caches: dict,
        device: torch.device,
        load_stream,
        store_stream,
    ) -> None:
        assert list(gpu_caches.keys()) == list(cpu_caches.keys())
        # Same per-entry bytes-per-block on both sides by construction of
        # register_kv_caches (CPU mirrors the GPU [num_blocks, bpb] layout).
        for k in gpu_caches:
            g, c = gpu_caches[k], cpu_caches[k]
            assert g.shape[1:] == c.shape[1:], f"shape mismatch for {k}"
            self._store_layers.append((g, c))
            self._load_layers.append((c, g))

        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._copy_loop,
            args=(self._queue, device, load_stream, store_stream),
            daemon=True,
            name="npu-kv-offload-copy",
        )
        self._thread.start()
        logger.info(
            "NpuDmaCopyBackend: started for device %s (%d cache tensors)",
            device,
            len(self._store_layers),
        )

    def launch_copy(
        self,
        src_blocks: list,
        dst_blocks: list,
        is_store: bool,
        event_idx: int,
        events_list: list,
    ) -> None:
        layers = self._store_layers if is_store else self._load_layers
        assert self._queue is not None
        # W2 diagnostic: one line per DMA batch actually submitted, so
        # offload engagement is provable from outside the process.
        try:
            import time

            with open("/tmp/w2_marker_transfers.txt", "a") as f:
                f.write(
                    f"{time.time():.3f} {'store' if is_store else 'load'}"
                    f" n_blocks={len(src_blocks)} event={event_idx}\n"
                )
        except OSError:
            pass
        self._queue.put((src_blocks, dst_blocks, layers, is_store, event_idx, events_list))

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self._queue is not None:
            self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    @staticmethod
    def _copy_loop(q, device, load_stream, store_stream) -> None:
        torch.npu.set_device(device)
        while True:
            item = q.get()
            if item is None:
                return
            src_blocks, dst_blocks, layers, is_store, event_idx, events_list = item
            stream = store_stream if is_store else load_stream
            # Both directions are enqueued on the current npu stream, which
            # the context manager selects; the low priority lets KV I/O yield
            # to compute streams.
            with torch.npu.stream(stream):
                for src_t, dst_t in layers:
                    for b_src, b_dst in zip(src_blocks, dst_blocks):
                        dst_t[b_dst].copy_(src_t[b_src], non_blocking=True)
                event = torch.npu.Event()
                event.record(stream)
            events_list.append((event_idx, event))
