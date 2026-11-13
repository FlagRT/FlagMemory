# SPDX-License-Identifier: Apache-2.0
"""MLU copy backend replacing DmaCopyBackend + cuda_mem_ops (cuMemcpyBatchAsync).

Port of the npu/xpu copy backends. Structure mirrors
vllm/v1/simple_kv_offload/copy_backend.py (Apache-2.0, Copyright vLLM
contributors): a daemon thread consumes a work queue so the main thread never
blocks on transfers, and completion is reported via events polled by the
upstream worker's get_finished().

Device notes (cambricon torch_mlu 1.29.2 @ neuware4.4.3):
- torch_mlu does not rewrite torch.cuda (unlike the kunlun torch_xmlir
  symbol rewrite), so streams/events/context managers are torch.mlu.*.
  Probed on MLU590 (2026-11-13): Stream(priority=), stream() context,
  Tensor.copy_(non_blocking=True) D2H/H2D, Event.record/query/synchronize
  all work; priority_range() == (0, -3).
- A torch.mlu.Event that was never recorded natively reports query() ==
  True (same quirk registered by the tiering mlu_backend event adapter).
  Events here are only ever appended to events_list AFTER record(), so the
  upstream _poll_stream_events never sees an unrecorded event.
- No batch-memcpy driver API is reachable from Python (cuda-python absent —
  the stock DmaCopyBackend dies at `from cuda.bindings import driver`), so
  each block is copied with Tensor.copy_(non_blocking=True) on the transfer
  stream. Correctness-first; per-step copy count is num_blocks x
  num_tensors, bandwidth tuning deferred.
"""

import queue
import threading

import torch
import torch_mlu  # noqa: F401  (registers torch.mlu; must precede any triton chain)

from vllm.logger import init_logger

logger = init_logger(__name__)


class MluCopyBackend:
    """Background-thread copy backend for MLU<->host block transfers."""

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
            name="mlu-kv-offload-copy",
        )
        self._thread.start()
        logger.info(
            "MluCopyBackend: started for device %s (%d cache tensors)",
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
        # Diagnostic marker: one line per copy batch actually submitted, so
        # offload engagement is provable from outside the process.
        try:
            import time

            with open("/tmp/w2m_marker_transfers.txt", "a") as f:
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
        torch.mlu.set_device(device)
        while True:
            item = q.get()
            if item is None:
                return
            src_blocks, dst_blocks, layers, is_store, event_idx, events_list = item
            stream = store_stream if is_store else load_stream
            # Copies are enqueued on the transfer stream selected by the
            # context manager; completion is recorded as an event the
            # upstream worker polls.
            with torch.mlu.stream(stream):
                for src_t, dst_t in layers:
                    for b_src, b_dst in zip(src_blocks, dst_blocks):
                        dst_t[b_dst].copy_(src_t[b_src], non_blocking=True)
                event = torch.mlu.Event()
                event.record(stream)
            events_list.append((event_idx, event))
