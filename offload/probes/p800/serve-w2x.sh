#!/bin/bash
# W2-P800: external XpuSimpleCPUOffloadConnector (offload.xpu_offload, 原 w2_xpu_offload)
# on kunlun stack. PYTHONPATH 指仓根（本探针假设仓已挂 /work/memory-manager，按实际改）。
# No vllm-patched.py needed: our worker never calls Stream.priority_range (P-1 avoided
# by construction) and never imports cuda-python (P-2 avoided by construction).
# --kv-cache-memory-bytes stays mandatory (P-0 env-level bypass).
# K>G geometry: GPU pool 64 MiB (32,768 tok @ 2 KiB/tok) / host pool 128 MiB = 2x GPU.
export PYTHONPATH=/work/memory-manager:$PYTHONPATH
source /root/miniconda/bin/activate python310_torch29_cuda
exec vllm serve /models/random-qwen3-tiny \
  --served-model-name tiny \
  --port 18307 \
  --gpu-memory-utilization 0.3 \
  --enforce-eager \
  --max-model-len 512 \
  --kv-cache-memory-bytes 67108864 \
  --kv-transfer-config '{"kv_connector":"XpuSimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_module_path":"offload.xpu_offload","kv_connector_extra_config":{"cpu_bytes_to_use":134217728}}'
