# probes/ppu/ —— PPU ZW810E 冒烟配方（2026-10-26）

## serve 启动（三件套 + reference 补丁挂载）

镜像：`harbor.baai.ac.cn/flagrelease-public/openreasoning-nemotron-1.5b-pp001-...-vllm0.24.0-...:202609201956-v4`
（发版镜像；flagos-dev thead-ci 镜像 vllm 本体 import 不通，弃用）

```bash
docker run -d --name cyx-ppu-vllm-smoke --network host --shm-size 8g \
  -e GEMS_VENDOR=thead -e VLLM_PLUGINS=fl \
  --device /dev/alixpu:/dev/alixpu --device /dev/alixpu_ctl:/dev/alixpu_ctl \
  --device /dev/alixpu_ppu7:/dev/alixpu_ppu7 \
  -v /bmcp_lvm_fs/chenyuanxin:/work \
  -v /bmcp_lvm_fs/chenyuanxin/ppu_plugin_fix/reference_register_ops.py:/opt/vllm-plugin-FL/vllm_fl/dispatch/backends/reference/register_ops.py:ro \
  -v /bmcp_lvm_fs/zhenghaojia/models/Llama-3.2-1B-Instruct:/models/llama1b:ro \
  --entrypoint bash <IMAGE> -c '
    source /usr/local/PPU_SDK/envsetup.sh;
    export UMD_PLATFORM_TYPE=1 HGGC_DRIVER_CANDIDATE=UMD PYTHONPATH=/work/FlagMemory;
    vllm serve /models/llama1b --served-model-name tiny --port 18330 \
      --gpu-memory-utilization 0.3 --enforce-eager --max-model-len 1024 \
      --kv-cache-memory-bytes 33554432 \
      --kv-transfer-config "{\"kv_connector\":\"PpuSimpleCPUOffloadConnector\",\"kv_connector_module_path\":\"offload.ppu_offload\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":1073741824}}"'
```

要点：
- 镜像自带 ENTRYPOINT 是 `sleep infinity`，必须 `--entrypoint bash`；
  `/opt/t-head/entrypoint.sh` 无执行位不能直接当 entrypoint，其效果用
  `source envsetup.sh` + 两个 UMD 变量等价实现。
- reference 补丁挂载属**栈 bug 绕过**（plugin dispatch 的 reference 后端注册崩溃），
  不挂则 rms_norm `Registered: []` 引擎起不来。
- FL 版默认 block_size=8（512KB/块）。`--kv-cache-memory-bytes 33554432` → 64 块
  → 池 1,024 tokens（max-model-len 也须 ≤1024）。

## recall_load4.py —— store→recall→load 触发剧本

A（短请求，finish 时 eager store 进 CPU 池）→ B0-B2 并发长请求（filler×6 ≈58 hash 块，
单条需求超 GPU 池剩余，强制覆写 A 的前缀块）→ replay A（GPU 只剩模板命中 computed=16，
remaining=7 → CPU 索引 hit → load DMA）。

验收行：
```
[PpuOffload-DMA] store blocks=N layers=16 ...     # 每请求 finish 后若干批
[PpuOffload-SCHED] entry req=... computed=16 n_hashes=8 remaining=7
[PpuOffload-DMA] load blocks=7 layers=16 ...      # 召回 DMA 实锤
```

## burst_load.py —— 并发压池驱动（8 并发长 prompt）

## 已知坑

- 15+ running 并发触发 flaggems `device-side assert`（PPU vendor 线病态，与 offload
  无关）；压池并发数控制在 3-8。
- cached-free 的 GPU 块 hash 保留，重放 prompt 只走 GPU 命中不产生 load——要 load 必须
  让前缀块被**覆写**（需求 > 池中非 A 部分）。
- `--kv-transfer-config` 嵌套引号：外层 bash -c 单引号内用 `\"` 转义即可，勿用 heredoc。
