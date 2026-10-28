# KV→Host 卸载 connector（Ascend 910C / vllm-ascend 栈）—— 上机 runbook

> 原 `w2_npu_offload`（2026-10 W2 实验包）迁入本仓后的版本；路径已改为
> `offload.npu_offload`，旧名仅存在于历史报告引用中。实验结论见
> dev-1.0 `dev/memory/docs/goals/proto-910c-202610/` 两报告。

目标：在 vllm-ascend 锁定栈（quay.io/ascend/vllm-ascend:v0.20.2rc1-a3，davinci4）上跑通
KV→Host 卸载最小闭环（卸→存→召回→结果正确）。本包不改任何栈代码。

## 0. 已解决的两死点（源码级确认，GitHub tag v0.20.2rc1 与容器逐行一致）

- 死点#1：`model_runner_v1.py:3486` 传给 connector 的是 `dict[str, (k, v) 元组]`
  （:4003），上游 `_repr_tensor` 只认 `Tensor | list` → AssertionError。
  修法 = `npu_worker.normalize_kv_caches`：元组拆成 `{name}.0/{name}.1` 两个条目，
  K/V 都进卸载集（与上游对 (2, num_blocks, ...) 张量的外维拆分处理一致）。
- 死点#2：copy backend 纯 CUDA（torch.cuda.Stream / cudaHostRegister /
  cuMemcpyBatchAsync）。修法 = `NpuDmaCopyBackend`：torch.npu 低优先级流 +
  逐块 `copy_(non_blocking=True)` + `torch.npu.Event` 完成上报；
  锁页用 `Tensor.pin_memory()`，不可用则降级 pageable（正确性不受影响，只慢）。

## 1. 上传与加载

```bash
# 仓收录后由 clone_all 落在 <公共仓根>/FlagMemory/；以下按该路径示例。
# 容器内 PYTHONPATH 指仓根（offload/ 的父目录），scheduler 和 worker 进程都要能 import：
export PYTHONPATH=/workspace/FlagMemory:$PYTHONPATH
# 模块路径相应写 offload.npu_offload（见 §2）；远程机上没有仓时 scp 整仓即可：
#   scp -r FlagMemory <910C节点>:~/runtime-team/dev/memory/probes/910c/
```

## 2. 启动命令（在 10/12 复测基线命令上加两处）

```bash
vllm serve /data_lib/models/OneRec-8B \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"NpuSimpleCPUOffloadConnector","kv_connector_module_path":"offload.npu_offload","kv_connector_extra_config":{"cpu_bytes_to_use":8589934592}}'
# 其余参数照 10/12 复测配方（gmu/max-model-len/eager 等与基线一致，保证开关对照可比）
```

- `cpu_bytes_to_use` 首跑 8 GiB（上游默认）；`cpu_bytes_to_use_per_rank` 可覆盖单卡份额。
- 工厂机制：module_path 存在时 `importlib.import_module` + `getattr(类名)`，外部模块优先于内置注册表——**无需任何注册动作**。

## 3. 验收检查点（按顺序）

1. 启动日志出现 `NpuSimpleCPUOffloadWorker: N unique GPU KV tensors, allocating M CPU blocks`
   → 死点#1 已过（OneRec-8B 预期：N=层数×2（k/v 分条），M≈3700 块 @8GiB，每块 ~2.2 MiB）。
2. `host memory mode: pinned|pageable` → 决定后续带宽预期。
3. 池数字与基线一致（39.22 GiB / 285,568 tok）→ 开 connector 不改池（W1 已证，复验）。
4. 首请求完成、输出正确（对照基线同 prompt）。
5. 长输出多请求后 host 侧 RSS 增长 ≈ 卸载块数×块大小；`/metrics` 无异常。
6. smoke 通过后跑 gen-matrix A/C 两组开关对照（W3 的第一块数据）。

## 4. 首跑可能翻车的点（按概率排序）

| 症状 | 原因 | 快修 |
|---|---|---|
| `torch.npu.Stream(priority=...)` 报错 | torch_npu 版本参数名差异 | `npu_mem_ops.stream_priority_range` 已兜底返回 0；Stream 构造处去掉 priority |
| `pin_memory()` 抛异常 | torch_npu 未实现 host pin | 已自动降级 pageable，只影响带宽（看日志确认） |
| D2H `copy_(non_blocking=True)` 报错 | 老版 torch_npu 对跨设备 non_blocking 限制 | 去掉 non_blocking（牺牲异步，先跑通） |
| Event `query()` 行为异常 | torch.npu.Event 与 CUDA 语义差异 | 改 `_poll_stream_events` 为 `event.synchronize()` 同步版（慢但稳） |
| dedup 后条目数异常（N≠2×层数） | vllm-ascend 对该模型走了 MLA/sparse 分支 | 打印 `seen_ptrs` 长度与 `kv_cache_config` 结构回报，勿盲改 |

## 5.5 W2+W3 同窗合并执行（2026-10-14 定：一个占卡窗口，分三段）

- **段1｜W2 门槛**（~半天内）：本文档 §2-§4 全流程，检查点 1-4 绿 = 最小闭环达成。
  红了就地按 §4 翻车表修；修不动就收工，W3 顺延，本窗数据作废不进报告。
- **段2｜W3 开腿采集**（段1 全绿后立刻做，服务不重启、只追请求）：
  - 复用 `gen-matrix_910c.sh` 的请求腿，connector 开着跑 **A / C / D**（+smoke sanity）；
  - 顺手加 **E-cancel 一条**：取消后看 host RSS 不回落即为 host 泄漏信号（卸载版独有检查）；
  - 请求腿的脚本/配方与 10-12 复测完全一致，**唯一差分 = kv-transfer-config 有无**。
- **段3｜离线分析**（不用机器）："关"腿 = 10-12 davinci4 复测基线，直接对照出
  扩容/代价数字。B/F 腿不需要开关重跑。

**配置对齐检查单（段2 开跑前过一遍）**：gmu、MAX_MODEL_LEN、eager、block size、
tokenizer、max_num_seqs 与基线逐项相同；`--enable-prefix-caching` 基线本就默认开
（D/F 组命中共享块已证）。

**加压变体决策点**："代价"数字 A/C/D 即可出；"扩容"数字要够到满池区——现矩阵
D-c64 仅用池 6.2%，若段2 数据撑不起扩容结论，现场加跑一条 c100+ 或加长 ctx 的
D 变体（机器时间 ~20min），是否加由段2 的实际 KV 峰值现场定。

## 6. 明确不做（本骨架边界）

- 不做带宽优化（staging 聚合拷贝、批量 DMA）——W3 拿到开关对照数据后再议。
- 不碰 OffloadingConnector 通用框架（`v1/kv_offload/`）——SimpleCPUOffload 路径先行。
- 不改 vllm / vllm-ascend / vllm_fl 任何文件；容器重建即回基线。

## 7. 溯源

- 上游代码（Apache-2.0）：`vllm/v1/simple_kv_offload/{worker,copy_backend,cuda_mem_ops}.py`、
  `vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py`，tag v0.20.2。
- 本地参考副本：`%TEMP%/vllm-ascend-src/`（含 vllm-ascend model_runner_v1.py，行号证据在
  REPORT 与 MSG_leader_v2 中引用为 model_runner_v1.py:3486/:4003）。
