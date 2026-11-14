# KV→Host 卸载 connector（Ascend 910C / vllm-ascend 栈）—— 运行环境定档与上机 runbook

> 现行模块路径 `flagmemory.tiering.offload.npu`（导出类 `NpuSimpleCPUOffloadConnector`，
> vLLM 0.20.2 `SimpleCPUOffloadConnector` 的昇腾适配：scheduler 侧原样继承，worker 侧换
> `NpuSimpleCPUOffloadWorker`——vllm-ascend 的 (k,v) 元组交付与纯 CUDA copy backend 两死点
> 构造性规避）。更早的
> `w2_npu_offload` 仅存在于历史报告引用中。实验结论见 dev-1.0
> `dev/memory/docs/goals/proto-910c-202610/` 两报告。

目标：在 vllm-ascend 锁定栈（见 §0 定档）上跑通 KV→Host 卸载最小闭环
（卸→存→召回→结果正确）。本包不改任何栈代码。

## 0. 运行环境（定档，2026-11-14）

npu KV connector 宿主栈 = 华为昇腾官方 vllm-ascend 镜像（910C 推理腿发布物，runtime-team
`dev/stack.lock.910c.v4.yaml` `lock.infer`，登记式基线）：

- **ref**：`quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`
- **digest（版本主锚）**：`sha256:5cf8a2b6db8b06eb1bc7fc7d191d667aebf2b197351bdba13f776918c11ec7a7`（`docker inspect` RepoDigests 实测，与 stack.lock.910c.v4 `lock.infer` 登记逐字一致）｜image id `sha256:2e56022ae5b39930e…`
- **核心栈（容器内实测）**：vllm 0.20.2+empty（自带 `vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector`，工厂支持 `kv_connector_module_path` 外挂）/ vllm-ascend 0.20.2rc1 / torch 2.10.0+cpu / torch-npu 2.10.0 / CANN 9.0.0 / triton 3.5.0 + triton-ascend 3.2.1 / transformers 5.5.3（Python 3.11.15）。镜像内无 vllm-plugin-FL 与 flag_gems（实测缺失，connector 路径不受影响）。

**注意（环境性，勿误判代码问题）**：带卡容器三坑（910C 实测）——①fresh 容器须
`--privileged` + `--network host`，aclInit 才能过（否则报 500000）；②同机已有带卡容器在跑
时可能撞 aclInit 507899（命名空间被先到者持）——先起最小 aclInit 探针验证可初始化再正式跑；
③起容器前 `npu-smi info` 查占卡，选空闲卡并以 `ASCEND_RT_VISIBLE_DEVICES` 映射（示例
davinci4）。

**真卡验证（2026-11-14，910C 单卡 davinci4）**：connector 冒烟六判据 SMOKE PASS——①工厂
`kv_connector_module_path` importlib 懒加载拿导出类（旧别名路径取到同一个类）②锁页实测
（`pin_memory()` 后 `is_pinned()=True`，host 池张量全锁页）+ pageable 降级链（pin 不可用时
原对象返回）③D2H 换出主机侧字节级逐字一致（4 张量 × 3 块）④H2D 换入后新旧块号分叉在场
（gpu[7]=召回旧内容与 gpu[0]=覆写新内容并存，load 完成上报到位）⑤审计三方相等（GPU 读回
= host 池 = 独立重算参考，逐字节全同且 bytes 三等）⑥destroy 双形态（带在途任务排空 / 空闲
幂等）清零；另 realcard import 行 1/1、host 全量 909 passed 2 skipped 零回归。本栈
`torch.npu.Stream.priority_range()` 不支持（RuntimeError），`npu_mem_ops.stream_priority_range()`
兜底返回 0，`torch.npu.Stream(priority=0)` 可用。

## 1. 已解决的两死点（源码级确认，GitHub tag v0.20.2rc1 与容器逐行一致）

- 死点#1：`model_runner_v1.py:3486` 传给 connector 的是 `dict[str, (k, v) 元组]`
  （:4003），上游 `_repr_tensor` 只认 `Tensor | list` → AssertionError。
  修法 = `npu_worker.normalize_kv_caches`：元组拆成 `{name}.0/{name}.1` 两个条目，
  K/V 都进卸载集（与上游对 (2, num_blocks, ...) 张量的外维拆分处理一致）。
- 死点#2：copy backend 纯 CUDA（torch.cuda.Stream / cudaHostRegister /
  cuMemcpyBatchAsync）。修法 = `NpuDmaCopyBackend`：torch.npu 低优先级流 +
  逐块 `copy_(non_blocking=True)` + `torch.npu.Event` 完成上报；
  锁页用 `Tensor.pin_memory()`，不可用则降级 pageable（正确性不受影响，只慢）。

## 2. 上传与加载

```bash
# 仓收录后由 clone_all 落在 <公共仓根>/FlagMemory/；以下按该路径示例。
# 容器内 PYTHONPATH 指仓根（flagmemory/ 的父目录），scheduler 和 worker 进程都要能 import：
export PYTHONPATH=/workspace/FlagMemory:$PYTHONPATH
# 模块路径相应写 flagmemory.tiering.offload.npu（见 §3）；远程机上没有仓时 scp 整仓即可：
#   scp -r FlagMemory <910C节点>:~/runtime-team/dev/memory/probes/910c/
```

## 3. 启动命令（在 10/12 复测基线命令上加两处）

```bash
vllm serve /data_lib/models/OneRec-8B \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"NpuSimpleCPUOffloadConnector","kv_connector_module_path":"flagmemory.tiering.offload.npu","kv_connector_extra_config":{"cpu_bytes_to_use":8589934592}}'
# 其余参数照 10/12 复测配方（gmu/max-model-len/eager 等与基线一致，保证开关对照可比）
```

- `cpu_bytes_to_use` 首跑 8 GiB（上游默认）；`cpu_bytes_to_use_per_rank` 可覆盖单卡份额。
- 工厂机制：module_path 存在时 `importlib.import_module` + `getattr(类名)`，外部模块优先于内置注册表——**无需任何注册动作**。

## 4. 验收检查点（按顺序）

1. 启动日志出现 `NpuSimpleCPUOffloadWorker: N unique GPU KV tensors, allocating M CPU blocks`
   → 死点#1 已过（OneRec-8B 预期：N=层数×2（k/v 分条），M≈3700 块 @8GiB，每块 ~2.2 MiB）。
2. `host memory mode: pinned|pageable` → 决定后续带宽预期。
3. 池数字与基线一致（39.22 GiB / 285,568 tok）→ 开 connector 不改池（W1 已证，复验）。
4. 首请求完成、输出正确（对照基线同 prompt）。
5. 长输出多请求后 host 侧 RSS 增长 ≈ 卸载块数×块大小；`/metrics` 无异常。
6. smoke 通过后跑 gen-matrix A/C 两组开关对照（W3 的第一块数据）。

## 5. 首跑可能翻车的点（按概率排序）

| 症状 | 原因 | 快修 |
|---|---|---|
| `torch.npu.Stream(priority=...)` 报错 | torch_npu 版本参数名差异 | `npu_mem_ops.stream_priority_range` 已兜底返回 0；Stream 构造处去掉 priority |
| `pin_memory()` 抛异常 | torch_npu 未实现 host pin | 已自动降级 pageable，只影响带宽（看日志确认） |
| D2H `copy_(non_blocking=True)` 报错 | 老版 torch_npu 对跨设备 non_blocking 限制 | 去掉 non_blocking（牺牲异步，先跑通） |
| Event `query()` 行为异常 | torch.npu.Event 与 CUDA 语义差异 | 改 `_poll_stream_events` 为 `event.synchronize()` 同步版（慢但稳） |
| dedup 后条目数异常（N≠2×层数） | vllm-ascend 对该模型走了 MLA/sparse 分支 | 打印 `seen_ptrs` 长度与 `kv_cache_config` 结构回报，勿盲改 |

## 6. W2+W3 同窗合并执行（2026-10-14 定：一个占卡窗口，分三段）

- **段1｜W2 门槛**（~半天内）：本文档 §3-§5 全流程，检查点 1-4 绿 = 最小闭环达成。
  红了就地按 §5 翻车表修；修不动就收工，W3 顺延，本窗数据作废不进报告。
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

## 7. 明确不做（本骨架边界）

- 不做带宽优化（staging 聚合拷贝、批量 DMA）——W3 拿到开关对照数据后再议。
- 不碰 OffloadingConnector 通用框架（`v1/kv_offload/`）——SimpleCPUOffload 路径先行。
- 不改 vllm / vllm-ascend / vllm_fl 任何文件；容器重建即回基线。

## 8. 溯源

- 上游代码（Apache-2.0）：`vllm/v1/simple_kv_offload/{worker,copy_backend,cuda_mem_ops}.py`、
  `vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py`，tag v0.20.2。
- 本地参考副本：`%TEMP%/vllm-ascend-src/`（含 vllm-ascend model_runner_v1.py，行号证据在
  REPORT 与 MSG_leader_v2 中引用为 model_runner_v1.py:3486/:4003）。

<!-- changelog
2026-11-14  首登定档：运行环境锁定 910C 推理腿镜像 quay.io/ascend/vllm-ascend:v0.20.2rc1-a3（digest 与 stack.lock.910c.v4 lock.infer 逐字一致）；910C 真卡 connector 冒烟六判据 SMOKE PASS + realcard import 1/1 + host 全量 909 passed 2 skipped 零回归；带卡容器三坑（aclInit 500000/507899/选卡映射）与 priority_range 不支持事实入档；章节重排（定档节前置为 §0，原 §0-§7 顺移 §1-§8，内文交叉引用同步）。
-->
