# KV→Host 卸载 connector（寒武纪 MLU590 / flagos-app vllm 栈）—— 运行环境定档与上机 runbook

> 模块路径 `flagmemory.tiering.offload.mlu`（导出类 `MluSimpleCPUOffloadConnector`，
> vLLM 0.20.2 `SimpleCPUOffloadConnector` 的寒武纪适配：scheduler 侧原样继承，worker 侧
> 换 `MluSimpleCPUOffloadWorker`；torch_mlu 不补 torch.cuda 面、镜像无 cuda-python，
> 两个 stock 死点构造性规避，全部设备 API 走 torch.mlu.*）。

## 0. 运行环境（定档，2026-11-13）

mlu KV connector 宿主栈 = flagos-app 线 vLLM 应用镜像（寒武纪推理腿发布物，runtime-team
`dev/stack.lock.cambricon.v1.yaml` `lock.infer`，与 910C 同为登记式基线）：

- **ref**：`harbor.baai.ac.cn/flagos-app/vllm0.20.2-cambricon-neuware4.4.3:2.2.0-0.2.2`
- **digest（版本主锚）**：`sha256:76a673a8b4aaa9d6b7752b678ec1fcbb5f6bcfc711414b809117a5b8077b9e94`（`docker inspect` RepoDigests 实测，与 stack.lock.cambricon.v1 登记一致）｜image id `sha256:4f298aa57807…`
- **核心栈（容器内实测）**：vllm 0.20.2+flagos（自带 `vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector`，工厂支持 `kv_connector_module_path` 外挂）/ vllm-plugin-FL 0.2.2 / torch 2.7.1+cpu / torch_mlu 1.29.2+torch2.7.1 / neuware 4.4.3（`/usr/local/neuware/version.txt`）/ triton 3.2.0+mlu1.7.2 / flag_gems 5.4.0rc2.post3

**注意（环境性，勿误判代码问题）**：本 connector 在 cambricon-base 系 pytorch 镜像（无
vLLM）内**不可用**——import 链断在 `vllm`。寒武纪侧 connector 验证一律用上面定档镜像。

**真卡验证（2026-11-13，MLU590 单卡）**：connector 冒烟六判据 SMOKE PASS——①工厂
`kv_connector_module_path` importlib 懒加载拿导出类（旧别名路径取到同一类）②锁页实测
（`pin_memory()` 等价物 is_pinned=True）+ pageable 降级链③D2H 换出主机侧字节级逐字
一致④H2D 换入后新旧块号分叉在场⑤审计三方相等（gpu 读回 = host 池 = 独立重算参考）
⑥destroy 双形态（带在途任务排空 / 空闲幂等）清零；另 realcard import 行 1/1、host
全量 909 passed 零回归（含本包 13 例主机契约）。

## 1. 使用方式

```bash
# PYTHONPATH 指仓根（scheduler 与 worker 进程都要能 import）
export PYTHONPATH=/path/to/FlagMemory:$PYTHONPATH
vllm serve <MODEL_DIR> \
  --kv-transfer-config '{"kv_connector":"MluSimpleCPUOffloadConnector","kv_connector_module_path":"flagmemory.tiering.offload.mlu","kv_connector_extra_config":{"cpu_bytes_to_use":8589934592}}'
```

- 工厂机制：`kv_connector_module_path` 存在时 `importlib` 动态加载，无需注册动作。
- 设备纪律同寒武纪基座口径（stack.lock.cambricon.v1）：cnmon 挑空闲卡 +
  `MLU_VISIBLE_DEVICES` 自律；`import triton` 必须在 `import torch_mlu` 之后（C-3）；
  容器内执行一律经 `bash -lc`（/opt/triton 只在 shell 启动时进 PYTHONPATH）。
- 流优先级：`torch.mlu.Stream.priority_range()` == (0, -3)；域外取值被静默接受（C-5），
  本包只用平台自报值（`mlu_mem_ops.stream_priority_range()`，异常兜底 0）。
- 已知噪声：vllm_fl 自动加载链 `import flag_gems` 在特定路径（无卡授予 / python 直调）
  报 ERROR（设备探测 `No MLU devices are available` 或 triton 链 `ModuleNotFoundError`），
  **不阻断**进程，connector 路径不受影响（2026-11-13 实测两种变体，照常继续）。

<!-- changelog
2026-11-13  首登：运行环境定档（flagos-app vllm0.20.2-cambricon 镜像，digest 锚）；MLU590 真卡 connector 冒烟六判据 SMOKE PASS；cambricon-base 系 pytorch 镜像（无 vllm）内 connector 环境性不可用，已如实登记。
-->
