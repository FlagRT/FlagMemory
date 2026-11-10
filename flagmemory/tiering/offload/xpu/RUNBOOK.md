# KV→Host 卸载 connector（昆仑芯 P800 / xVLLM 栈）—— 运行环境定档与上机 runbook

> 模块路径 `flagmemory.tiering.offload.xpu`（导出类 `XpuSimpleCPUOffloadConnector`，
> vLLM 0.20.2 `SimpleCPUOffloadConnector` 的昆仑芯适配：scheduler 侧原样继承，worker 侧
> 换 `XpuSimpleCPUOffloadWorker`；XMLIR 无流优先级语义、镜像无 cuda-python，两死点构造性规避）。

## 0. 运行环境（定档，ADR-0026，2026-11-09）

xpu KV connector 宿主栈 = 上游 xVLLM 发布线镜像（机器 P800-A 已就位，2026-11-09 真卡三件验证通过）：

- **ref**：`harbor.baai.ac.cn/flagrelease-public/kunlunxin001-gems5.0.0-treenone-triton3.0.0-cx0.13.0-plugin0.2.0-vllm0.20.2-cp310-pt29-x64-v5.0.21.43:tele4`
- **digest（版本主锚）**：`sha256:e180a0dd4267ebdfbce4ea5434710870a6ea7e96118457538cc02442907d54b1`（Harbor manifest API 与本机 RepoDigest 双向核对一致；image id 同串）
- **核心栈（实测）**：vllm 0.20.2+empty（自带 `vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector`）/ vllm-plugin-fl 0.2.0+g38e7dbc / torch 2.9.0+cu129 / xmlir 1.0.0.1 / triton 3.5.0 / flag_gems 5.0.0 / flagcx 0.13.0（conda env `python310_torch29_cuda`）

**注意（环境性，勿误判代码问题）**：本 connector 在昆仑芯训练腿基座 `flagrt/kunlun-operator-runtime:2.0.0-…`（内置 vllm 0.13.0）内**不可用**——该 vllm 版本无 `simple_cpu_offload_connector` 模块，import 链断。昆仑芯侧验证一律用上面定档镜像（runtime-team `dev/stack.lock.kunlunxin.v1.yaml` 推理腿 `lock.infer`；两腿分栈口径同 910C ADR-0007）。

## 1. 使用方式

```bash
# PYTHONPATH 指仓根（scheduler 与 worker 进程都要能 import）
export PYTHONPATH=/path/to/FlagMemory:$PYTHONPATH
vllm serve <MODEL_DIR> \
  --kv-transfer-config '{"kv_connector":"XpuSimpleCPUOffloadConnector","kv_connector_module_path":"flagmemory.tiering.offload.xpu","kv_connector_extra_config":{"cpu_bytes_to_use":8589934592}}'
```

- 工厂机制：`kv_connector_module_path` 存在时 `importlib` 动态加载，无需注册动作。
- 设备授予与选卡纪律同昆仑芯基座口径（xpu-smi 挑空闲卡 + `CUDA_VISIBLE_DEVICES` 自律 + `/dev/xpuctrl`）。
- 已知噪声：该镜像启动 vLLM 时插件 fl 自动加载在 `import flag_gems` 阶段报 triton pow 属性错误（flag_gems 5.0.0 × triton 3.5.0），日志 ERROR 但不阻断进程，connector 路径不受影响。

<!-- changelog
2026-11-09  首登：运行环境定档（ADR-0026，选项 A）——xVLLM vllm 0.20.2 系发布镜像（digest 锚）；operator-runtime v2（vllm 0.13.0）内 connector import 链断、环境性不可用，已如实登记。2026-11-09 P800-A 真卡三件验证通过。
-->
