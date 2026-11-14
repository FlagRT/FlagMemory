# KV→Host 卸载 connector（平头哥 PPU ZW810E / vllm 0.24 vendor 栈）—— 运行环境定档与上机 runbook

> 模块路径 `flagmemory.tiering.offload.ppu`（导出类 `PpuSimpleCPUOffloadConnector`，
> vLLM 0.24 内置 `SimpleCPUOffloadConnector` 六文件 vendor + 四补丁 P-0/P-1/P-2/P-3，
> 补丁清单与移植路线见 `../README.md` PPU 节；与 npu/xpu/mlu 的 0.20.2 线不同源——
> 0.24 vendor 实现更深绑 CUDA 批量 DMA，stock 路径在 PPU 不可用，P-2 改逐块
> `copy_(non_blocking=True)`）。

## 0. 运行环境（定档，2026-11-14）

ppu KV connector 宿主栈 = FlagRelease 发布线 vllm 0.24 应用镜像（厂商系，PPU 推理发布线）：

- **ref**：`harbor.baai.ac.cn/flagrelease-public/flagrelease_ppu_vllm024_gems_540:1012`
- **digest（版本主锚）**：`sha256:399937f1d77a0565ffcfae386d6abc0b9f4f16b1546e9ae46f49d7d75355bb03`（`docker inspect` RepoDigests 实测）｜image id `sha256:6cd05dcdcf306fa292b1a9f766e92cdb6654a50a330323d4d672af8ce8108527`
- **核心栈（容器内实测 2026-11-14）**：vllm 0.24.0+empty（自带
  `vllm.distributed.kv_transfer.kv_connector.factory` 与
  `KVTransferConfig.kv_connector_module_path` 工厂外挂面）/ vllm-plugin-FL
  0.3.0rc2.post3（`/opt/vllm-plugin-FL`）/ torch 2.11.0+v0.1.0.ppu2.1.0.oe（复用
  `torch.cuda` 命名空间）/ triton 3.5.0+gitcc5446cf / flag_gems 5.4.0+g0d95c2092 /
  PPU SDK v2.1.0（CUDA wrapper cuda-13.0）/ Python 3.12.3

**登记口径（如实）**：PPU 侧现行锁定声明 `runtime-team dev/stack.lock.thead-ppu.v1.yaml`
锁的是 `thead-ppu-runtime v2` 派生镜像（vllm **0.19** 线，训练/推理同底座），**不含**本
connector 的 0.24 宿主镜像；本 RUNBOOK 即该 connector 运行环境的登记处（digest 锚如上），
入锁另行立项。**注意（环境性，勿误判代码问题）**：本 connector 在 thead-ppu-runtime v2
镜像内**不可用**——import 链断在 `from vllm.utils.torch_utils import PIN_MEMORY`
（vllm 0.19 无该符号，2026-11-14 容器内实测）；PPU 侧 connector 验证一律用上面定档镜像。

**真卡验证（2026-11-14，PPU 0 号卡，定档镜像容器内）**：connector 冒烟六判据 SMOKE PASS——
①工厂 `kv_connector_module_path` importlib 懒加载拿导出类（旧别名路径 `offload.ppu_offload`
取到同一个类，DeprecationWarning 在场；工厂面 `KVTransferConfig.kv_connector_module_path`
属性与 factory 模块在镜像内核实）②锁页实测（一级 `cudaHostRegister` 实锁，
`is_pinned=True`；三级降级链逐级强制验证：一级失败退 `pin_memory()` 副本、再失败退不锁页
身份返回）③D2H 换出主机侧字节级逐字一致（对独立重算参考）④H2D 换入新旧块号分叉在场
（host 块 0-2 落 GPU 块 4-6：旧内容落新块号、原块号保留新内容，8/8 子检查）⑤审计三等
（GPU 读回 = 主机池 = 独立重算参考逐字相等；字节账 store=load=主机池占用=3×131072B
三方相等，DMA 计数 store 1 批 3 块 / load 1 批 3 块）⑥destroy 双形态（在途关停排空不丢
数据 + 空闲二次关停幂等）清零（事件表清空、pending 清空、主机池清零、线程退出）；另
realcard import 行 1/1、host 全量 909 passed 2 skipped 零回归。

## 1. 使用方式

```bash
# connector 级验证/单测（一次性容器，PPU0 授卡；体例同 tests/tiering/test_ppu_*_realcard.py 头注释）
docker run --rm --network none \
    --device /dev/alixpu_ppu0 --device /dev/alixpu --device /dev/alixpu_ctl \
    -v <repo>:/ws:ro -w /ws -e PYTHONPATH=/ws \
    harbor.baai.ac.cn/flagrelease-public/flagrelease_ppu_vllm024_gems_540:1012 \
    bash -c "source /usr/local/PPU_SDK/envsetup.sh && \
    UMD_PLATFORM_TYPE=1 HGGC_DRIVER_CANDIDATE=UMD \
    python -m pytest -p no:cacheprovider -m realcard \
    tests/tiering/test_offload_realcard.py -v -k ppu"

# 服务化（起 vLLM；PYTHONPATH 指仓根，scheduler 与 worker 进程都要能 import）
export PYTHONPATH=/path/to/FlagMemory:$PYTHONPATH
vllm serve <MODEL_DIR> \
  --kv-transfer-config '{"kv_connector":"PpuSimpleCPUOffloadConnector","kv_connector_module_path":"flagmemory.tiering.offload.ppu","kv_connector_extra_config":{"cpu_bytes_to_use":8589934592}}'
```

- 工厂机制：`kv_connector_module_path` 存在时 `importlib` 动态加载，无需注册动作。
- 服务化启动三件套与栈级已知边界见 `../README.md`「PPU 栈已知边界」节（`--entrypoint
  bash` + `source /usr/local/PPU_SDK/envsetup.sh` + `UMD_PLATFORM_TYPE=1
  HGGC_DRIVER_CANDIDATE=UMD`；发版镜像 plugin dispatch 注册表 bug 的
  reference_register_ops.py 逐 op guard 覆盖属栈 bug 绕过、非本包交付件；15+ 并发
  device-side assert 属 vendor 线，均与本 connector 无关）。
- 选卡纪律：ppu-smi 挑空闲卡（宿主机多人共享）；诊断标记 `[PpuOffload-DMA]` /
  `[PpuOffload-SCHED]` 走 stderr（vllm 日志系统只放行 `vllm.*` logger）。
- 事件面铁则（与 `flagmemory/tiering/backends/ppu_backend.py` 登记同源）：该栈**未 record
  的事件原生查询返回「已完成」**；完成确认凭据 = copy backend 后台线程 record 后入事件表
  （2026-11-06 真卡复现）。
- 锁页：三级降级链（`cudaHostRegister` → `pin_memory()` → 不锁页）；本镜像实测一级可用。
- `torch.equal` 该栈不支持跨设备比较——字节校验一律取回主机侧做。

<!-- changelog
2026-11-14  首登：运行环境定档（FlagRelease vllm0.24 发布线镜像，docker inspect RepoDigest 锚）；PPU 0 号卡 connector 冒烟六判据 SMOKE PASS；thead-ppu-runtime v2（vllm 0.19 线，stack.lock.thead-ppu.v1 现行锁定）内 connector import 链断（缺 PIN_MEMORY）、环境性不可用，已如实登记。
-->
