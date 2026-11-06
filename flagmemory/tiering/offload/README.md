# flagmemory/tiering/offload/ —— KV→Host 卸载 connector（三栈）

分层存储下的**框架绑定适配层**，经 vLLM `kv_connector_module_path` 外挂加载，不改任何栈代码。
池本体不经过这里，也不反向依赖这里。
使用前提：**仓根在 PYTHONPATH 上**（或 `pip install -e .`），scheduler 与 worker 进程都要能 import。

## 子包一览

| 子包 | 芯片/栈 | 导出类 | kv_connector_module_path | 实测状态 |
|---|---|---|---|---|
| `npu/` | Ascend 910C · vllm-ascend v0.20.2rc1 | `NpuSimpleCPUOffloadConnector` | `flagmemory.tiering.offload.npu` | W2 闭环 + W3 开关对照（2026-10-14 报告） |
| `xpu/` | 昆仑 P800 · vllm-plugin-fl 0.2.0 | `XpuSimpleCPUOffloadConnector` | `flagmemory.tiering.offload.xpu` | 四步闭环（2026-10-23），真模型 A/B 11 月补 |
| `ppu/` | PPU ZW810E · vllm-plugin-fl 0.3.0 + vllm 0.24.0 | `PpuSimpleCPUOffloadConnector` | `flagmemory.tiering.offload.ppu` | 冒烟闭环 + A/B 对照（2026-10-26）：池内零代价、超池 200% +2.1%；真模型待排 |

旧模块路径 `offload.{npu,ppu,xpu}_offload` 由仓根兼容别名包转发，0.2.x 等效可用、0.3.0 移除。

`npu`/`xpu` 两包同构：connector（swap worker_handler）+ worker（KV 张量归一/
Host 池）+ copy_backend（守护线程 + 逐块 `copy_(non_blocking=True)` + Event 完成上报）+
mem_ops（锁页/降级）。差异只在芯片 API 面（torch.npu vs torch.cuda）与各栈 KV cache 交付格式。

`ppu` 是**另一种移植路线**：源不是 0.20.2 老接口，而是 vLLM 0.24 内置
`SimpleCPUOffloadConnector` 六文件 vendor+补丁（该版实现更深绑 CUDA 批量 DMA，stock 路径
在 PPU 上不可用）。四个补丁：

| 补丁 | 位置 | 内容 |
|---|---|---|
| P-2（必需） | `ppu_mem_ops.py` | PPU HGGC 驱动不实现 `cuMemcpyBatchAsync`（err=998），`copy_blocks` 改逐块 `tensor.copy_(non_blocking=True)` |
| P-0 | `ppu_worker.py` | `_repr_tensor` 容忍 tuple 形态的层 KV 值 |
| P-1 | `ppu_worker.py` | `Stream.priority_range()` 不可用时退默认流 |
| P-3 | `ppu_mem_ops.py` | `cudaHostRegister` 失败退 `pin_memory()` 再退不锁页 |

诊断标记：`[PpuOffload-DMA]`（stderr，每次 store/load 批次一行，含累计计数）、
`[PpuOffload-SCHED]`（stderr，召回匹配的 entry/hit_len）。vllm 日志系统只放行 `vllm.*`
logger，本包的 logger 输出会被吞，所以诊断走 stderr。

## PPU 栈已知边界（2026-10-26 冒烟）

- 两个镜像的 vllm 均缺 `vllm._C_stable_libtorch`（空心构建），但 plugin 0.3.0 注册的
  `PlatformFL` 平台层可绕开；**启动三件套**：`--entrypoint bash` + `source
  /usr/local/PPU_SDK/envsetup.sh` + `UMD_PLATFORM_TYPE=1 HGGC_DRIVER_CANDIDATE=UMD`。
- 发版镜像 plugin 的 dispatch 注册表有独立 bug：reference 后端整链注册失败
  （`ReferenceBackend` 缺 `moe_align_block_size` 属性），rms_norm 等算子无人实现导致引擎
  起不来。修复=逐 op hasattr guard 版 `reference_register_ops.py`（ppu_plugin_fix/，
  bind-mount 覆盖 `/opt/vllm-plugin-FL/.../reference/register_ops.py`）。**此修复属栈
  bug 绕过，非 offload 交付件**，随报备另行说明。
- 高并发（15+ running）下 flaggems triton kernel 在 PPU 上 `device-side assert`（与
  offload 无关，属 vendor 线病态，单独报）。
- `--kv-cache-memory-bytes` 在该栈有效；FL 版默认 block_size=8（512KB/块），
  推池大小时按 512KB/块算，别按 16tok/块。
- 触发 recall-load 的配方：GPU 池内前缀块被**覆写**（而非 free）后重放原 prompt；
  cached-free 块 hash 保留，重放只走 GPU 命中，不产生 load。

## XPU 栈 API 面事实（2026-11-05 取证）

- 该栈上 `torch.xpu.*` 恒不可用（Torch not compiled with XPU enabled）；实际后端 =
  `torch_xmlir` 符号重写，import 时把 `torch.cuda.*` 映射到 XPU（打印
  `SYMBOL_REWRITE torch success`）。监控/记账 API 必须走 `torch.cuda.*`
  （如 `torch.cuda.memory_stats`），不得走 `torch.xpu.*`（代码侧登记：
  `xpu/xpu_copy_backend.py` 模块头 Device notes）。
- 实例环境锚：镜像 `flagrt/kunlun-operator-runtime:2.0.0`（id `cac5ddae5f62`）、
  torch 2.9.0+cu129、驱动栈 1.0:2.43:1.86.5.3、单卡单次实测；换栈须重探。
  取证记录见指挥台仓 `inbox/取证-昆仑芯P800分配粒度基数-20261105.md`。
- 同场取证的分配粒度实例值（块级最小 512 B / 段级 2 MiB 对齐 / 大小块分界 ≈10 MiB）
  登记在《显存池概念与设计基础》第八节第 4 条；档位输入实例默认 =
  `flagmemory.pool.KUNLUN_P800_INSTANCE_TIER_SPEC`。

## 从 w2_* 旧名迁移（2026-10-24）

历史报告与服务器旧探针引用旧包名，复现时按下表换算：

| 旧 | 新（本仓） |
|---|---|
| PYTHONPATH 指 `w2_npu_offload` 父目录；module path `w2_npu_offload` | PYTHONPATH 指仓根；module path `flagmemory.tiering.offload.npu` |
| PYTHONPATH 指 `w2_xpu_offload` 父目录；module path `w2_xpu_offload` | PYTHONPATH 指仓根；module path `flagmemory.tiering.offload.xpu` |
| 导出类名 `Npu/XpuSimpleCPUOffloadConnector` | **不变**（启动配置无需改类名） |
| 诊断标记 `/tmp/w2x_marker_*.txt`、`/tmp/offload_diag.log` | **不变**（与服务器端探针脚本配套） |

## 启动配方

- **910C**：`npu/RUNBOOK.md`（基线命令 + kv-transfer-config、验收检查点、翻车表）。
- **P800**：`flagmemory/profiling/probes/p800/`（serve-w2x.sh / recall_p800.py / calib_p800.py，2026-10-23 闭环原件）。
  注意：`--kv-cache-memory-bytes` 在该栈为必填（P-0 环境级绕过）；gmu 按当时卡况调。

## 边界（两包共同）

- 不做带宽优化（staging 聚合、批量 DMA）；不碰 `v1/kv_offload/` 通用框架；
  不改 vllm / vllm-ascend / vllm_fl / vllm-plugin-fl 任何文件，容器重建即回基线。

## 分层存储接线现状（TierBackend 协议面）

| 栈 | 状态 | 接线层 |
|---|---|---|
| Ascend 910C（npu） | 已接 TierBackend 协议面（协议形状 / 能力声明映射 / 降级登记 / 注册路径主机可验；真卡闭环用例带 realcard 标记，跑卡窗口另排） | `flagmemory/tiering/backends/npu_backend.py` |
| 昆仑 P800（xpu） | 待改造 | — |
| PPU ZW810E（ppu） | 待改造 | — |

接线方向（设计 T-5）：外部把适配层包成 `TierBackend` 注册给 manager
（`register_npu_tier_backend`）；manager 本体不 import offload 实现与接线层，
connector 经推理引擎 KV 外挂点的加载方式不变。能力按本 README 与
`npu/RUNBOOK.md` 登记的实测事实填：批量搬运原语无（声明即降级态）、锁页声明支持
（注册时实测试锁，探测不到按降级登记上报）、搬运粒度以 bind 时 connector 推导
值为准（未 bind 用 OneRec-8B @910C 实例值 ≈2.2 MiB/块）。条目级访问统计
connector 不提供，恒按「取不到 → 退用其余判据并标降级」处置。
