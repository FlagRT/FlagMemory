# offload/ —— KV→Host 卸载 connector（三栈）

栈外模块，经 vLLM `kv_connector_module_path` 外挂加载，不改任何栈代码。
使用前提：**仓根（本目录的父目录）在 PYTHONPATH 上**，scheduler 与 worker 进程都要能 import。

## 子包一览

| 子包 | 芯片/栈 | 导出类 | kv_connector_module_path | 实测状态 |
|---|---|---|---|---|
| `npu_offload/` | Ascend 910C · vllm-ascend v0.20.2rc1 | `NpuSimpleCPUOffloadConnector` | `offload.npu_offload` | W2 闭环 + W3 开关对照（2026-10-14 报告） |
| `xpu_offload/` | 昆仑 P800 · vllm-plugin-fl 0.2.0 | `XpuSimpleCPUOffloadConnector` | `offload.xpu_offload` | 四步闭环（2026-10-23），真模型 A/B 11 月补 |
| `ppu_offload/` | PPU ZW810E · vllm-plugin-fl 0.3.0 + vllm 0.24.0 | `PpuSimpleCPUOffloadConnector` | `offload.ppu_offload` | 冒烟闭环 + A/B 对照（2026-10-26）：池内零代价、超池 200% +2.1%；真模型待排 |

`npu_offload`/`xpu_offload` 两包同构：connector（swap worker_handler）+ worker（KV 张量归一/
Host 池）+ copy_backend（守护线程 + 逐块 `copy_(non_blocking=True)` + Event 完成上报）+
mem_ops（锁页/降级）。差异只在芯片 API 面（torch.npu vs torch.cuda）与各栈 KV cache 交付格式。

`ppu_offload` 是**另一种移植路线**：源不是 0.20.2 老接口，而是 vLLM 0.24 内置
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
logger，`offload.*` 的 logger 输出会被吞，所以诊断走 stderr。

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

## 从 w2_* 旧名迁移（2026-10-24）

历史报告与服务器旧探针引用旧包名，复现时按下表换算：

| 旧 | 新（本仓） |
|---|---|
| PYTHONPATH 指 `w2_npu_offload` 父目录；module path `w2_npu_offload` | PYTHONPATH 指仓根；module path `offload.npu_offload` |
| PYTHONPATH 指 `w2_xpu_offload` 父目录；module path `w2_xpu_offload` | PYTHONPATH 指仓根；module path `offload.xpu_offload` |
| 导出类名 `Npu/XpuSimpleCPUOffloadConnector` | **不变**（启动配置无需改类名） |
| 诊断标记 `/tmp/w2x_marker_*.txt`、`/tmp/offload_diag.log` | **不变**（与服务器端探针脚本配套） |

## 启动配方

- **910C**：`npu_offload/RUNBOOK.md`（基线命令 + kv-transfer-config、验收检查点、翻车表）。
- **P800**：`probes/p800/`（serve-w2x.sh / recall_p800.py / calib_p800.py，2026-10-23 闭环原件）。
  注意：`--kv-cache-memory-bytes` 在该栈为必填（P-0 环境级绕过）；gmu 按当时卡况调。

## 边界（两包共同）

- 不做带宽优化（staging 聚合、批量 DMA）；不碰 `v1/kv_offload/` 通用框架；
  不改 vllm / vllm-ascend / vllm_fl / vllm-plugin-fl 任何文件，容器重建即回基线。
