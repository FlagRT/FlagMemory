# FlagMemory —— 显存方向交付仓

显存（memory）方向的交付物收拢仓：KV 卸载、OOM 防护、显存画像、pip 功能包。
自建仓（`kind: direction`），main 单线；收录进公共仓 `dev/repos.yaml` 后由 `clone_all.sh`
对齐到各开发机，容器内挂载引用名 `/workspace/FlagMemory`。
> 命名注记：GitHub 仓名 `FlagMemory`；历史实验记录与早期服务器工作目录中的 `memory-manager` 为当时名称，证据文档保留不改。

## 结构

| 目录 | 内容 | 状态 |
|---|---|---|
| `offload/` | KV→Host 卸载 connector（三栈）+ 探针配方 | 910C/P800 双栈闭环已实测；PPU 冒烟闭环已实测 |
| `guard/` | OOM 防护 | 11 月进场（真机验证） |
| `profile/` | 显存画像工具链 | 11 月收拢 |
| `docs/` | runbook 与报告指针 | 正式报告在公共仓 dev-1.0 |

## connector 快速使用

栈外模块，不改任何栈代码。两步：

1. 仓根进 PYTHONPATH（scheduler 与 worker 进程都要能 import）：
   `export PYTHONPATH=/workspace/FlagMemory:$PYTHONPATH`
2. `--kv-transfer-config` 写对应模块路径与类名：

| 栈 | kv_connector | kv_connector_module_path |
|---|---|---|
| Ascend 910C（vllm-ascend v0.20.2rc1） | NpuSimpleCPUOffloadConnector | `offload.npu_offload` |
| 昆仑 P800（vllm-plugin-fl 0.2.0） | XpuSimpleCPUOffloadConnector | `offload.xpu_offload` |
| PPU ZW810E（vllm-plugin-fl 0.3.0 / vllm 0.24.0） | PpuSimpleCPUOffloadConnector | `offload.ppu_offload` |

细节、翻车表与验收检查点见 [offload/README.md](offload/README.md)。

一切真机测定按 [docs/PROFILE-PROTOCOL.md](docs/PROFILE-PROTOCOL.md)（v0.1）执行：暖跑配对、差分唯一、数字五要素。

## 状态与环境

当前状态见 [STATUS.md](STATUS.md)；env 版线按 ENV-SPEC 从 env/v1 起（建仓收录后补登）。
