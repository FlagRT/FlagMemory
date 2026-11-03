# FlagMemory —— 显存池管理器组件

显存（memory）方向的组件交付仓：显存池管理器本体、芯片能力抽象、分层存储（含三栈 KV 卸载）、
性能分析工具、对外接口声明。自建仓（`kind: direction`），已收录进公共仓 `dev/repos.yaml`，
由 `clone_all.sh` 对齐到各开发机，容器内挂载引用名 `/workspace/FlagMemory`。

分支模型：`main` 只放发版同步 / `dev-1.0` 公共开发 / `<名字>/<功能>` 个人分支。
实测证据与方向看板在公共仓 `dev/memory/`，**不并仓**。

> 命名注记：GitHub 仓名 `FlagMemory`；历史实验记录与早期服务器工作目录中的 `memory-manager` 为当时名称，证据文档保留不改。

## 结构

import 路径根是 `flagmemory`，顶层五块按池组件分层：

| 包 | 内容 | 状态 |
|---|---|---|
| `flagmemory/pool/` | 池本体：块分级、流有序分配，分配/复用/回收/碎片四类操作 | 概念与设计基础 + 最小可运行实现进行中 |
| `flagmemory/caps/` | 芯片能力抽象层：能力查询与降级分支 | 能力分级与降级路径已定（四级 + 五条） |
| `flagmemory/tiering/` | 分层存储；`tiering/offload/{npu,ppu,xpu}` 为三栈 KV→Host 卸载 | 910C/P800 双栈闭环已实测；PPU 冒烟闭环已实测 |
| `flagmemory/profiling/` | 性能分析工具与探针 | 探针配方随栈 |
| `flagmemory/interfaces/` | 对外接口声明（接口头文件目录） | 接口清单随设计文档落位 |

`offload/` 是**兼容别名包**：旧 import 路径 `offload.{npu,ppu,xpu}_offload` 在 0.2.x 仍可用、
0.3.0 移除，取类时发 `DeprecationWarning`。它存在的唯一理由是容器里已写死的
`kv_connector_module_path` 字符串不必跟着改。

依赖方向是硬约束：**池本体不得 import 推理框架**，只向下依赖 `flagmemory.caps`；
`tiering/offload` 是框架绑定的可选适配层，可以向下用池与能力层，反向不行。

## 安装

标准 pip 包（包名 `flagmemory`），零依赖——torch/vLLM 由宿主栈环境自带，本包不替环境定版本：

```bash
pip install -e .   # 在仓根执行；editable 安装，源码即生效
```

安装后 `import flagmemory` 即可寻址（`PYTHONPATH` 不再必需）。不安装、仅把仓根加入
`PYTHONPATH` 的用法继续有效。`flagmemory/profiling/probes/` 为脚本集（无 `__init__.py`，
不作为模块接口），文件随包携带，按 RUNBOOK 原样调用。

池本体与能力层在**无框架、无真卡**的环境里可导入，单测与分层门禁靠这一点。

## connector 快速使用

栈外模块，不改任何栈代码。两步：

1. 仓根进 PYTHONPATH（scheduler 与 worker 进程都要能 import）：
   `export PYTHONPATH=/workspace/FlagMemory:$PYTHONPATH`
2. `--kv-transfer-config` 写对应模块路径与类名：

| 栈 | kv_connector | kv_connector_module_path |
|---|---|---|
| Ascend 910C（vllm-ascend v0.20.2rc1） | NpuSimpleCPUOffloadConnector | `flagmemory.tiering.offload.npu` |
| 昆仑 P800（vllm-plugin-fl 0.2.0） | XpuSimpleCPUOffloadConnector | `flagmemory.tiering.offload.xpu` |
| PPU ZW810E（vllm-plugin-fl 0.3.0 / vllm 0.24.0） | PpuSimpleCPUOffloadConnector | `flagmemory.tiering.offload.ppu` |

旧写法 `offload.npu_offload` / `offload.xpu_offload` / `offload.ppu_offload` 在 0.2.x 等效可用。

细节、翻车表与验收检查点见 [flagmemory/tiering/offload/README.md](flagmemory/tiering/offload/README.md)。

一切真机测定按 [docs/PROFILE-PROTOCOL.md](docs/PROFILE-PROTOCOL.md)（v0.1）执行：暖跑配对、差分唯一、数字五要素。

## 测试

```bash
pip install -e ".[dev]"
pytest                 # 默认只跑不依赖真卡的用例
pytest -m realcard     # 真卡用例，须在带卡机器的栈内跑
python tools/check_layering.py   # 分层依赖门禁
```

## 状态与环境

当前状态见 [STATUS.md](STATUS.md)；设计文档在 [docs/design/](docs/design/)；
改造说明在 [docs/refactor/](docs/refactor/)。
env 版线按 ENV-SPEC 从 env/v1 起（建仓收录后补登）。
