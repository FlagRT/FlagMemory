# FlagMemory —— 显存池管理器组件

本仓交付显存池本体、芯片能力抽象、分层存储卸载适配器、探针脚本与接口声明。公共开发线为 `dev-1.0`，个人工作在 `<名字>/<功能>` 分支；容器挂载引用名为 `/workspace/FlagMemory`。

## 结构与当前边界

| 路径 | 内容与状态 |
|---|---|
| `flagmemory/pool/` | 已实现自管分级块池、当前租约访问、流序回收及可恢复关闭托管；主机回归可运行，真卡尚未验证 |
| `flagmemory/caps/` | 统一能力协议、分级与降级查询 |
| `flagmemory/tiering/offload/` | 框架绑定的三套 KV→Host 卸载适配器；尚未成为池的业务消费方 |
| `flagmemory/profiling/probes/` | 随包分发的脚本集，按各目录说明调用，不作为稳定模块 API |
| `flagmemory/interfaces/` | 类型存根及接口文档；接口语言面为 Proposed |
| `docs/design/` | 池与分层存储设计；通用分层管理器和原生优化尚未实现 |

池本体不导入推理框架；底座由外部注入，芯片差异经能力层处理。当前原生流有序标记为假；首次芯片验证只针对显式张量级自管路径。SDK 头文件/符号调查不等于运行验证。池没有换出回调，允许分层换出的回收请求明确抛 `NotWired`。

## 安装与入口

```bash
pip install -e .
```

包不替宿主栈选择 torch/vLLM 版本，池与能力层可在无框架、无卡环境导入。现行建池、租约访问与调用方别名责任见 [实现说明](docs/impl/11-28-分级块池与流有序分配.md)；接口契约见 [接口文档](flagmemory/interfaces/接口文档-显存池对外接口.md)。

卸载 connector 的模块分别为 `flagmemory.tiering.offload.npu`、`.xpu`、`.ppu`；具体类、栈要求与配置见 [卸载使用说明](flagmemory/tiering/offload/README.md)。旧 `offload.*` 为发出弃用警告的兼容别名，新增调用使用正式路径。三栈卸载结果不能作为池接入的运行证据。

## 检查与证据

```bash
pip install -e ".[dev]"
pytest
python tools/check_layering.py
```

默认排除 `realcard` 用例。**不要直接运行全部 `-m realcard`**：旧 CUDA 兼容池测试的适配器和断言仍未适配当前协议，并含按空闲显存倍数申请的用例。独立候选适配器与测试入口已实现，本地主机检查已完成；真实设备 API、物理到逻辑映射、单芯片隔离及生命周期尚未验证，设备执行仍需独立批准。须显式选择独立测试文件并遵守资源上限；入口与启用条件见 [测试说明](tests/README.md)。

探针及历史实测材料的口径见 [采集协议](docs/PROFILE-PROTOCOL.md)。代码/证据分家尚未收口，`reports/` 仍含历史材料；旧真卡取证逻辑默认仅打印，显式指定仓外路径才写文件。不得把未执行或被排除的用例计为通过。

设计入口与证据指针见 [文档索引](docs/README.md)，方向状态见 [STATUS.md](STATUS.md)。
