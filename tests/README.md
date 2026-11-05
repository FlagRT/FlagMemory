# 测试说明

## 主机检查

```bash
pip install -e ".[dev]"
pytest
python tools/check_layering.py
pytest tests/interfaces -q
```

默认 `addopts = "-m 'not realcard'"` 排除真卡标记。输出中的 deselected 不算通过；池本体与能力层可在无框架环境检查。

| 路径 | 覆盖 | 需要真卡 |
|---|---|---|
| `tests/caps/` | 能力、统一后端协议与假后端 | 否 |
| `tests/pool/` 中主机文件 | 分配/复用、档位、查询、回收屏障、当前租约访问、首次目标流就绪、关闭恢复与强托管 | 否 |
| `tests/interfaces/` | 类型存根与实现签名一致 | 否 |
| `tests/tiering/test_alias_package.py` | 旧包兼容别名与弃用警告 | 否 |
| `tests/tools/` | 分层依赖门禁 | 否 |
| `tests/tiering/test_offload_realcard.py` | 宿主栈内 connector 类导入 | 是，仅证明导入能力 |
| `tests/pool/test_pool_realcard.py` | 旧 CUDA 兼容命名空间的测试草稿 | 是，**当前禁止作为首芯片入口直接运行** |

## 真卡入口限制

**不要运行整批 `pytest -m realcard` 或清空 marker 限制。** 旧池真卡文件缺当前适配器协议、旧租约查询/复用断言失效，并包含按空闲显存倍数进行超量申请的测试；它不是现行可用验收入口，也不能当作昇腾结果。默认排除该模块不代表它已适配。

[独立候选适配器](pool/ascend_test_adapter.py)与[独立测试入口](pool/test_pool_ascend_realcard.py)已有昇腾 910C 单芯片受控验证实例：映射、4096 字节读写健康及三个生命周期用例通过，生命周期无跳过；另有 [模型输入测试](pool/test_onerec_inputs_ascend.py) 验证 OneRec-8B 实际消费池输入并与普通输入数值对照通过。固定源码、环境及取证限制见 [验证实例记录](../docs/validation/2026-11-04-ascend-pool-onerec.md)。这些结果只覆盖该实例，新的设备执行仍须核定目标、窗口及资源范围。须按下方启用条件显式选择该测试文件，并先审阅实际收集名单、资源上限、目标流就绪、池块公开访问及清理步骤。未执行、跳过或未观察到的场景不算通过；主机假后端结果不证明厂商运行时和设备生命周期。

## 取证与历史材料

旧真卡文件的取证逻辑默认仅打印 JSON；`FLAGMEMORY_REALCARD_EVIDENCE` 非空时才按给定路径写入。该旧逻辑不会自动拦截仓内路径，使用者须显式指定仓外位置，不得覆盖已有证据。此说明不是执行旧文件的许可。

探针脚本随包分发，按各目录说明以脚本调用，不作为稳定模块 API。`reports/` 只保留索引；历史画像/卸载资料在协作仓个人汇总线待公共合入。三栈卸载证据与池首次芯片运行证据分别核对。


### Ascend test preparation

The standalone `tests/pool/test_pool_ascend_realcard.py` has a bounded hardware validation instance documented in [the validation record](../docs/validation/2026-11-04-ascend-pool-onerec.md). Its presence does not establish coverage for another configuration. Each execution requires independently verified physical-to-logical mapping, device authorization, runtime APIs, nonblocking event query, a confirmed execution window, resource limits and separate approval. The fixture requires explicit `FLAGMEMORY_ASCEND_ENABLE=approved-mapping-and-bounds`, `FLAGMEMORY_ASCEND_DEVICE`, `FLAGMEMORY_ASCEND_BYTE_LIMIT` (at most 1 GiB cumulative across all tests in the process), and `FLAGMEMORY_ASCEND_QUERY_NONBLOCKING=verified`. Select that file explicitly with `-o addopts= -m realcard`; never infer physical identity from the logical number. Use an external watchdog and bounded output. A skipped pending observation means uncovered, not passed. The quota excludes runtime internal allocations and caching. Cleanup errors are failures.

The obsolete CUDA suite is disabled unless both its exact file is selected and `FLAGMEMORY_ENABLE_LEGACY_CUDA=I_ACCEPT_LEGACY_RISKS` is set. It contains hazardous and outdated tests and is not an Ascend test. Collection does not import the framework.

A separate health-probe process must receive a separately reserved sub-budget; the probe and test sub-budgets must sum to no more than the approved batch limit. Release does not refund the cumulative test budget.

设备统一接口的昇腾可选绑定及张量消费主机合同见 `tests/interfaces/test_ascend_profile.py`；替身只验证公开调用形状和拒绝路径，不代表设备实测。

统一设备桥接使用独立入口 `tests/pool/test_pool_device_context_ascend_realcard.py`。它依赖 `device-runtime/manifest.json` 和同目录完整公开 `runtime/` 冻结包；清单 SHA256、公共提交和树身份须与本次 preflight 的 `device_runtime` 相符。组件提交与外部设备包分别登记，完整组合归档用同一 source 摘要绑定。不要把旧直接适配器成绩用于此入口。

有界执行器的 `bridge-lifecycle` 阶段沿用已核映射及健康前置，固定一颗映射目标、2 CPU/4 GiB、64 MiB 累计底座申请、300 秒总时限和只读源挂载。该阶段只选择上述文件的三个生命周期，显式启用值为 `approved-bridge-lifecycle`；来源、目标和查询合同先通过门禁，再延迟导入框架。只做公开事件查询与有限等待，不做全设备同步。调用方初始化并稳定持有框架上下文；框架 current_device 不证明独立 ACL 上下文，context_ok 为未校验。

设备实测结果要求精确三个新入口且零失败、零错误、零跳过。待完成事件未观察到会跳过并记为 uncovered；该结果不产生可复用成功证据。不同原生 Python 流对象不证明不同物理流。释放只交还自有引用，累计额度不返还；清理失败或未完成均不能记为通过。显式收集可用 `pytest tests/pool/test_pool_device_context_ascend_realcard.py --collect-only -q -o addopts= -m realcard`，收集不导入厂商框架。


### 统一桥接连续输入

`tests/pool/test_onerec_device_context_ascend.py` 的单个真卡入口使用独立 enable 值
`approved-bridge-sequence` 与 `onerec-bridge-sequence` 资源契约。门控验证固定公共设备包、
已独立接受的三生命周期原始证据，以及分别登记的生命周期源与模型源祖先关系；旧单次输入门不生效。
调用方初始化框架上下文并稳定持有。模型与 tokenizer 各加载一次；19 token 实例种子按 repeat/narrow
构造 `(19,64,128,19)*4`，batch=1、全一 mask、BF16/eager、use_cache=False。
逐组普通/池输入成对前向、embedding 消费校验、有限完成查询、归还后复用；只覆盖输入缓冲生命周期，
不涵盖权重、KV、激活接管、业务质量或性能收益。使用独立审查的单用途宿主看门狗实施总期限、资源观察及精确自有容器清理。
