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

首次昇腾验证需后续独立测试文件和显式路径，先审阅实际收集名单、资源上限、目标流就绪、池块公开访问及清理步骤；当前文档不宣称该适配器已实现或真卡已通过。仅复制旧文件的运行命令无法满足这些条件。主机假后端结果不证明厂商运行时和设备生命周期。

## 取证与历史材料

旧真卡文件的取证逻辑默认仅打印 JSON；`FLAGMEMORY_REALCARD_EVIDENCE` 非空时才按给定路径写入。该旧逻辑不会自动拦截仓内路径，使用者须显式指定仓外位置，不得覆盖已有证据。此说明不是执行旧文件的许可。

探针脚本随包分发，按各目录说明以脚本调用，不作为稳定模块 API。`reports/` 仍含历史数据，证据迁移未收口；本次文档工作不迁、不删、不改历史材料。三栈卸载证据与池首次芯片运行证据分别核对。
