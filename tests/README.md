# 单测说明

> 依据：ADR-0014 §3.5 第 2 件改造（补单测与门禁）——「测试目录 + 测试配置 + 正常 / 边界 / 异常
> 三类用例 + 持续集成；**池本体的测试不得依赖真卡**，真卡部分单独标记」。

## 一句话：默认跑的那批不需要卡

```bash
pip install -e ".[dev]"
python tools/check_layering.py     # 分层门禁（纯标准库，不用 pytest）
pytest                             # 默认只跑不依赖真卡的用例
```

`pyproject.toml` 里 `addopts = "-m 'not realcard'"`，所以直接 `pytest` 会把真卡用例
**deselect 掉**（输出里会写 deselected 几条，那是正常的，不是跳过掩盖问题）。

## 真卡用例怎么跑

标了 `@pytest.mark.realcard` 的用例必须在**带卡机器的宿主栈容器内**跑：

```bash
pytest -m realcard          # 只跑真卡那批
pytest -m ""                # 两批都跑（宿主栈齐全时才有意义）
```

## 为什么要这么分（不是洁癖，是三条硬事实）

1. **无卡成员跑不了真卡用例**。组里有人在没有国产卡的机器上改池本体，如果单测默认需要卡，
   他们等于没有单测可跑 —— ADR-0014 §3.5 第 2 条点名要防的就是这个。
2. **本包刻意零依赖**。`torch` / vLLM 由各芯片容器自带，四家的版本互不兼容，CI 里装不出一套
   「正确」的。所以 CI 只验池本体与能力层（它们本来就不许碰框架），框架绑定那层的真实可用性
   只能在带卡机器的栈内复验。
3. **三套卸载栈同一环境只有一套能导入成功**。`flagmemory.tiering.offload.{npu,ppu,xpu}` 各绑
   一个宿主栈版本，拿 CI 去 import 它们必然两套失败，失败的原因还和代码对不对无关。

## 目录

| 路径 | 测什么 | 要卡吗 |
|---|---|---|
| `tests/caps/test_capabilities.py` | 能力枚举、四级分法、池的用法策略、三条不变量文本在位 | 否 |
| `tests/caps/test_degradation.py` | 五条降级路径查表、默认实现路径、按声明算降级计划 | 否 |
| `tests/caps/test_backend.py` | `DeviceBackend` 协议面与假后端：三键、流、事件、句柄记账 | 否 |
| `tests/caps/test_query.py` | 能力查询入口 `has` / `require` / `report` 与 L0 齐备判读 | 否 |
| `tests/tiering/test_alias_package.py` | 兼容别名包 `offload.*` 的懒加载与 `DeprecationWarning` | 否 |
| `tests/tiering/test_offload_realcard.py` | 三套卸载栈在宿主栈内能真导入、connector 类能取到 | **是**（`realcard`） |
| `tests/tools/test_check_layering.py` | 分层门禁脚本：干净树通过、违规样本非零退出 | 否 |

每个文件里的用例按 **正常 / 边界 / 异常** 三类分节，节名写在注释里，类名也带上
（`TestNormal` / `TestBoundary` / `TestError`），便于按判据点数。

## 不在这里的东西

- **实测证据**（真模型跑出来的 json / 日志）不是单测，归 ADR-0014 §3.5 第 3 件改造处置，
  本次不动 `reports/`。
- **池的业务实现**（分级块池 + 流有序分配）是 11-28 的出口，本次只有能力层的协议与假后端，
  所以这里没有池的功能用例 —— 池实现落地时同批补。
