# reports/ —— 实测证据索引（证据本体不在本仓）

本仓是**组件代码仓**：显存池本体、芯片能力抽象层、分层存储（三套 KV→Host 卸载栈）、
性能分析工具与接口头文件目录。**实测证据不在本仓**，在公共协作仓
`git@github.com:FlagRT/runtime-team.git` 的 `dev/memory/` 下，分两处存放：

| 存放处 | 放什么 |
|---|---|
| `dev/memory/docs/goals/<批次目录>/` | 结果报告：PLAN（预注册）/ RESULT（技术版）/ 通俗版 |
| `dev/memory/evidence/flagmemory-offload/<批次目录>/` | 原始数据：逐腿 JSON、服务端与压测链日志、DMA marker 账本、腿本脚本与采样探针 |

两处各有 README 说明目录与命名约定，入口：`dev/memory/STATUS.md` 的「实测证据目录」一条。

依据 ADR-0014 §3.1 / §3.5 第 3 条（证据与组件代码分仓）。

---

## 一、批次索引

| 批次 | 测期 | 芯片 / 栈 | 验证了什么 | 结果报告（公共仓 `dev/memory/`） | 原始数据（公共仓 `dev/memory/`） |
|---|---|---|---|---|---|
| P800 诊断与四步闭环 | 2026-10-23 | P800（昆仑芯）· vLLM 0.20.2 + vllm-plugin-fl 0.2.0 + xpu 卸载 connector | W1 诊断，当日递进至 W2 四步闭环（卸→存→召回→结果正确）全通，含三模型链 | `docs/goals/proto-p800-202610/RESULT_诊断与四步闭环_P800昆仑_20261023.md` | 无原始数据条目（本批交付物只有结果报告） |
| P800 真模型 A/B | 2026-10-24 | 同上 | 真模型 OneRec-8B 卸载开关 A/B 收口（暖跑配对口径） | `docs/goals/proto-p800-202610/RESULT_AB卸载开关_P800昆仑_20261024.md` | 无原始数据条目 |
| 画像规范首次双芯执行 | 2026-10-25 | P800（昆仑芯）+ 910C（昇腾）双腿 | 画像规范 v0.3 首次双芯完整执行（真模型全腿本，「换芯不换脑」）；执行中抓出规范 4 处漏洞 | `docs/goals/proto-p800-202610/`：`PLAN_protocol_run_20261025.md` + `RESULT_规范验证跑_{P800,910C,通俗版}_20261025.md` | `evidence/flagmemory-offload/p800_results_1025/`（`proto/` = P800 腿，`proto-910c/` = 910C 对照腿） |
| PPU 卸载移植闭环 | 2026-10-26 | PPU（ZW810E）单卡 TP=1 · vLLM 0.24.0 + vllm-plugin-fl 0.3.0rc0 + ppu 卸载四补丁 | ppu 栈移植闭环 + 开关冒烟对照（冒烟口径，非规范验证口径） | `docs/goals/proto-ppu-202610/`：`PLAN_AB_PPU_20261026.md` + `RESULT_AB卸载开关_PPU_20261026.md` | 无原始数据条目 |
| 真模型增量验证收口 | 2026-10-27 / 28 | P800（昆仑芯）· 真模型 OneRec-8B | 真模型全流程增量验证：64 GiB 锁定池档 + 自然池档双锚点；新仓重挂载零漂移、load 侧补腿首次测成、L/R 按预注册判据记无效 | `docs/goals/proto-p800-202610/`：`PLAN_规范验证跑_真模型_20261027.md` + `RESULT_规范验证跑_真模型{,_通俗版}_20261027.md` | `evidence/flagmemory-offload/p800_results_1027/`（`proto_lock/` 锁定档含预注册锚与作废趟、`proto_nat/` 自然档、`kit/` 腿本与探针） |

## 二、覆盖面一句话

三款国产芯片（昆仑芯 P800 / 昇腾 910C / PPU ZW810E）上的 KV→Host 卸载栈，各自跑到
「卸→存→召回→结果正确」四步闭环；P800 与 910C 另有真模型（OneRec-8B）开关 A/B 与
画像规范级全腿本执行，P800 再加 64 GiB 锁定池与自然池双档的增量验证收口。

## 三、怎么引用

- **引用证据**：写公共协作仓的路径 + **该仓的提交哈希**。证据是追加式的，只认提交，不认分支名。
  例：`runtime-team@<commit>:dev/memory/evidence/flagmemory-offload/p800_results_1027/proto_lock/ON_pass2/chain.log`。
- **报告正文里的相对路径**：各份 RESULT 正文引用原始数据时写的是**批次目录内的相对路径**
  （如 `proto_lock/ON_pass2/chain.log`）。还原方式见
  `dev/memory/docs/goals/proto-p800-202610/README.md` 的「原始数据」一节换算表。
- **只想拿数据不想 clone 整仓**：公共协作仓在本机做全量检出会撞 Windows 文件名长度上限，
  用稀疏检出只取 `dev/memory`：

  ```
  git clone --no-checkout --filter=blob:none -b dev-1.0 git@github.com:FlagRT/runtime-team.git rt
  cd rt && git config core.longpaths true
  git sparse-checkout init --cone && git sparse-checkout set dev/memory
  git checkout dev-1.0
  ```

## 四、本仓纪律：实测数据不回流

仓根 `.gitignore` 已拦下实测数据的常见形态（`reports/**/*.json`、`*.log`、`*.txt` 等，
见该文件「实测证据不入本仓」一段）。本目录**只许有本索引**；
新证据一律进公共协作仓的上述两处，本索引加一行批次。
