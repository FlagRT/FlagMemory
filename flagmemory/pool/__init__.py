"""池本体：显存池管理器的核心路径。

硬约束（依据 ADR-0014 §3.5 第 4 条与四芯片显存能力调研第五节）：
- 不得 import 推理框架（vllm / torch_npu / 各厂商 plugin）；框架绑定下沉到
  flagmemory.tiering.offload 的可选适配层；
- 核心路径只许用四家齐备的 L0 能力，L1 能力一律走 flagmemory.caps 的能力分支；
- 池持有流必须强持有，禁止用弱引用做自动清理。
"""
