"""FlagMemory —— 显存池管理器组件（招标功能拆分表「显存池管理器」18 条交付条目的代码落点）。

包分层（顶层五块）：

    flagmemory.pool        池本体：块分级、流有序分配，分配/复用/回收/碎片四类操作。
                           不得 import 任何推理框架，只向下依赖 flagmemory.caps。
    flagmemory.caps        芯片能力抽象层：能力查询与降级分支，对接设备方向统一后端接口。
    flagmemory.tiering     分层存储：三级放置与换入换出。三套 KV 卸载栈在
                           flagmemory.tiering.offload 下，属框架绑定的可选适配层。
    flagmemory.profiling   性能分析工具与探针。
    flagmemory.interfaces  对外接口声明（接口头文件目录）。

依赖方向（门禁脚本 tools/check_layering.py 与单测各钉一次）：

    interfaces ← pool ← caps          池本体与能力层都不碰框架
    tiering.offload → pool / caps     适配层可以向下用，反向不行

import 路径是 flagmemory.*；旧的 offload.* 作为兼容别名保留一个版本（0.2.x 有效，
0.3.0 移除），见 offload/__init__.py。
"""

__version__ = "0.3.0"

# 顶层刻意不做任何子包的 eager import：tiering.offload 需要宿主栈的 torch / vLLM，
# 而池本体必须在无框架、无真卡的环境里可导入（单测与分层门禁都依赖这一点）。
