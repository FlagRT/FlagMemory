# KV->Host 卸载 connector（PPU ZW810E / vllm-plugin-fl 0.24 栈）。
# 栈外模块，经 kv_connector_module_path=offload.ppu_offload 加载。
# 源：vLLM 0.24 vendor SimpleCPUOffloadConnector 六文件 vendor+补丁
# （P-0 元组容忍 / P-1 流优先级兜底 / P-2 逐块 copy_ 替代批量 cuMemcpy / P-3 pin 兜底）。
# 工厂按 getattr(module, kv_connector名字) 取类，本包必须导出 PpuSimpleCPUOffloadConnector。
from .connector import PpuSimpleCPUOffloadConnector

__all__ = ["PpuSimpleCPUOffloadConnector"]
