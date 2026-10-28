# KV→Host 卸载 connector（昆仑芯 P800 / vllm-plugin-fl 栈，原 w2_xpu_offload 迁入）。
# 栈外模块，经 kv_connector_module_path=offload.xpu_offload 加载。
# 目标栈：vLLM 0.20.2 + vllm-plugin-fl 0.2.0 (vendor.kunlunxin) @ 昆仑芯 P800。
# 工厂按 getattr(module, kv_connector名字) 取类，本包必须导出 XpuSimpleCPUOffloadConnector。
from .connector import XpuSimpleCPUOffloadConnector

__all__ = ["XpuSimpleCPUOffloadConnector"]
