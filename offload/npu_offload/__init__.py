# KV→Host 卸载 connector（Ascend 910C / vllm-ascend 栈，原 w2_npu_offload 迁入）。
# 栈外模块，经 kv_connector_module_path=offload.npu_offload 加载。
# 目标栈：vLLM 0.20.2 + vllm-ascend v0.20.2rc1 @ Ascend 910C。
# 工厂按 getattr(module, kv_connector名字) 取类，本包必须导出 NpuSimpleCPUOffloadConnector。
from .connector import NpuSimpleCPUOffloadConnector

__all__ = ["NpuSimpleCPUOffloadConnector"]
