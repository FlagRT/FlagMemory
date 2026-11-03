# KV→Host 卸载 connector（Ascend 910C / vllm-ascend 栈，原 w2_npu_offload 迁入）。
# 栈外模块，经 kv_connector_module_path=flagmemory.tiering.offload.npu 加载
# （旧路径 offload.npu_offload 作为兼容别名保留一个版本）。
# 目标栈：vLLM 0.20.2 + vllm-ascend v0.20.2rc1 @ Ascend 910C。
# 工厂按 getattr(module, kv_connector名字) 取类，本包必须导出 NpuSimpleCPUOffloadConnector。
from .connector import NpuSimpleCPUOffloadConnector

__all__ = ["NpuSimpleCPUOffloadConnector"]
