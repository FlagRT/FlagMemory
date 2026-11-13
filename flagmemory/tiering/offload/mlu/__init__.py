# KV→Host 卸载 connector（寒武纪 MLU590 / flagos-app vllm0.20.2 推理腿栈）。
# 栈外模块，经 kv_connector_module_path=flagmemory.tiering.offload.mlu 加载
# （旧路径 offload.mlu_offload 作为兼容别名保留一个版本）。
# 目标栈：vllm 0.20.2+flagos / vllm-plugin-FL 0.2.2 / neuware4.4.3（torch_mlu 1.29.2）@ 寒武纪 MLU590。
# 工厂按 getattr(module, kv_connector名字) 取类，本包必须导出 MluSimpleCPUOffloadConnector。
from .connector import MluSimpleCPUOffloadConnector

__all__ = ["MluSimpleCPUOffloadConnector"]
