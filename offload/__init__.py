"""兼容别名：旧 import 路径 offload.*，在 0.2.x 有效，0.3.0 移除。

新路径是 flagmemory.tiering.offload.{mlu,npu,ppu,xpu}。别名存在的唯一理由是容器里
已经写死的 vLLM 参数 kv_connector_module_path=offload.<栈>_offload —— 那串字符串
继续能加载到同一个 connector 类，不必跟着改启动命令。

注意：别名包本身不 import 任何栈（四套栈各绑一个宿主栈版本，同一环境里只有一套能
导入成功），取类时才真正加载，见各别名子包的 __getattr__。
"""

__all__ = ["mlu_offload", "npu_offload", "ppu_offload", "xpu_offload"]
