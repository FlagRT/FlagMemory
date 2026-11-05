"""兼容别名 offload.xpu_offload → flagmemory.tiering.offload.xpu（0.2.x 有效，0.3.0 移除）。

vLLM 的 connector 工厂按 getattr(module, kv_connector名字) 取类，所以本模块只需在
取属性时把类递出去；import 本模块本身不触发栈依赖。
"""

import warnings

_NEW = "flagmemory.tiering.offload.xpu"
_EXPORTS = ("XpuSimpleCPUOffloadConnector",)

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    warnings.warn(
        "offload.xpu_offload 是兼容别名，0.3.0 起移除；"
        f"请把 kv_connector_module_path 与 import 改为 {_NEW}。",
        DeprecationWarning,
        stacklevel=2,
    )
    from importlib import import_module

    return getattr(import_module(_NEW), name)


def __dir__():
    return sorted(__all__)
