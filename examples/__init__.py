"""examples 目录显式为正规包（带 __init__.py）。

机制登记：部分厂商栈镜像的 site-packages 内装有同名 `examples` 正规包（2026-11-14
PPU vllm0.24 镜像实测，来源为某 pip 包安装产物），PEP 420 语义下正规包优先于命名
空间包、且与 sys.path 顺序无关，会整体遮蔽仓根的 `examples/`，使
`tests/interfaces/test_ascend_profile.py` 的 `import examples.device_context_tensor_pool`
解析到镜像内的包而抛 ModuleNotFoundError（该镜像内 10 败，干净环境全过——环境性，
非用例缺陷）。本目录转正规包后，位于 sys.path 前部的仓根条目在解析时优先于
site-packages 同名包，测试在干净环境与栈容器内行为一致。
"""
