# 显存方向 KV 卸载交付件。各栈子包经 vLLM kv_connector_module_path 外挂加载，
# 使用时把本仓根目录加入 PYTHONPATH，模块路径写 offload.<子包名>；
# 仓根 `pip install -e .`（包名 flagmemory）后 PYTHONPATH 不再必需。
__version__ = "0.1.0"
