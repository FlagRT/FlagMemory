"""三套 KV→Host 卸载栈（框架绑定的可选适配层）。

各栈经 vLLM 的 kv_connector_module_path 外挂加载，子包与导出类见本目录 README.md。
这里刻意不做 eager import：三套栈各绑一个宿主栈版本，同一环境里只有一套能导入成功，
顶层 import 任何一套都会把另两套的 ImportError 带出来。
"""
