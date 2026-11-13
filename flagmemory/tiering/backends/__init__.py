# 四套 KV→Host 卸载适配层到 TierBackend 协议面的接线层包。
# 方向（T-5）：本包 import offload 实现（外部把适配层包成 TierBackend 注册给
# manager）；manager 本体不 import 本包，也不 import offload。
# 现状：npu / xpu / ppu 三栈已接（bind 默认懒加载各自 offload connector 侧拷贝
# 后端）；mlu 栈无 connector（offload/ 下无 mlu 子包），bind 默认走 mlu_backend
# 内嵌的 torch.mlu 最小拷贝通路（见该模块头），connector 落地后切换、协议面不变。
