# 三套 KV→Host 卸载适配层到 TierBackend 协议面的接线层包。
# 方向（T-5）：本包 import offload 实现（外部把适配层包成 TierBackend 注册给
# manager）；manager 本体不 import 本包，也不 import offload。
# 现状：昇腾栈（npu）已接；xpu / ppu 待改造（offload/README.md 登记）。
