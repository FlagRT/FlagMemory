# STATUS

更新：2026-10-26｜陈元新 / halloyuanxin

| 线 | 状态 | 依据 |
|---|---|---|
| offload/npu_offload（910C · vllm-ascend） | W2 闭环 + W3 开关对照已实测 | dev-1.0 两报告（merge 5f880a1）：吞吐 +20.4% / decode 代价 7-8%；64G 重测：Host 池 > GPU 池（阈值 ≈39 GiB 起档）才成真 L2 |
| offload/xpu_offload（P800 · kunlunxin） | W2 四步闭环全通（卸/存/召回/正确）+ 真模型 A/B 已收口 | 2026-10-23：267 store + 1 load 召回；2026-10-24 A/B：expand 106.4% 池下开卸载快 38%（暖跑配对口径） |
| offload/ppu_offload（PPU ZW810E · plugin-fl 0.3.0/vllm 0.24.0） | 冒烟闭环 + A/B 开关对照已收口 | 2026-10-26：store 190 块 + load 7 块 DMA 实锤；A/B（tiny/64 块池）：池内零代价（decode 10.34→10.37 tok/s），超池 200% 净代价 +2.1%；真模型未测 |
| guard/（OOM 防护） | 未进场 | 11 月：真机验证 |
| profile/（画像工具链） | 未收拢 | 画像报告已入 dev-1.0，工具待归位 |
| pip 功能包 | 初版待做 | 依赖本仓收录 |

## 环境

env/v1 未建；收录后按 ENV-SPEC 补登各栈锁定镜像的六事实
（image id / remote / commit / activation / dirty / container_path）。
PPU 栈启动三件套与镜像坑见 offload/README.md「PPU 栈已知边界」。
