import json
import threading
import urllib.request

URL = "http://127.0.0.1:18330/v1/chat/completions"


def fire(i, prompt):
    body = json.dumps({
        "model": "tiny",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=300)
        print(i, r.status, flush=True)
    except Exception as e:
        print(i, "FAIL", repr(e)[:120], flush=True)


long_prompt = (
    "请详细解释以下技术概念，每个概念至少写八十个字：张量并行、流水线并行、数据并行、"
    "专家并行、序列并行、上下文并行、零冗余优化器、梯度检查点、激活值卸载、优化器状态卸载、"
    "混合精度训练、动态损失缩放、梯度累积、学习率预热、余弦退火、权重衰减。"
) * 3

threads = [threading.Thread(target=fire, args=(i, f"第{i}批次。{long_prompt}")) for i in range(8)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print("burst done", flush=True)

# replay the first prompt from the earlier session to trigger a recall-load
body = json.dumps({
    "model": "tiny",
    "messages": [{"role": "user", "content": "天空为什么是蓝色"}],
    "max_tokens": 48,
}).encode()
req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
r = urllib.request.urlopen(req, timeout=300)
print("replay", r.status, flush=True)
