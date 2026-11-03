import json
import threading
import time
import urllib.request

URL = "http://127.0.0.1:18330/v1/chat/completions"


def fire(prompt, max_tokens=48, timeout=600):
    body = json.dumps({
        "model": "tiny",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout).status


prompt_a = (
    "请从以下八个维度详细介绍张量并行技术，每个维度至少八十字："
    "一、基本原理与切分方式；二、通信模式与代价；三、与流水线并行的对比；"
    "四、与数据并行的组合；五、对显存占用的影响；六、典型实现框架；"
    "七、适用模型规模与场景；八、工程实践中的坑与调优经验。"
)

filler = (
    "请逐条解释以下术语并各写一个例子：张量并行、流水线并行、数据并行、专家并行、"
    "序列并行、上下文并行、零冗余优化器、梯度检查点、激活值卸载、优化器状态卸载、"
    "混合精度训练、动态损失缩放、梯度累积、学习率预热、余弦退火、权重衰减、"
    "KV缓存、页注意、前缀缓存、请求抢占、块表、写时复制、交换、重计算、量化感知训练。"
) * 6

print("A:", fire(prompt_a, max_tokens=48), flush=True)
time.sleep(2)


def burst(i):
    p = f"第{i}组{i}{i}{i}。{filler}"
    try:
        print("B%d:" % i, fire(p, max_tokens=48), flush=True)
    except Exception as e:
        print("B%d: FAIL" % i, repr(e)[:100], flush=True)


ts = [threading.Thread(target=burst, args=(i,)) for i in range(3)]
for t in ts:
    t.start()
for t in ts:
    t.join()
time.sleep(3)

print("replay:", fire(prompt_a, max_tokens=32), flush=True)
