#!/usr/bin/env python3
"""K>G recall proof: fill the small GPU pool with distinct prompts so the
earliest one is evicted to the host pool, then re-ask it and compare the
output against the pre-eviction canonical answer.

Phase A  ask prompt #0 once, save output (canonical).
Phase B  ask fill_n distinct prompts (unique hashes, ~512 tok each) to force
         eviction of prompt #0's blocks from the GPU pool to host.
Phase C  re-ask prompt #0: expect host-pool recall (load DMA) and an output
         identical to Phase A.
Evidence companions: /tmp/w2x_marker_transfers.txt (store/load DMA batches).
"""
import argparse, json, time, urllib.request

p = argparse.ArgumentParser()
p.add_argument('--port', type=int, default=18307)
p.add_argument('--model', default='tiny')
p.add_argument('--prompt-tokens', type=int, default=250)
p.add_argument('--max-tokens', type=int, default=32)
p.add_argument('--fill-n', type=int, default=150)
a = p.parse_args()
base = f'http://127.0.0.1:{a.port}'


def make_prompt(i: int) -> str:
    # Unique 2-digit number sequence per request (first element differs by i,
    # so block hashes never collide across requests). ~1 token per number on
    # Qwen-style BPE; 250 numbers + 32 gen fits the 512 budget with headroom
    # even if the tokenizer spends up to 1.5 tokens per number.
    return ' '.join(str((i * 37 + j * 7) % 100) for j in range(a.prompt_tokens))


def ask(i: int, timeout: int = 600):
    body = json.dumps({'model': a.model, 'prompt': make_prompt(i),
                       'max_tokens': a.max_tokens, 'ignore_eos': True,
                       'temperature': 0}).encode()
    req = urllib.request.Request(base + '/v1/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'HTTP {e.code}: {e.read().decode()[:300]}') from None
    return r, time.time() - t0


r0, dt = ask(0)
out0 = r0['choices'][0]['text']
print(f'phaseA probe0: {dt:.2f}s ct={r0["usage"]["completion_tokens"]}', flush=True)

t0 = time.time()
for i in range(1, a.fill_n + 1):
    _, dt = ask(i)
    if i % 20 == 0:
        print(f'fill {i}/{a.fill_n} elapsed={time.time()-t0:.0f}s', flush=True)
print(f'fill done in {time.time()-t0:.1f}s', flush=True)

r1, dt = ask(0)
out1 = r1['choices'][0]['text']
print(f'phaseC probe0: {dt:.2f}s', flush=True)
print(json.dumps({'recall_output_equal': out0 == out1,
                  'out0_head': out0[:60], 'out1_head': out1[:60]}), flush=True)
