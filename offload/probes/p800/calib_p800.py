#!/usr/bin/env python3
"""Calibrate prompt token counts against the live server's /tokenize."""
import json, sys, urllib.request

base = 'http://127.0.0.1:18307'


def tok(prompt):
    body = json.dumps({'model': 'tiny', 'prompt': prompt}).encode()
    req = urllib.request.Request(base + '/tokenize', data=body,
                                 headers={'Content-Type': 'application/json'})
    r = json.loads(urllib.request.urlopen(req, timeout=60).read())
    return r['count']


cases = [
    ('nums250', ' '.join(str((j * 7) % 100) for j in range(250))),
    ('nums150', ' '.join(str((j * 7) % 100) for j in range(150))),
    ('nums120', ' '.join(str((j * 7) % 100) for j in range(120))),
    ('words95', ' '.join('w%d' % ((j * 7919) % 100000) for j in range(95))),
    ('tiny', 'hello world'),
]
for name, p in cases:
    try:
        print(name, tok(p))
    except Exception as e:
        print(name, 'ERR', repr(e)[:200])
