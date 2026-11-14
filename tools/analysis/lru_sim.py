"""LRU cache of non-resident (cold) experts in HBM: hit rate vs capacity, one request and 16 interleaved requests."""
import glob, os, sys, numpy as np
from collections import OrderedDict
from safetensors import safe_open
R = sys.argv[1]; NE, L = 384, 40
W = {}
for f in sorted(glob.glob(f"{R}/*.safetensors")):
    w = os.path.basename(f)[:-12]
    with safe_open(f, "np") as h:
        W[w] = h.get_tensor("experts").astype(np.int64)[:, :2048]      # same length for all windows
names = list(W)
cnt = {w: np.stack([np.bincount(W[w][l].ravel(), minlength=NE) for l in range(L)]) for w in names}
tot = sum(cnt.values())
def resident(excl, n):
    c = (tot - sum(cnt[w] for w in excl)).ravel()
    res = np.zeros(L * NE, bool); res[np.argsort(-c, kind="stable")[:n]] = True
    return res.reshape(L, NE)
def stream(ws):     # round-robin positions of the windows (one row per request per step)
    T = 2048
    for t in range(T):
        for w in ws:
            yield w, t
def sim(ws, res, caps):
    out = {}
    for C in caps:
        lru = OrderedDict(); hit = miss = 0; new_per_step = []
        step_new = 0
        for i, (w, t) in enumerate(stream(ws)):
            e = W[w][:, t]                       # [L, 6]
            for l in range(L):
                for x in e[l]:
                    if res[l, x]: continue
                    k = l * NE + int(x)
                    if k in lru:
                        hit += 1; lru.move_to_end(k)
                    else:
                        miss += 1; lru[k] = 1; step_new += 1
                        if len(lru) > C: lru.popitem(last=False)
            if (i + 1) % (len(ws) * 6) == 0:     # a C1-like step = 6 rows per request
                new_per_step.append(step_new); step_new = 0
        out[C] = (hit / max(hit + miss, 1) * 100, np.mean(new_per_step))
    return out
NRES = 10000
caps = (250, 500, 1000, 2000, 4000)
print(f"static resident: hottest {NRES} slots (ranked without the test windows); LRU over the rest (slots = experts at FP4, 18.8 MB)")
print("single request: hit% of cold picks / new (fetched) experts per 6-row step, by LRU capacity", caps)
agg = {C: [] for C in caps}
for w in names:
    r = sim([w], resident([w], NRES), caps)
    print(f"  {w:8s} " + "  ".join(f"{r[C][0]:5.1f}/{r[C][1]:5.1f}" for C in caps))
    for C in caps: agg[C].append(r[C])
print("  mean     " + "  ".join(f"{np.mean([a[0] for a in agg[C]]):5.1f}/{np.mean([a[1] for a in agg[C]]):5.1f}" for C in caps))
mix = ["chat-0", "chat-1", "code-0", "code-1", "code-2", "math-0", "math-1", "wiki-0", "wiki-1", "wiki-2", "zh-0", "zh-1", "long-0", "long-1", "chat-2", "code-3"]
caps16 = (1000, 2000, 4000, 5360)
r = sim(mix, resident(mix, NRES), caps16)
print("16 interleaved requests (one row each per step; resident ranked without them):", "  ".join(f"cap {C}: {r[C][0]:.1f}% hit, {r[C][1]:.0f} new/step(6 rows each)" for C in caps16))
