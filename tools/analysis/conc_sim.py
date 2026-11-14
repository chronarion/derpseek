"""Wavefront refinement cache vs concurrency: share of picks at base precision, with the link budget per step."""
import glob, os, sys, numpy as np
from collections import OrderedDict
from safetensors import safe_open
R = sys.argv[1]; NE, L = 384, 40
W = {}
for f in sorted(glob.glob(f"{R}/*.safetensors")):
    w = os.path.basename(f)[:-12]
    with safe_open(f, "np") as h:
        W[w] = h.get_tensor("experts").astype(np.int64)[:, :2048]
cnt = {w: np.stack([np.bincount(W[w][l].ravel(), minlength=NE) for l in range(L)]) for w in W}
tot = sum(cnt.values())
order = ["chat-0", "code-0", "zh-0", "wiki-0", "math-0", "code-2", "chat-1", "long-0", "zh-1", "code-1",
         "wiki-1", "math-1", "chat-2", "code-3", "long-1", "wiki-2", "chat-3", "wiki-3"]
def resident(mix, n):
    if n == 0: return np.zeros((L, NE), bool)
    c = (tot - sum(cnt[w] for w in mix)).ravel(); res = np.zeros(L * NE, bool)
    res[np.argsort(-c, kind="stable")[:n]] = True
    return res.reshape(L, NE)
STEP_MS = {1: 6.5, 2: 7.5, 4: 9.5, 8: 12.6, 16: 17.0}
ROWS = {1: 6, 2: 6, 4: 6, 8: 4, 16: 4}
LINK_GBS = 400.0
CONFIGS = {   # name: (static lossless slots, refinement cache slots, refinement MB, base rel. error)
    "2+2 static10000": (10000, 1654, 8.85, 0.38),
    "3+1 all-base":    (0,     7949, 4.42, 0.20),
}
def run(mix, static_n, cap, ref_mb, conc):
    res = resident(mix, static_n)
    budget = int(STEP_MS[conc] * LINK_GBS / ref_mb)        # refinements the link delivers per step
    rows = ROWS[conc]; T = 2048
    lru = OrderedDict(); base = full = 0; fetched = []
    for s in range(T // rows):
        left = budget; f = 0
        for l in range(L):
            for w in mix:
                for x in W[w][l, s * rows:(s + 1) * rows].ravel():
                    if res[l, x]: full += 1; continue
                    k = l * NE + int(x)
                    if k in lru: lru.move_to_end(k); full += 1
                    elif left > 0:
                        left -= 1; f += 1; full += 1; lru[k] = 1
                        if len(lru) > cap: lru.popitem(last=False)
                    else: base += 1
        fetched.append(f)
    return base / (base + full), np.mean(fetched), budget
print("share of ALL picks served at base precision (lower is better), mean refinements fetched per step / link budget per step")
for name, (sn, cap, mb, err) in CONFIGS.items():
    print(f"\n{name}: static {sn}, refinement cache {cap} slots of {mb} MB, base error ~{err*100:.0f}%")
    for conc in (1, 2, 4, 8, 16):
        groups = [order[i:i + conc] for i in range(0, len(order) - conc + 1, max(conc, 1))][: max(1, 18 // conc)]
        r = [run(g, sn, cap, mb, conc) for g in groups]
        b = np.mean([x[0] for x in r]); fe = np.mean([x[1] for x in r])
        print(f"  C{conc:<2d}: base-precision picks {b*100:5.2f}%  (distortion proxy {b*err*err*1e3:.2f}e-3)  "
              f"fetched/step {fe:6.0f} of budget {r[0][2]}  ({len(groups)} mixes)")
