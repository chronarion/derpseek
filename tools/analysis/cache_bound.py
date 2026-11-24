"""Cache-policy bound (Fable 11-23): replay served decode steps' expert picks (DERPSEEK_PICK_DUMP, [layers, 384] uint16 per
step) through each layer's non-static experts and C cache rows; per step a picked expert not static and not cached is
one read from CPU memory. Policies: LRU (insert every missed expert after the step, evict least recently used) and
Belady (offline optimum: evict the cached expert needed furthest in the future; insertion as LRU). Cache sizes swept,
so the curve also prices extra rows. Distinct reads per step summed over layers.
   python3 cache_bound.py DUMP ROWMAP.json [--sizes 96,108,120,144] [--layers 20-39]"""
import argparse, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dump"); ap.add_argument("rowmap")
ap.add_argument("--sizes", default="64,96,108,120,144"); ap.add_argument("--layers", default="0-39")
ap.add_argument("--budget", type=int, default=0, help="promotions per step over all simulated layers (0 = unlimited); a "
                "promotion is a copy over the same link, so reads + copies is the link traffic")
a = ap.parse_args()
L, E = 40, 384
raw = np.fromfile(a.dump, dtype=np.uint16)
T = raw.size // (L * E)
picks = raw[:T * L * E].reshape(T, L, E) > 0
rm = json.load(open(a.rowmap))["layers"]
lo, hi = (int(x) for x in a.layers.split("-"))
sizes = [int(x) for x in a.sizes.split(",")]
print(f"{T} decode steps, layers {lo}-{hi}; mean distinct experts per layer-step {picks[:, lo:hi + 1].sum(-1).mean():.1f}")


def lru(seq, C):
    from collections import OrderedDict
    cache, miss = OrderedDict(), 0
    for need in seq:
        for e in need:
            if e in cache:
                cache.move_to_end(e)
        m = [e for e in need if e not in cache]
        miss += len(m)
        for e in m:
            cache[e] = True
            cache.move_to_end(e)
            if len(cache) > C:
                cache.popitem(last=False)
    return miss


def belady(seq, C):
    # next-use index per (step, expert)
    nxt = {}
    future = [None] * len(seq)
    for t in range(len(seq) - 1, -1, -1):
        future[t] = {e: nxt.get(e, 10**9) for e in seq[t]}
        for e in seq[t]:
            nxt[e] = t
    cache, miss = {}, 0
    for t, need in enumerate(seq):
        for e in need:
            if e in cache:
                cache[e] = future[t][e]
        m = [e for e in need if e not in cache]
        miss += len(m)
        for e in m:
            cache[e] = future[t][e]
        while len(cache) > C:
            far = max(cache, key=lambda x: cache[x])
            del cache[far]
    return miss


def budgeted(seqs, C, B, belady_mode):
    """all layers together: each step, misses are read in place; then at most B of them are inserted (copies). LRU:
    the most recently missed first (those repeated in this step), evicting least recently used. Belady: the missed
    experts with the nearest next use first, each only if it is needed before the cache's farthest-needed resident."""
    Lx = len(seqs); T_ = len(seqs[0])
    nxt_all = []
    for seq in seqs:
        nxt, future = {}, [None] * T_
        for t in range(T_ - 1, -1, -1):
            future[t] = {e: nxt.get(e, 10**9) for e in seq[t]}
            for e in seq[t]:
                nxt[e] = t
        nxt_all.append(future)
    caches = [dict() for _ in range(Lx)]      # expert -> last use (LRU) or next use (Belady)
    reads = copies = 0
    for t in range(T_):
        cand = []
        for li in range(Lx):
            cache, need, fut = caches[li], seqs[li][t], nxt_all[li][t]
            for e in need:
                if e in cache:
                    cache[e] = fut[e] if belady_mode else t
            m = [e for e in need if e not in cache]
            reads += len(m)
            for e in m:
                cand.append((fut[e] if belady_mode else -t, li, e))
        cand.sort()
        used = 0
        for key, li, e in cand:
            if B and used >= B:
                break
            cache = caches[li]
            if len(cache) < C:
                cache[e] = key if belady_mode else t; used += 1; continue
            if belady_mode:
                far = max(cache, key=lambda x: cache[x])
                if cache[far] <= key:
                    continue                    # every resident is needed sooner: bypass
                del cache[far]; cache[e] = key
            else:
                old = min(cache, key=lambda x: cache[x])
                del cache[old]; cache[e] = t
            used += 1
        copies += used
    return reads / T_, copies / T_


seqs = []
nonstatic_picks = 0
for l in range(lo, hi + 1):
    hot = set(rm[str(l)]["hot"])
    ns = np.array([e not in hot for e in range(E)])
    seq = [list(np.nonzero(picks[t, l] & ns)[0]) for t in range(T)]
    nonstatic_picks += sum(len(s) for s in seq)
    seqs.append(seq)
print(f"distinct non-static experts per step (sum over layers): {nonstatic_picks / T:.1f}; promotion budget {a.budget or 'unlimited'} per step")
for C in sizes:
    lr, lc = budgeted(seqs, C, a.budget, False)
    br, bc = budgeted(seqs, C, a.budget, True)
    print(f"cache {C:3d} rows: LRU reads {lr:6.1f} + copies {lc:5.1f} = {lr + lc:6.1f}/step; Belady reads {br:6.1f} + copies "
          f"{bc:5.1f} = {br + bc:6.1f}/step; optimum saves {100 * (1 - (br + bc) / max(lr + lc, 1e-9)):.1f}% of link traffic")
