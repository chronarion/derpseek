"""Cold-expert predictability from routing traces (leave-one-window-out). CPU only."""
import glob, os, sys, numpy as np
from safetensors import safe_open
R = sys.argv[1]
NE, L, V = 384, 40, 129280
W = {}
for f in sorted(glob.glob(f"{R}/*.safetensors")):
    w = os.path.basename(f)[:-12]
    with safe_open(f, "np") as h:
        e = h.get_tensor("experts").astype(np.int64)          # [L, T, 6]
        tid = h.get_tensor("target_ids"); pos = h.get_tensor("positions")
    T = e.shape[1]
    tok = np.full(T, -1, np.int64)                             # input token at position t = target of t-1
    ok = pos + 1 < T
    tok[pos[ok] + 1] = tid[ok]
    W[w] = (e, tok)
names = list(W)
# global pick counts per (layer, expert) slot, per window
cnt = {w: np.stack([np.bincount(W[w][0][l].ravel(), minlength=NE) for l in range(L)]) for w in names}
tot = sum(cnt.values())
def resident(test, n):
    c = (tot - cnt[test]).ravel()
    order = np.argsort(-c, kind="stable")
    res = np.zeros(L * NE, bool); res[order[:n]] = True
    return res.reshape(L, NE)
# token->expert table keys over all windows (layer, token, expert)
def keys_of(w):
    e, tok = W[w]; T = e.shape[1]
    lt = (np.arange(L)[:, None] * V + tok[None, :])            # [L, T]
    k = (lt[:, :, None] * NE + e)                              # [L, T, 6]
    valid = np.broadcast_to((tok >= 0)[None, :, None], k.shape)
    return k, valid
allk = np.concatenate([keys_of(w)[0][keys_of(w)[1]] for w in names])
uk_all, uc_all = np.unique(allk, return_counts=True)
def train_count(test, k):
    kt, vt = keys_of(test)
    ut, ct = np.unique(kt[vt], return_counts=True)
    i = np.searchsorted(uk_all, k); i = np.clip(i, 0, len(uk_all) - 1)
    c = np.where(uk_all[i] == k, uc_all[i], 0)
    j = np.searchsorted(ut, k); j = np.clip(j, 0, len(ut) - 1)
    c = c - np.where(ut[j] == k, ct[j], 0)                     # leave the test window out
    return c
for NRES in (10240, 11737):
    print(f"\n=== resident set: hottest {NRES} slots ({NRES/(L*NE)*100:.0f}%), ranked on the other windows ===")
    print("window   cold%  C1-step(R=6) cold experts mean/p95 (GB at FP4)  recall: recent32 recent256 recent2048 | uni-table(any) online-uni | union(r256+uni+online)")
    agg = []
    for w in names:
        e, tok = W[w]; T = e.shape[1]
        res = resident(w, NRES)
        lay = np.broadcast_to(np.arange(L)[:, None, None], e.shape)
        cold = ~res[lay, e]                                    # [L, T, 6]
        coldpct = cold.mean() * 100
        # C1 verify blocks of 6 consecutive positions: distinct cold (layer, expert)
        nb = T // 6; d = []
        for b in range(nb):
            blk = e[:, b*6:(b+1)*6]; cb = cold[:, b*6:(b+1)*6]
            d.append(sum(len(np.unique(blk[l][cb[l]])) for l in range(L)))
        d = np.array(d); gb = d * 18.8e-3
        # recent-use predictor: gap to the previous pick of the same (layer, expert) in this window
        rec = {}
        lt_idx = np.arange(T)
        gaps = []
        for l in range(L):
            last = np.full(NE, -10**9)
            el = e[l]; cl = cold[l]
            g = np.full(el.shape, 10**9)
            for t in range(T):
                row = el[t]
                g[t] = t - last[row]
                last[row] = t
            gaps.append(g[cl])
        gaps = np.concatenate(gaps)
        r32, r256, r2048 = [(gaps <= h).mean() * 100 for h in (32, 256, 2048)]
        # token tables
        k, valid = keys_of(w)
        ck = k[cold & valid]
        tc = train_count(w, ck)
        uni = (tc > 0)
        # online, request-local: same (layer, token, expert) seen earlier in this window
        kk = np.where(cold & valid, k, -1)
        flat = kk.transpose(1, 0, 2).reshape(T, -1)            # time-major
        seen = set(); onl = []
        for t in range(T):
            for x in flat[t]:
                if x < 0: continue
                onl.append(x in seen)
            for x in flat[t]:
                if x >= 0: seen.add(x)
        onl = np.array(onl)
        # union (positions with known tokens only, aligned in time-major order)
        gap_tm = []
        for l in range(L):
            pass
        uni_tm = train_count(w, flat[flat >= 0]) > 0
        # recent256 aligned to time-major order
        G = np.full(k.shape, 10**9)
        for l in range(L):
            last = np.full(NE, -10**9)
            for t in range(T):
                G[l, t] = t - last[e[l, t]]; last[e[l, t]] = t
        G_tm = G.transpose(1, 0, 2).reshape(T, -1)[flat >= 0]
        union = (G_tm <= 256) | uni_tm | onl
        agg.append((w, coldpct, d.mean(), np.percentile(d, 95), r32, r256, r2048, uni.mean()*100, onl.mean()*100, union.mean()*100))
        print(f"{w:8s} {coldpct:5.1f}  {d.mean():6.1f} / {np.percentile(d,95):5.0f}  ({gb.mean():.2f} / {np.percentile(gb,95):.2f})"
              f"            {r32:5.1f}  {r256:5.1f}  {r2048:5.1f}   |  {uni.mean()*100:5.1f}  {onl.mean()*100:5.1f}  |  {union.mean()*100:5.1f}")
    a = np.array([x[1:] for x in agg], float)
    print("mean     " + "  ".join(f"{v:.1f}" for v in a.mean(0)))
