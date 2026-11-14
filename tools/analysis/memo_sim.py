"""Routing memo: do the cold experts a continuation uses already appear in its prompt's routing (prefill)?"""
import glob, os, sys, numpy as np
from safetensors import safe_open
R = sys.argv[1]; NE, L = 384, 40
W = {}
for f in sorted(glob.glob(f"{R}/*.safetensors")):
    w = os.path.basename(f)[:-12]
    with safe_open(f, "np") as h:
        W[w] = h.get_tensor("experts").astype(np.int64)
names = list(W)
cnt = {w: np.stack([np.bincount(W[w][l].ravel(), minlength=NE) for l in range(L)]) for w in names}
tot = sum(cnt.values())
def resident(w, n):
    c = (tot - cnt[w]).ravel(); res = np.zeros(L * NE, bool)
    res[np.argsort(-c, kind="stable")[:n]] = True
    return res.reshape(L, NE)
NRES = 10000
print(f"static resident {NRES} slots (ranked without the window). prompt = first P positions, continuation = the rest")
print("window    P     cold experts in prompt (GB FP4) | continuation cold picks covered by the prompt's set | + by the continuation's own earlier picks")
rows = []
for w in names:
    e = W[w]; T = e.shape[1]; res = resident(w, NRES)
    for P in ([512, 1024, 1536] if T == 2048 else [2048, 6144]):
        pset = np.zeros((L, NE), bool)
        for l in range(L):
            x = e[l, :P].ravel(); x = x[~res[l, x]]; pset[l, x] = True
        seen = pset.copy(); cov = covs = n = 0
        for t in range(P, T):
            for l in range(L):
                for x in e[l, t]:
                    if res[l, x]: continue
                    n += 1; cov += pset[l, x]; covs += seen[l, x]; seen[l, x] = True
        k = int(pset.sum())
        rows.append((w, P, k, cov / n * 100, covs / n * 100))
        print(f"{w:8s} {P:5d}  {k:6d} ({k*0.0188:5.1f} GB)          | {cov/n*100:5.1f}%                                   | {covs/n*100:5.1f}%")
for P in (512, 1024, 1536):
    r = [x for x in rows if x[1] == P]
    print(f"mean P={P}: prompt set {np.mean([x[2] for x in r]):.0f} experts, covers {np.mean([x[3] for x in r]):.1f}%, with own history {np.mean([x[4] for x in r]):.1f}%")
