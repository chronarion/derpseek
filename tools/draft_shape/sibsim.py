"""Offline: position-1 sibling candidates by recursive rejection sampling without replacement (SpecTr / RRS-style).
Candidates x1, x2, x3 ~ q without replacement; x1 tested with min(1, p/q); on rejection p <- norm((p - q)+), q <- q
without x1 renormalized, test x2 with min(1, p'/q'), ... Only x1 carries the drafted chain (positions 2..W); a sibling
accepted ends the step with its row's bonus token. Monte Carlo on the sparse top-64 data (tail lumped as one atom)."""
import os as _os, sys as _sys
exec(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "fit_acc3.py")).read().split("def main():")[0])
import random
rng = np.random.default_rng(0)
per = {int(k): v for k, v in periods.items()}
def rrs(pv, qv, k):
    p, q = pv.copy(), qv.copy(); alive = np.ones_like(q, bool)
    for i in range(k):
        qq = q * alive; s = qq.sum()
        if s <= 0: return False, None
        x = rng.choice(len(q), p=qq / s)
        if rng.random() < min(1.0, p[x] / max(qq[x] / s, 1e-12)):
            return True, i
        r = np.maximum(p - qq / s, 0); rs = r.sum()
        if rs <= 0: return False, None
        p = r / rs; alive[x] = False
    return False, None
for key in [("lil", "held"), ("chat", "held"), ("writing", "held"), ("code", "held")]:
    ix = np.array(groups[key])[::4][:1500]
    res = {}
    for k in (1, 2, 3):
        tot = {W: 0.0 for W in range(1, 8)}
        for s in ix:
            # position-1 distributions over the union (q top-64 + p extra + tail atom)
            qv = np.concatenate([q_top[s, 0].numpy(), extra_q[s, 0].numpy(), [max(q_tail[s, 0].item() - extra_q[s, 0].sum().item(), 0)]])
            pv = np.concatenate([pq[s, 0].numpy(), extra_p[s, 0].numpy(), [0.0]])
            pv[-1] = max(1 - pv.sum(), 0); qv = qv / qv.sum(); pv = pv / pv.sum()
            acc, which = rrs(pv, qv, k)
            a = a0[s].numpy()
            for W in range(1, 8):
                if not acc: e = 0
                elif which == 0: e = 1 + float(np.cumprod(a[1:W]).sum())   # the primary child's chain (a_j given x1)
                else: e = 1
                tot[W] += 1 + e
        n = len(ix)
        best = max(((tot[W] / n) / (per[W + k - 1 if W + k - 1 <= 7 else 7] + 0.25 * max(0, W + k - 1 - 7)) * 1e3, W) for W in range(1, 8))
        res[k] = best
    b = res[1][0]
    print(f"{key[0]:8s}: " + "  ".join(f"k={k}: {v[0]:.0f} tok/s @W{v[1]} ({100 * (v[0] / b - 1):+.1f}%)" for k, v in res.items()))
