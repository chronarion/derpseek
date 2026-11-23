"""Carbon c5 idea 5: an entropy-gated 2-way branch at position 1 (recursive rejection, the sibling drawn from q without
replacement), only on steps whose position-1 entropy >= theta, at widths W <= 3 (the extra row costs T(W+1) - T(W)).
theta chosen on the training steps; scored held out at the best fixed width per kind."""
import os as _os
exec(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "fit_acc3.py")).read().split("def main():")[0])
rng = np.random.default_rng(0)
per = {int(k): v for k, v in periods.items()}
def rrs(pv, qv, k):
    p, q = pv.copy(), qv.copy(); alive = np.ones_like(q, bool)
    for i in range(k):
        qq = q * alive; s = qq.sum(); x = rng.choice(len(q), p=qq / s)
        if rng.random() < min(1.0, p[x] / max(qq[x] / s, 1e-12)): return True, i
        r = np.maximum(p - qq / s, 0); rs = r.sum()
        if rs <= 0: return False, None
        p = r / rs; alive[x] = False
    return False, None
ent1 = ent[:, 0].numpy()
def run(ix, theta, W):
    tok, tim = 0.0, 0.0
    for s in ix:
        a = a0[s].numpy()
        branch = ent1[s] >= theta and W <= 3
        if branch:
            qv = np.concatenate([q_top[s, 0].numpy(), extra_q[s, 0].numpy(), [max(q_tail[s, 0].item() - extra_q[s, 0].sum().item(), 0)]])
            pv = np.concatenate([pq[s, 0].numpy(), extra_p[s, 0].numpy(), [0.0]]); pv[-1] = max(1 - pv.sum(), 0)
            acc, which = rrs(pv / pv.sum(), qv / qv.sum(), 2)
            e = 0 if not acc else (1 + float(np.cumprod(a[1:W]).sum()) if which == 0 else 1)
            tok += 1 + e; tim += per[W + 1]
        else:
            tok += 1 + float(np.cumprod(a[:W]).sum()); tim += per[W]
    return tok / len(ix), tim / len(ix)
tr = np.array(groups[("train", "train")])[::20]
best = max(((t / tm, th) for th in (0.5, 1.0, 1.5, 2.0, 3.0, 99) for W in (2, 3) for t, tm in [run(tr, th, W)]))
theta = best[1]
print(f"theta chosen on train: {theta}")
for key in [("lil", "held"), ("chat", "held"), ("writing", "held"), ("cjk", "held")]:
    ix = np.array(groups[key])[::3]
    r = {}
    for th in (99, theta):
        r[th] = max(((t / tm * 1e3, W) for W in range(1, 8) for t, tm in [run(ix, th, W)]))
    print(f"{key[0]:8s}: base {r[99][0]:.0f}@W{r[99][1]}  gated branch {r[theta][0]:.0f}@W{r[theta][1]}  {100 * (r[theta][0] / r[99][0] - 1):+.1f}%")
