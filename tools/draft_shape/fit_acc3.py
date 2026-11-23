"""Offline fits of drafter corrections on acc3 dumps (C1 push). Corrections reweight the drafter's top-64 candidates,
q'(v) = q(v) exp(d_v) / Z for v in top64(q), q'(v) = q(v) / Z elsewhere, Z = sum_top q e^d + q_tail. Acceptance
sum_v min(p, q') is exact on the top-64 of q and of p; the overlap outside both sets (r_out = a - overlap on the union,
measured for the baseline) is scaled by min(1, 1/Z). Chain value per step: E = sum_j prod_{i<=j} a_i (positions > 1 are
conditioned on DSpark's own drafts, as in the acc2 ceilings). Fits on the training prompts only; scored on the evaluation
prompts (tune and held halves) and lil, per kind, as tokens per step and tok/s at the best fixed width.
   python3 -I fit_acc3.py DIR   (DIR holds acc3.jsonl{,.bin,.h16}, acc3-phases.txt, real3.jsonl, periods.json)"""
import json
import math
import sys
from collections import defaultdict

import numpy as np
import torch

D = sys.argv[1]
K, TOP = 7, 64
torch.manual_seed(0)
meta = [json.loads(l) for l in open(f"{D}/acc3.jsonl")]
N = len(meta)
B = np.fromfile(f"{D}/acc3.jsonl.bin", dtype=np.float32)
N = min(N, B.size // (K * 6 * TOP))
meta = meta[:N]
B = B[:N * K * 6 * TOP].reshape(N, K, 6, TOP)
HH = None
try:
    h = np.fromfile(f"{D}/acc3.jsonl.h16", dtype=np.float16)
    Hd = h.size // (N * K) if N else 0
    if Hd:
        HH = h[:N * K * Hd].reshape(N, K, Hd)
except FileNotFoundError:
    pass
qi = B[:, :, 0].view(np.int32).astype(np.int64)
lq = torch.from_numpy(B[:, :, 1].copy())
pq = torch.from_numpy(B[:, :, 2].copy())
pi = B[:, :, 3].view(np.int32).astype(np.int64)
lp = torch.from_numpy(B[:, :, 4].copy())
qp = torch.from_numpy(B[:, :, 5].copy())
a0 = torch.tensor([m["a"] for m in meta])
# p-top tokens that are not in q's top-64: their (p, q) enter the overlap exactly, with q scaled by 1/Z
inq = torch.from_numpy((pi[..., :, None] == qi[..., None, :]).any(-1))
p_top = lp.exp()
extra_p = p_top.masked_fill(inq, 0)
extra_q = qp.masked_fill(inq, 0)
q_top = lq.exp()
q_tail = (1 - q_top.sum(-1)).clamp(min=0)
ov_base = torch.minimum(pq, q_top).sum(-1) + torch.minimum(extra_p, extra_q).sum(-1)
r_out = (a0 - ov_base).clamp(min=0)
ent = -(q_top * lq).sum(-1)                       # entropy over the top-64 (the tail adds little)
rank = torch.arange(TOP, dtype=torch.float32)

# labels by time: phases, eval requests from real3.jsonl, lil cells alternate tune / held in 200-step blocks
ph = [l.split() for l in open(f"{D}/acc3-phases.txt")]
ph = [(n, float(t)) for n, t in ph]
real = [json.loads(l) for l in open(f"{D}/real3.jsonl")]
t = np.array([m["t"] for m in meta])


def label(k):
    tt = t[k]
    name = None
    for n, t0 in ph:
        if tt >= t0:
            name = n
    if name == "train":
        return "train", "train"
    if name == "eval":
        for r in real:
            if r["t0"] <= tt <= r["t0"] + r["s"] + 0.5:
                return r["kind"], r["half"]
        return None, None
    if name == "lil":
        return "lil", "tune" if (k // 200) % 2 == 0 else "held"
    return None, None


lab = [label(k) for k in range(N)]
groups = defaultdict(list)
for k, (kind, half) in enumerate(lab):
    if kind:
        groups[(kind, half)].append(k)
periods = json.load(open(f"{D}/periods.json"))


def accept(d):
    """d: [n, K, TOP] log-weights on q's top-64 -> per-position acceptance [n, K]"""
    w = q_top_s * torch.exp(d)
    Z = w.sum(-1) + q_tail_s
    qn = w / Z[..., None]
    a = torch.minimum(pq_s, qn).sum(-1) + torch.minimum(extra_p_s, extra_q_s / Z[..., None]).sum(-1)
    return a + r_out_s * torch.clamp(1 / Z, max=1)


def chain(a, W=K):
    return torch.cumprod(a[:, :W], 1).sum(1)


def tps(a):
    best = (0, 0, 0)
    for W in range(1, K + 1):
        tk = 1 + chain(a, W).mean().item()
        v = tk / periods[str(W)] * 1e3
        if v > best[0]:
            best = (v, W, tk)
    return best


def select(idx):
    global q_top_s, q_tail_s, pq_s, extra_p_s, extra_q_s, r_out_s
    ix = torch.as_tensor(idx)
    q_top_s, q_tail_s, pq_s = q_top[ix], q_tail[ix], pq[ix]
    extra_p_s, extra_q_s, r_out_s = extra_p[ix], extra_q[ix], r_out[ix]
    return ix


# ---- models: d = f(features; theta) ----
POSF = torch.eye(K)


def feats(ix):
    """per candidate: [lq, rank one-hots (0, 1, 2-3, 4-7, 8+), entropy, position one-hot] -> interactions"""
    l = lq[ix]                                      # [n, K, 64]
    e = ent[ix][..., None].expand_as(l)
    rb = torch.stack([(rank == 0), (rank == 1), (rank >= 2) & (rank < 4), (rank >= 4) & (rank < 8), rank >= 8], -1).float()
    rb = rb.expand(*l.shape, 5)
    pos = POSF[None, :, None, :].expand(*l.shape, K)
    base = torch.cat([l[..., None], (l * e)[..., None], rb, rb * e[..., None]], -1)   # 12
    return torch.cat([base, (base[..., :1] * pos)], -1)                                    # + per-position temperature


class Lin(torch.nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(nf))

    def forward(self, f):
        return f @ self.w


def fit(model, featf, idx, epochs=60, lr=0.02, bs=4096):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    idx = np.array(idx)
    for ep in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            ix = select(b)
            a = accept(model(featf(ix)))
            loss = -chain(a).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def evaluate(name, dfun):
    print(f"== {name}")
    for key in sorted(groups):
        if key[0] == "train":
            continue
        ix = select(groups[key])
        with torch.no_grad():
            a = accept(dfun(ix))
            ab = accept(torch.zeros_like(lq[ix]))
        b, m = tps(ab), tps(a)
        print(f"  {key[0]:11s} {key[1]:5s} n={len(ix):6d}  pos1 a {ab[:, 0].mean():.3f}->{a[:, 0].mean():.3f}  "
              f"base {b[0]:.0f}@W{b[1]} ({b[2]:.2f})  model {m[0]:.0f}@W{m[1]} ({m[2]:.2f})  {100 * (m[0] / b[0] - 1):+.1f}%")


def segments(idx):
    """split a time-ordered index list into requests (history length drops or gaps > 2 s)"""
    out, cur, prev = [], [], None
    for k in idx:
        m = meta[k]
        if prev is not None and ((m["hl"] or 0) < (meta[prev]["hl"] or 0) or m["t"] - meta[prev]["t"] > 2):
            out.append(cur)
            cur = []
        cur.append(k)
        prev = k
    if cur:
        out.append(cur)
    return out


VOCAB = 129280


def online(idx, eta, upd_pos=3, decay=1.0):
    """per-request token bias b (dense vocab), applied to q's top-64 at every position, updated after each step from
    the verified positions 1..upd_pos: b[v] += eta * (p_v - q'_v) on q's top-64 and p's top-64 (a KL(p || q') step).
    Returns per-position acceptance [n, K] in idx order."""
    res = torch.zeros(len(idx), K)
    pos_of = {k: i for i, k in enumerate(idx)}
    qi_t, pi_t = torch.from_numpy(qi), torch.from_numpy(pi)
    for seg in segments(idx):
        b = torch.zeros(VOCAB)
        for k in seg:
            ix = select([k])
            d = b[qi_t[k]][None]                              # [1, K, 64]
            a = accept(d)[0]
            res[pos_of[k]] = a
            w = q_top[k] * torch.exp(d[0])
            Z = w.sum(-1) + q_tail[k]
            qn = w / Z[:, None]
            g_q = (pq[k] - qn)[:upd_pos]                      # on q's top-64
            g_p = (extra_p[k] - extra_q[k] / Z[:, None])[:upd_pos]   # on p's top-64 outside q's top
            if decay != 1.0:
                b.mul_(decay)
            b.index_add_(0, qi_t[k][:upd_pos].reshape(-1), eta * g_q.reshape(-1))
            b.index_add_(0, pi_t[k][:upd_pos].reshape(-1), eta * g_p.reshape(-1))
    return res


def evaluate_online(name, eta, **kw):
    print(f"== {name}")
    for key in sorted(groups):
        if key[0] == "train":
            continue
        idx = groups[key]
        a = online(idx, eta, **kw)
        select(idx)
        with torch.no_grad():
            ab = accept(torch.zeros(len(idx), K, TOP))
        b, m = tps(ab), tps(a)
        print(f"  {key[0]:11s} {key[1]:5s} n={len(idx):6d}  pos1 a {ab[:, 0].mean():.3f}->{a[:, 0].mean():.3f}  "
              f"base {b[0]:.0f}@W{b[1]} ({b[2]:.2f})  model {m[0]:.0f}@W{m[1]} ({m[2]:.2f})  {100 * (m[0] / b[0] - 1):+.1f}%")


def main():
    print(f"{N} steps; groups: {[(k, len(v)) for k, v in sorted(groups.items())]}; hidden {None if HH is None else HH.shape}")
    # sanity: the reconstruction reproduces a at d = 0
    ix = select(np.arange(N))
    with torch.no_grad():
        print(f"reconstruction error at d=0: {(accept(torch.zeros_like(lq)) - a0).abs().max():.2e}")
    tr = groups[("train", "train")]
    # (1) per-position temperature only
    m1 = Lin(K)
    fit(m1, lambda ix: lq[ix][..., None] * POSF[None, :, None, :], tr)
    print("temperature-like weights:", [round(1 + x, 3) for x in m1.w.tolist()])
    evaluate("per-position power", lambda ix: m1(lq[ix][..., None] * POSF[None, :, None, :]))
    # (2) features: power by entropy, rank bonuses
    f0 = feats(np.arange(2))
    m2 = Lin(f0.shape[-1])
    fit(m2, feats, tr)
    print("feature weights:", [round(x, 3) for x in m2.w.tolist()])
    evaluate("shape calibration (entropy, rank, position)", lambda ix: m2(feats(ix)))
    # (3) online per-request token bias: eta chosen on the training prompts
    tr_sub = tr[:min(len(tr), 15000)]
    best = None
    for eta in (0.5, 1, 2, 4, 8):
        a = online(tr_sub, eta)
        v = (1 + chain(a)).mean().item()
        print(f"  online eta {eta}: train tokens/step at W7 {v:.3f}")
        if best is None or v > best[1]:
            best = (eta, v)
    with torch.no_grad():
        select(tr_sub)
        print(f"  baseline train tokens/step at W7 {(1 + chain(accept(torch.zeros(len(tr_sub), K, TOP)))).mean():.3f}")
    evaluate_online(f"online token bias (eta {best[0]})", best[0])


if __name__ == "__main__":
    main()
