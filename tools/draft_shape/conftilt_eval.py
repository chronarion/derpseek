"""Carbon lever 2B, scored offline on acc3 dumps: tilt each draft position's proposal toward candidates after which the
checkpoint's confidence head expects the next position to be accepted.
  conf_{j+1}(v) = sigmoid(w_h . H_{j+1} + w_e . M[v])   (mtp.2.confidence_head: [head hidden; Markov embed of the
                                                         previous token] -> logit; M = the Markov head's embedding)
  q~_j(v) ~ q_j(v) exp(beta conf_{j+1}(v)) on q's top-64 (the tail keeps its shape).
Position j's own acceptance with q~ is exact (sum min(p, q~)). The next position's acceptance changes through which
token is accepted at j: E[a_{j+1}] shifts by sum_v [min(p,q~)(v)/a~_j - min(p,q)(v)/a_j] conf_{j+1}(v), with the
head (calibrated on dsv41's logs, checked here) as the estimator. Positions further on keep their measured values.
   python3 -I conftilt_eval.py DIR MODEL_DIR"""
import json
import sys

import numpy as np
import torch
from safetensors import safe_open

D, MD = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0], D]
exec(open(__file__.replace("conftilt_eval.py", "fit_acc3.py")).read().split("def main():")[0])
idx = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]


def load(name):
    with safe_open(f"{MD}/{idx[name]}", "pt") as f:
        return f.get_tensor(name)


w = load("mtp.2.confidence_head.proj.weight").float().view(-1)
M = load("mtp.2.markov_head.embed.weight").float()
Hd = HH.shape[-1]
w_h, w_e = w[:Hd], w[Hd:]
s_v = M @ w_e                                    # [V]
print(f"conf head: hidden {Hd}, markov rank {M.shape[1]}; token term std {s_v.std():.3f}, range {s_v.min():.2f}..{s_v.max():.2f}")
Ht = torch.from_numpy(HH)
qi_t = torch.from_numpy(qi)
d_tok = torch.tensor([m["d"][:K] for m in meta])  # drafted tokens per position


def offsets(ix):
    return Ht[ix].float() @ w_h                   # [n, K]


def calib(ix):
    """predicted conf_{j+1}(y_j) for the drafted token vs the measured a_{j+1}"""
    o = offsets(ix)
    c = torch.sigmoid(o[:, 1:] + s_v[d_tok[ix][:, :-1]])
    a = a0[ix][:, 1:]
    bins = torch.tensor([0, .2, .4, .6, .8, .9, 1.01])
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (c >= lo) & (c < hi)
        if m.any():
            out.append(f"[{lo:.1f},{hi:.1f}) n={int(m.sum())} pred {c[m].mean():.3f} meas {a[m].mean():.3f}")
    return out


def tilted(ix, beta):
    """per-position acceptance [n, K] with the tilt (lookahead shift applied to positions 2..K)"""
    o = offsets(ix)
    cand = qi_t[ix]                              # [n, K, 64]
    cn = torch.zeros(len(ix), K, TOP)
    cn[:, :-1] = torch.sigmoid(o[:, 1:, None] + s_v[cand[:, :-1]])   # conf of the next position per candidate
    d = beta * cn
    d[:, -1] = 0                                 # the last position has no next one
    select(ix)
    a_t = accept(d)
    # which token is accepted at j: min(p, q) restricted to q's top-64 (the tail's share keeps the old conf mean)
    def acc_mix(dd):
        ww = q_top_s * torch.exp(dd)
        Z = ww.sum(-1) + q_tail_s
        return torch.minimum(pq_s, ww / Z[..., None])
    m0, m1 = acc_mix(torch.zeros_like(d)), acc_mix(d)
    e0 = (m0 * cn).sum(-1) / m0.sum(-1).clamp(min=1e-9)
    e1 = (m1 * cn).sum(-1) / m1.sum(-1).clamp(min=1e-9)
    shift = torch.zeros_like(a_t)
    shift[:, 1:] = (e1 - e0)[:, :-1]
    return (a_t + shift).clamp(0, 1)


if __name__ == "__main__":
    tr = groups[("train", "train")]
    print("calibration of the head on the drafted tokens (train):")
    for l in calib(torch.as_tensor(tr[:20000])):
        print("  ", l)
    best = (0.0, None)
    for beta in (0.25, 0.5, 1.0, 2.0, 4.0):
        ix = torch.as_tensor(tr[:20000])
        v = (1 + chain(tilted(ix, beta))).mean().item()
        select(ix)
        v0 = (1 + chain(accept(torch.zeros(len(ix), K, TOP)))).mean().item()
        print(f"beta {beta}: train tokens/step at W7 {v:.4f} (base {v0:.4f})")
        if v - v0 > best[0]:
            best = (v - v0, beta)
    beta = best[1]
    print(f"chosen beta on train: {beta}")
    if beta is not None:
        for key in sorted(groups):
            if key[0] == "train":
                continue
            ix = torch.as_tensor(groups[key])
            at = tilted(ix, beta)
            select(ix)
            ab = accept(torch.zeros(len(ix), K, TOP))
            b, m = tps(ab), tps(at)
            print(f"  {key[0]:11s} {key[1]:5s} n={len(ix):6d} base {b[0]:.0f}@W{b[1]} ({b[2]:.2f})  tilt {m[0]:.0f}@W{m[1]} "
                  f"({m[2]:.2f})  {100 * (m[0] / b[0] - 1):+.1f}%")
