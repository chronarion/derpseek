"""Offline: acceptance lost by a drafter vocabulary shortlist (tokens ranked by target mass on the training steps).
   python3 -I shortlist.py DIR"""
import sys
import os as _os, sys as _sys
exec(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "fit_acc3.py")).read().split("def main():")[0])
tr = groups[("train", "train")]
# token frequency from the training steps: target top-64 mass per token (p-weighted), plus drafted tokens
V = 129280
freq = torch.zeros(V, dtype=torch.float64)
pi_t = torch.from_numpy(pi)
ix = torch.as_tensor(tr)
freq.index_add_(0, pi_t[ix].reshape(-1), lp[ix].exp().reshape(-1).double())
order = freq.argsort(descending=True)
print(f"tokens with any target mass in training: {(freq > 0).sum().item()}")
qi_t = torch.from_numpy(qi)
for n in (8192, 16384, 32768, 49152, 65536):
    keep = torch.zeros(V, dtype=torch.bool); keep[order[:n]] = True
    for key in [("lil", "held"), ("chat", "held"), ("code", "held"), ("cjk", "held"), ("writing", "held")]:
        ix = torch.as_tensor(groups[key])
        # overlap lost: min(p, q) on dropped tokens (q's top-64 and p's top-64 outside q's), position 1 and all
        m_q = ~keep[qi_t[ix]]
        lost = (torch.minimum(pq[ix], q_top[ix]) * m_q).sum(-1)
        m_p = ~keep[pi_t[ix]]
        lost += (torch.minimum(extra_p[ix], extra_q[ix]) * m_p).sum(-1)
        a = a0[ix]
        # q renormalizes over the kept tokens (mass moves up): lower bound on the change = -lost
        a_new = (a - lost)
        b0 = tps(a)[0]; b1 = tps(a_new)[0]
        print(f"  shortlist {n:6d} {key[0]:8s}: accept lost per position {lost.mean():.4f} (pos1 {lost[:, 0].mean():.4f})  tok/s {b0:.0f} -> >= {b1:.0f} ({100 * (b1 / b0 - 1):+.2f}%)")
