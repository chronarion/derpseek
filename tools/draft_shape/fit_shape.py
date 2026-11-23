"""Variants of the context-free q reshape that a sampling kernel can apply pointwise (C1 push). Fit on training
prompts, scored held-out. d_v is added to the logit of every candidate with q_v >= TAU (q's top-64 restricted to it;
the rest keeps its logit), features from (log q_v, entropy of q, position) only.
   python3 -I fit_shape.py DIR"""
import sys
exec(open(__file__.replace("fit_shape.py", "fit_acc3.py")).read().split("def main():")[0])
TAU = 1e-3


def pbins(l):
    q = l.exp()
    return torch.stack([q >= .5, (q >= .2) & (q < .5), (q >= .05) & (q < .2), (q >= .01) & (q < .05), q < .01], -1).float()


def F_pos(ix):            # per-position power
    l = lq[ix]
    return (l[..., None] * POSF[None, :, None, :]), l >= math.log(TAU)


def F_pe(ix):             # power by position and entropy, entropy offset
    l = lq[ix]; e = ent[ix][..., None].expand_as(l)
    f = torch.cat([l[..., None] * POSF[None, :, None, :], (l * e)[..., None], e[..., None]], -1)
    return f, l >= math.log(TAU)


def F_pb(ix):             # + probability bins (pointwise), bins x entropy
    l = lq[ix]; e = ent[ix][..., None].expand_as(l); b = pbins(l)
    f = torch.cat([l[..., None] * POSF[None, :, None, :], (l * e)[..., None], b, b * e[..., None]], -1)
    return f, l >= math.log(TAU)


class M(torch.nn.Module):
    def __init__(self, nf):
        super().__init__(); self.w = torch.nn.Parameter(torch.zeros(nf))
    def forward(self, fm):
        f, m = fm
        return (f @ self.w) * m


def fitm(Ff, tr):
    f0, _ = Ff(np.arange(2)); m = M(f0.shape[-1])
    fit(m, Ff, tr, epochs=40)
    return m


ent = -lq[..., 0]          # uncertainty = -log q_max (the kernel has it exactly: log of the normalizer's sum)
tr = groups[("train", "train")]
for name, Ff in (("+ probability bins, -log qmax", F_pb),):
    m = fitm(Ff, tr)
    print(name, "weights", [round(x, 3) for x in m.w.tolist()])
    json.dump({"w": m.w.tolist(), "log_tau": math.log(TAU), "features": "lq*pos[7], lq*u, bins[5], bins*u; u=-log qmax"},
              open(f"{D}/shape_coef.json", "w"))
    evaluate(f"{name} (q >= {TAU})", lambda ix, m=m, Ff=Ff: m(Ff(ix)))
