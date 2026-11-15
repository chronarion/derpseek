"""docs/rare-experts.md, second pass: which rarely used experts are specialists, with the statistics a claim needs.
Routing files ("experts" [40, T, 6], dsv41's ref_logits.py) for the KLD corpus and the varied corpus
(tools/varied_corpus.py); windows are the unit (each is packed from several documents of one kind).
  split       discovery = each category's first window (-0), validation = the others
  enrichment  per expert x text group, discovery only: log-odds ratio with an informative Dirichlet prior (Monroe et
              al. 2008) of the expert's picks in the group against outside it; significance by a studentized permutation test
              (group labels shuffled across windows, --perms), Benjamini-Hochberg over every expert x group test (q = --q); an expert counts
              as a specialist only with log-odds >= log 3, picks in >= 3 windows and no window > 50% of its picks
  null        the same pipeline on pseudo-experts (random positions, same pick count, same layer): the number of
              "specialists" it finds is printed next to every count
  replication each discovery specialist re-tested on validation windows (same sign, studentized permutation p <= 0.05)
  token       per layer, how well a token's most frequent experts in discovery predict its experts in validation
              (mean Jaccard of the top-6 sets against the chance level 36 / 384)
   python3 -I rare_experts_stats.py ROWMAP.json OUT.json CORPUS.json:ROUTING_DIR [CORPUS.json:ROUTING_DIR ...]"""
import argparse
import collections
import json
import math
import os
import sys

import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("rowmap")
ap.add_argument("out")
ap.add_argument("sources", nargs="+")
ap.add_argument("--perms", type=int, default=400)
ap.add_argument("--q", type=float, default=0.01)
ap.add_argument("--alpha0", type=float, default=1000.0)
a = ap.parse_args()
torch.manual_seed(20261115)
L, E, K = 40, 384, 6
R = json.load(open(a.rowmap))["layers"]
cold = torch.zeros(L, E, dtype=torch.bool)
for l in range(L):
    cold[l, sorted(set(R[str(l)]["host"]) - set(R[str(l)]["cache"]))] = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench", "deb"))


def group_of(stratum):
    kind, _, rest = stratum.partition("_")
    if kind in ("wiki", "web"):
        lat = {"en", "de", "fr", "es", "it", "pt", "nl", "sv", "fi", "pl", "tr", "vi", "id", "sw", "yo", "spa", "eng"}
        return "latin-script" if rest in lat else "cjk" if rest in {"ja", "ko", "zh", "cmn", "jpn"} else "other-scripts"
    if kind == "code":
        rare = {"haskell", "commonlisp", "fortran", "cobol", "x86assembly", "prolog", "erlang", "ocaml", "forth", "ada",
                "algol68", "j"}
        return "code-rare" if rest in rare else "code-main"
    return {"struct": "structured", "chat": "chat", "math": "math-science", "sci": "math-science"}.get(kind, "ordinary")


wins = []   # (id, group, split, ids[T], experts[40, T, 6])
for src in a.sources:
    cpath, rdir = src.split(":")
    for w in json.load(open(cpath)):
        f = os.path.join(rdir, w["id"] + ".safetensors")
        if not os.path.exists(f):
            continue
        with safe_open(f, "pt") as fh:
            if "experts" not in fh.keys():
                continue
            ex = fh.get_tensor("experts").long()
        T = ex.shape[1]
        stratum = w.get("stratum", w["id"].split("-")[0])
        grp = group_of(stratum) if "_" in stratum else "ordinary"
        wins.append((w["id"], grp, "disc" if w["id"].endswith("-0") else "val", w["ids"][:T], ex[:, 1:]))  # no pos 0
groups = sorted({g for _, g, _, _, _ in wins})
G = len(groups)
gi = {g: i for i, g in enumerate(groups)}
print(f"{len(wins)} windows ({sum(1 for w in wins if w[2] == 'disc')} discovery), groups: {', '.join(groups)}")


def counts(split):
    """[W, L, E] picks per window, group index per window, picks per window per layer."""
    ws = [w for w in wins if w[2] == split]
    C = torch.zeros(len(ws), L, E)
    for i, (_, _, _, _, ex) in enumerate(ws):
        for l in range(L):
            C[i, l].index_add_(0, ex[l].reshape(-1), torch.ones(ex[l].numel()))
    return ws, C, torch.tensor([gi[w[1]] for w in ws])


def logodds(C, grp, alpha0):
    """[G, L, E] log-odds ratio (picks in group vs outside, informative Dirichlet prior) and its z."""
    onehot = torch.zeros(G, C.shape[0])
    onehot[grp, torch.arange(C.shape[0])] = 1
    a_in = torch.einsum("gw,wle->gle", onehot, C)
    n_in = a_in.sum(-1, keepdim=True)
    tot = C.sum(0, keepdim=True)
    a_out, n_out = tot - a_in, tot.sum(-1, keepdim=True) - n_in
    p = (tot / tot.sum(-1, keepdim=True))
    al = alpha0 * p
    d = (torch.log((a_in + al) / (n_in + alpha0 - a_in - al)) - torch.log((a_out + al) / (n_out + alpha0 - a_out - al)))
    z = d / torch.sqrt(1 / (a_in + al) + 1 / (a_out + al))
    return d, z, a_in


def perm_p(C, grp, z):
    """Studentized permutation test: the permuted z-scores of each test (group labels shuffled across windows, sizes
    kept) give its null mean and spread, which absorb the extra variance of text arriving in whole windows; p is the
    upper normal tail of the standardized score, so it can go as low as the evidence allows (a plain permutation p
    cannot go below 1 / (perms + 1), far above what Benjamini-Hochberg needs over ~10^5 tests)."""
    s1, s2 = torch.zeros_like(z), torch.zeros_like(z)
    for _ in range(a.perms):
        _, zp, _ = logodds(C, grp[torch.randperm(len(grp))], a.alpha0)
        s1 += zp
        s2 += zp * zp
    mu = s1 / a.perms
    sd = (s2 / a.perms - mu * mu).clamp(min=1e-6).sqrt()
    return 0.5 * torch.erfc(((z - mu) / sd) / math.sqrt(2))


def specialists(C, grp, label):
    d, z, a_in = logodds(C, grp, a.alpha0)
    pval = perm_p(C, grp, z)
    # Benjamini-Hochberg over every expert x group test
    flat = pval.reshape(-1)
    order = torch.argsort(flat)
    m = flat.numel()
    thresh = a.q * torch.arange(1, m + 1) / m
    passed = flat[order] <= thresh
    kmax = int(passed.nonzero().max()) + 1 if passed.any() else 0
    sig = torch.zeros(m, dtype=torch.bool)
    sig[order[:kmax]] = True
    sig = sig.reshape(pval.shape)
    # effect size and dispersion: log-odds >= log 3, in >= 3 windows, no window > 50% of its picks
    tot = C.sum(0)
    nwin = (C > 0).sum(0)
    maxshare = C.max(0).values / tot.clamp(min=1)
    ok = (d >= math.log(3)) & (nwin >= 3) & (maxshare <= 0.5) & (tot >= 20)
    spec = sig & ok
    print(f"  {label}: {int(spec.sum())} expert x group specialists (BH q={a.q}), "
          f"{int(spec.any(0).sum())} distinct experts; cold {int((spec.any(0) & cold).sum())} of {int(cold.sum())}, "
          f"hot {int((spec.any(0) & ~cold).sum())} of {int((~cold).sum())}")
    return spec, d, z


ws_d, C_d, g_d = counts("disc")
ws_v, C_v, g_v = counts("val")
print("discovery:")
spec, d, z = specialists(C_d, g_d, "real experts")
# pseudo-experts: each layer's picks reassigned at random to experts, keeping every expert's pick count (positions
# random, counts matched), so any "specialist" here is the method's false-discovery rate
P = torch.zeros_like(C_d)
for l in range(L):
    flat = torch.cat([torch.full((int(C_d[i, l].sum()),), i) for i in range(C_d.shape[0])])
    perm = flat[torch.randperm(len(flat))]
    ids = torch.repeat_interleave(torch.arange(E), C_d[:, l].sum(0).long())
    P[:, l].index_put_((perm, ids), torch.ones(len(perm)), accumulate=True)
pspec, _, _ = specialists(P, g_d, "pseudo-experts (same pick counts, random positions)")
# replication on validation windows
dv, zv, _ = logodds(C_v, g_v, a.alpha0)
pv = perm_p(C_v, g_v, zv)
rep = spec & (dv > 0) & (pv <= 0.05)
print(f"validation: {int(rep.sum())} of {int(spec.sum())} discovery specialists replicate "
      f"({int(rep.sum()) / max(int(spec.sum()), 1):.0%}); cold {int((rep.any(0) & cold).sum())} experts, "
      f"hot {int((rep.any(0) & ~cold).sum())}")
per_group = {g: (int(rep[gi[g]].sum()), int((rep[gi[g]] & cold).sum())) for g in groups}
print("replicated specialists per group (all / cold): " + ", ".join(f"{g} {n}/{c}" for g, (n, c) in per_group.items()))
# token predictability: a token's top-6 experts in discovery vs its experts in validation
print("token predictability (mean Jaccard of a token's 6 most frequent discovery experts with its validation experts; "
      f"chance {36 / 384:.3f}):")
tokj = []
TOKL = [0, 5, 10, 15, 20, 25, 30, 35, 39]
for l in TOKL:
    top = {}
    cnt = collections.defaultdict(collections.Counter)
    for _, _, _, ids, ex in [w for w in wins if w[2] == "disc"]:
        for t, row in zip(ids[1:], ex[l].tolist()):
            cnt[t].update(row)
    top = {t: {e for e, _ in c.most_common(6)} for t, c in cnt.items()}
    js = []
    for _, _, _, ids, ex in [w for w in wins if w[2] == "val"]:
        for t, row in zip(ids[1:], ex[l].tolist()):
            if t in top:
                s = set(row)
                js.append(len(s & top[t]) / len(s | top[t]))
    tokj.append(sum(js) / max(len(js), 1))
print("  " + " ".join(f"L{l}:{j:.2f}" for l, j in zip(TOKL, tokj)))
res = {"groups": groups, "specialists": int(spec.sum()), "pseudo_specialists": int(pspec.sum()),
       "replicated": int(rep.sum()), "per_group": per_group, "token_jaccard": dict(zip(TOKL, tokj)),
       "replicated_list": [(groups[g], l, e, float(d[g, l, e]), float(dv[g, l, e]), bool(cold[l, e]))
                           for g, l, e in rep.nonzero().tolist()]}
json.dump(res, open(a.out, "w"))
