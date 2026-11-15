"""docs/rare-experts.md: what do the rarely used experts do? Routing of the official model on the 18 KLD corpus windows (49,152 tokens;
experts [40, T, 6] in the reference files), the cold set = each layer's CPU-memory experts outside the initial cache
(Derpseek's rowmap). Prints a compact summary and writes cold.json."""
import collections, json, sys
import torch
from safetensors import safe_open
from tokenizers import Tokenizer

CORPUS, REF, ROWMAP, TOKJ, OUT = sys.argv[1:6]
tok = Tokenizer.from_file(TOKJ)
W = json.load(open(CORPUS))
R = json.load(open(ROWMAP))["layers"]
L, E = 40, 384
cold = {l: sorted(set(R[str(l)]["host"]) - set(R[str(l)]["cache"])) for l in range(L)}
fams = sorted({w["id"].split("-")[0] for w in W})
cnt = torch.zeros(L, E, dtype=torch.int64)
famc = {f: torch.zeros(L, E, dtype=torch.int64) for f in fams}
rankc = torch.zeros(L, E, 6, dtype=torch.int64)
famtok = collections.Counter()
tokfreq = collections.Counter()
hits = collections.defaultdict(list)          # (l, e) -> [(window, t, rank)]
ids_of = {}
for wi, w in enumerate(W):
    with safe_open(f"{REF}/{w['id']}.safetensors", "pt") as f:
        ex = f.get_tensor("experts").long()   # [40, T, 6]
    T = ex.shape[1]
    ids = w["ids"][:T]
    ids_of[wi] = ids
    fam = w["id"].split("-")[0]
    famtok[fam] += T
    tokfreq.update(ids)
    for l in range(L):
        x = ex[l]
        one = torch.ones(x.numel(), dtype=torch.int64)
        cnt[l].index_add_(0, x.reshape(-1), one)
        famc[fam][l].index_add_(0, x.reshape(-1), one)
        for k in range(6):
            rankc[l, :, k].index_add_(0, x[:, k], torch.ones(T, dtype=torch.int64))
        cs = torch.tensor(cold[l])
        m = torch.isin(x, cs)
        for t, k in m.nonzero().tolist():
            hits[(l, int(x[t, k]))].append((wi, t, k))
ntok = sum(famtok.values())
print(f"{len(W)} windows, {ntok} tokens; families: " + ", ".join(f"{f} {famtok[f] / ntok:.0%}" for f in fams))
res = {}
for band, ls in (("layers 0-19", range(0, 20)), ("layers 20-39", range(20, 40))):
    ls = list(ls)
    ncold = sum(len(cold[l]) for l in ls)
    tot = int(cnt[ls].sum())
    cp = sum(int(cnt[l, cold[l]].sum()) for l in ls)
    dormant = sum(int((cnt[l, cold[l]] == 0).sum()) for l in ls)
    dormant_all = int((cnt[ls] == 0).sum())
    fam_share = {f: sum(int(famc[f][l, cold[l]].sum()) for l in ls) / max(cp, 1) for f in fams}
    print(f"\n{band}: {ncold} cold experts ({ncold // len(ls)} per layer) take {cp / tot:.2%} of picks; "
          f"{dormant} never picked ({dormant_all} of all {len(ls) * E} experts)")
    print("  family share of cold picks vs of tokens: " +
          ", ".join(f"{f} {fam_share[f]:.0%}/{famtok[f] / ntok:.0%}" for f in fams))
    rk = sum(rankc[l, cold[l]].sum(0) for l in ls).double()
    print("  rank of cold picks (0 = strongest): " + " ".join(f"{i}:{v / rk.sum():.0%}" for i, v in enumerate(rk.tolist())))
# per cold expert: dominant family and its lift, top tokens by lift
prof = []
for l in range(L):
    for e in cold[l]:
        n = int(cnt[l, e])
        if n < 20:
            continue
        sh = {f: int(famc[f][l, e]) / n for f in fams}
        fd = max(sh, key=sh.get)
        lift = sh[fd] / (famtok[fd] / ntok)
        tc = collections.Counter(ids_of[wi][t] for wi, t, k in hits[(l, e)])
        top = sorted(tc.items(), key=lambda kv: -kv[1] * (kv[1] / n) / (tokfreq[kv[0]] / ntok))[:6]
        conc = sum(c for _, c in tc.most_common(5)) / n
        wi, t, k = hits[(l, e)][len(hits[(l, e)]) // 2]
        ctx = tok.decode(ids_of[wi][max(0, t - 10):t]) + " [[" + tok.decode([ids_of[wi][t]]) + "]] " + \
              tok.decode(ids_of[wi][t + 1:t + 6])
        prof.append(dict(layer=l, expert=e, picks=n, fam=fd, fam_share=sh[fd], fam_lift=lift, top5_conc=conc,
                         top=[(tok.decode([i]), c) for i, c in top], ctx=ctx.replace("\n", "\\n")[:160],
                         mean_rank=sum(k for _, _, k in hits[(l, e)]) / n))
json.dump(prof, open(OUT, "w"))
dom = collections.Counter(p["fam"] for p in prof if p["fam_share"] >= 0.6)
print(f"\ncold experts with >= 20 picks: {len(prof)}; dominated (>= 60% of picks) by one family: " +
      ", ".join(f"{f} {c}" for f, c in dom.most_common()) + f"; mixed: {len(prof) - sum(dom.values())}")
print("\nmost specific cold experts (family share x token concentration):")
for p in sorted(prof, key=lambda p: -(p["fam_share"] * p["top5_conc"]))[:28]:
    tops = " ".join(repr(s)[1:-1] + f"x{c}" for s, c in p["top"][:5])
    print(f"  L{p['layer']:2d} e{p['expert']:3d} n{p['picks']:4d} {p['fam']} {p['fam_share']:.0%} rank{p['mean_rank']:.1f} | {tops}")
    print(f"      ...{p['ctx']}")
