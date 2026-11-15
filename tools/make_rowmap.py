"""Derpseek rowmap: which experts stay resident on the GB300 (MegaMoE rows) and which are left out (never selected).
Ranks every (layer, expert) slot by pick count over calibration routing traces (safetensors with "experts"
[layers, T, 6], as tools/kld/ref_logits.py writes) and keeps the N most-picked slots, globally across layers, with a
floor per layer. Output: {"layers": {i: {"hot": [...], "cold": [...], "warm": []}}, "meta": {...}} (dsv41 format).
   python3 make_rowmap.py OUT.json N TRACE_DIR [TRACE_DIR ...] [--min-per-layer 256]"""
import argparse, glob, json, os
import torch
from safetensors import safe_open
ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("n", type=int); ap.add_argument("dirs", nargs="+")
ap.add_argument("--min-per-layer", type=int, default=256)
ap.add_argument("--exact", default="", help="S,C: Derpseek exact mode per layer: the S most-picked experts static (hot), the next C start in the HBM cache rows, every non-static expert keeps a host row (n is ignored)")
ap.add_argument("--exact-split", default="",
                help="per layer band 'A-B:S,C;...' (e.g. '0-19:224,96;20-39:160,96'): exact mode with more static rows "
                     "where prompts run on every token (layers 0-19) and fewer where only decode runs; C must match "
                     "across bands (every layer keeps 384 + C rows)")
ap.add_argument("--static-rank", default="count", choices=("count", "maxmin"),
                help="exact mode: rank the static experts by total picks (count) or by their lowest pick share over the "
                     "calibration families (maxmin: experts hot in every family; a family = the file name before '-')")
a = ap.parse_args()
L, E = 40, 384
c = torch.zeros(L, E, dtype=torch.float64); files = []
fam = {}   # family -> [L, E] picks
for d in a.dirs:
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(f, "pt") as h:
            if "experts" not in h.keys():
                continue
            e = h.get_tensor("experts").long()
        f_name = os.path.basename(f).split("-")[0]
        fc = fam.setdefault(f_name, torch.zeros(L, E, dtype=torch.float64))
        for l in range(L):
            b = torch.bincount(e[l].flatten(), minlength=E).double()
            c[l] += b
            fc[l] += b
        files.append(os.path.relpath(f, d))
jit = torch.rand(L, E, generator=torch.Generator().manual_seed(0), dtype=torch.float64) * 1e-3
band = {}
if a.exact_split:
    for part in a.exact_split.split(";"):
        rng, sc = part.split(":")
        lo, _, hi = rng.partition("-")
        bs, bc = (int(x) for x in sc.split(","))
        for l in range(int(lo), int(hi or lo) + 1):
            band[l] = (bs, bc)
    if sorted(band) != list(range(L)) or len({c for _, c in band.values()}) != 1:
        raise SystemExit("--exact-split must cover layers 0-39 once, with one cache size C")
    a.exact = "%d,%d" % band[0]
if a.exact:
    S, C = (int(x) for x in a.exact.split(","))
    layers, tot = {}, c.sum()
    hot_picks = cache_picks = 0.0
    if a.static_rank == "maxmin":
        shares = torch.stack([fc / fc.sum(dim=1, keepdim=True) for fc in fam.values()])   # [F, L, E]
        key = shares.min(dim=0).values                                                   # lowest family share
    for l in range(L):
        S, C = band.get(l, (S, C))
        if a.static_rank == "maxmin":
            hot_set = torch.argsort(-(key[l] + jit[l] * 1e-9)).tolist()[:S]
            rest = [x for x in torch.argsort(-(c[l] + jit[l])).tolist() if x not in set(hot_set)]
            order = hot_set + rest
        else:
            order = torch.argsort(-(c[l] + jit[l])).tolist()
        hot, cache = sorted(order[:S]), order[S:S + C]
        host = order[S:]                                 # most-picked first; the cache experts are its head
        layers[str(l)] = {"hot": hot, "cache": cache, "host": host, "cold": [], "warm": []}
        hot_picks += float(c[l][hot].sum()); cache_picks += float(c[l][cache].sum())
    meta = {"mode": "exact", "static": S, "cache": C, "host": E - S, "rows": S + C + E - S, "calibration_files": len(files),
            "split": a.exact_split or None,
            "static_rank": a.static_rank, "families": sorted(fam),
            "in_sample_picks_static_pct": round(hot_picks / float(tot) * 100, 3),
            "in_sample_picks_initial_cache_pct": round(cache_picks / float(tot) * 100, 3)}
    json.dump({"layers": layers, "meta": meta}, open(a.out, "w"))
    print(json.dumps(meta)); raise SystemExit
keep = torch.zeros(L, E, dtype=torch.bool)
for l in range(L):                                   # the floor first
    keep[l, torch.argsort(-(c[l] + jit[l]))[:a.min_per_layer]] = True
rest = a.n - int(keep.sum())
assert rest >= 0, f"N={a.n} is below the floor {L * a.min_per_layer}"
score = (c + jit).masked_fill(keep, -1.0).view(-1)
keep.view(-1)[torch.argsort(-score)[:rest]] = True
tot = c.sum()
layers = {str(l): {"hot": torch.nonzero(keep[l]).flatten().tolist(), "cold": torch.nonzero(~keep[l]).flatten().tolist(),
                   "warm": []} for l in range(L)}
per = keep.sum(1)
meta = {"resident": int(keep.sum()), "left_out": int((~keep).sum()), "min_per_layer": a.min_per_layer,
        "calibration_files": len(files), "calibration_picks": int(tot),
        "in_sample_picks_left_out_pct": round(float(c[~keep].sum() / tot * 100), 3),
        "resident_per_layer_min_max": [int(per.min()), int(per.max())]}
json.dump({"layers": layers, "meta": meta}, open(a.out, "w"))
print(json.dumps(meta))
