"""Per-layer static counts for an existing exact rowmap (Fable 11-23: static rows by marginal miss reduction).

Keeps the band's total static rows (default layers 20-39) and each layer's existing static set as its core, but moves
static slots between the band's layers: one ranking across layers by a per-layer pick share over calibration routing
traces (key = the lowest share over the trace families, as make_rowmap's maxmin, or the total share), each layer between
LO and HI static rows. A layer that gains takes its best-ranked non-static experts; a layer that loses drops its
worst-ranked static ones. Cache rows (count and initial content) and the 480-row width stay as they are.
   python3 flex_rowmap.py IN.json OUT.json TRACE_DIR [TRACE_DIR ...] [--band 20-39] [--lo 150] [--hi 200] [--rank maxmin]"""
import argparse, glob, json, os
import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("dirs", nargs="+")
ap.add_argument("--band", default="20-39"); ap.add_argument("--lo", type=int, default=150)
ap.add_argument("--hi", type=int, default=200); ap.add_argument("--rank", default="maxmin", choices=("maxmin", "count"))
a = ap.parse_args()
L, E = 40, 384
c = torch.zeros(L, E, dtype=torch.float64); fam = {}; n = 0
for d in a.dirs:
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(f, "pt") as h:
            if "experts" not in h.keys():
                continue
            e = h.get_tensor("experts").long()
        fc = fam.setdefault(os.path.basename(f).split("-")[0], torch.zeros(L, E, dtype=torch.float64))
        for l in range(L):
            b = torch.bincount(e[l].flatten(), minlength=E).double()
            c[l] += b; fc[l] += b
        n += 1
if a.rank == "maxmin":
    key = torch.stack([fc / fc.sum(1, keepdim=True).clamp(min=1) for fc in fam.values()]).min(0).values
else:
    key = c / c.sum(1, keepdim=True)
r = json.load(open(a.inp))
Ls = r["layers"]
lo_b, hi_b = (int(x) for x in a.band.split("-"))
band = list(range(lo_b, hi_b + 1))
budget = sum(len(Ls[str(l)]["hot"]) for l in band)
# per layer: static set ordered best-first by key (core), then non-static by key
order = {}
for l in band:
    hot = Ls[str(l)]["hot"]
    rest = [x for x in range(E) if x not in set(hot)]
    order[l] = sorted(hot, key=lambda x: -float(key[l][x])) + sorted(rest, key=lambda x: -float(key[l][x]))
take = {l: a.lo for l in band}
cand = sorted(((float(key[l][order[l][k]]), l) for l in band for k in range(a.lo, a.hi)), key=lambda x: -x[0])
left = budget - a.lo * len(band)
for v, l in cand:
    if left <= 0:
        break
    take[l] += 1; left -= 1
assert left == 0
tot = c.sum()
before = sum(float(c[l][Ls[str(l)]["hot"]].sum()) for l in band)
for l in band:
    v = Ls[str(l)]
    hot = sorted(order[l][:take[l]])
    host = [x for x in range(E) if x not in set(hot)]
    host_sorted = sorted(host, key=lambda x: -float(c[l][x]))
    cache = [x for x in v["cache"] if x in set(host)]
    cache += [x for x in host_sorted if x not in set(cache)][:len(v["cache"]) - len(cache)]
    v["hot"], v["cache"], v["host"] = hot, cache, [x for x in host_sorted]
after = sum(float(c[l][Ls[str(l)]["hot"]].sum()) for l in band)
r.setdefault("meta", {})["flex"] = {"band": a.band, "lo": a.lo, "hi": a.hi, "rank": a.rank, "trace_files": n,
                                    "families": sorted(fam), "static_per_layer": {str(l): take[l] for l in band},
                                    "in_sample_band_static_picks_pct": [round(before / float(c[band].sum()) * 100, 3),
                                                                        round(after / float(c[band].sum()) * 100, 3)]}
json.dump(r, open(a.out, "w"))
print(json.dumps(r["meta"]["flex"]))
