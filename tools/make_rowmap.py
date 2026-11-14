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
a = ap.parse_args()
L, E = 40, 384
c = torch.zeros(L, E, dtype=torch.float64); files = []
for d in a.dirs:
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(f, "pt") as h:
            if "experts" not in h.keys():
                continue
            e = h.get_tensor("experts").long()
        for l in range(L):
            c[l] += torch.bincount(e[l].flatten(), minlength=E).double()
        files.append(os.path.relpath(f, d))
jit = torch.rand(L, E, generator=torch.Generator().manual_seed(0), dtype=torch.float64) * 1e-3
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
