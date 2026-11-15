"""Expert-pinning bench: per window, the share of expert picks that go to rarely used experts (a rowmap's host experts
outside its starting cache), from routing files with "experts" [layers, T, 6] (dsv41's ref_logits.py output).
Feeds make_suites.py --cold (the cold-heavy group).
   python3 -I cold_rates.py ROUTING_DIR ROWMAP.json OUT.json [--layers 20-39]"""
import argparse
import glob
import json
import os

import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("routing")
ap.add_argument("rowmap")
ap.add_argument("out")
ap.add_argument("--layers", default="0-39")
a = ap.parse_args()
lo, hi = map(int, a.layers.split("-"))
R = json.load(open(a.rowmap))["layers"]
cold = {l: torch.tensor(sorted(set(R[str(l)]["host"]) - set(R[str(l)]["cache"]))) for l in range(lo, hi + 1)}
rates = {}
for f in sorted(glob.glob(os.path.join(a.routing, "*-*.safetensors"))):
    with safe_open(f, "pt") as fh:
        if "experts" not in fh.keys():
            continue
        ex = fh.get_tensor("experts").long()
    n = c = 0
    for l in range(lo, hi + 1):
        c += int(torch.isin(ex[l], cold[l]).sum())
        n += ex[l].numel()
    rates[os.path.basename(f)[:-len(".safetensors")]] = c / n
json.dump(rates, open(a.out, "w"), indent=0)
top = sorted(rates.items(), key=lambda kv: -kv[1])
print(f"{len(rates)} windows; highest: " + ", ".join(f"{k} {v:.1%}" for k, v in top[:12]))
print("lowest: " + ", ".join(f"{k} {v:.1%}" for k, v in top[-6:]))
