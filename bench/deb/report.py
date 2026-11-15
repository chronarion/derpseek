"""Degenerate expert bench (DEB), step 3: KL(base || candidate) per window group, from decode_kld.py output dirs (the
`full` route of each run: logits [R, V], target ids and positions per window). The base is an `exact` run on the same
windows at the same batch; a second `exact` run as candidate gives the floor (the engine is not bit-reproducible run
to run). Groups: ctrl (wiki-0, code-0, chat-0, zh-0), rand, shuf, switch.
   python3 -I report.py BASE_DIR LABEL=CAND_DIR [LABEL=CAND_DIR ...]"""
import glob
import os
import sys

import torch
from safetensors.torch import load_file


def group(name: str) -> str:
    stem = name.split("-")[0]
    return stem if stem in ("rand", "shuf", "switch") else "ctrl"


def compare(base: str, cand: str):
    """{group: (kl rows, top-1 agreement rows)} over the windows both dirs hold, matched by position."""
    out = {}
    for f in sorted(glob.glob(os.path.join(cand, "*-*.safetensors"))):
        bf = os.path.join(base, os.path.basename(f))
        if not os.path.exists(bf):
            continue
        c, b = load_file(f), load_file(bf)
        pos_b = {int(p): i for i, p in enumerate(b["positions"])}
        keep = [i for i, p in enumerate(c["positions"]) if int(p) in pos_b]
        if not keep:
            continue
        lb = b["logits"][[pos_b[int(c["positions"][i])] for i in keep]].double()
        lc = c["logits"][keep].double()
        pb, pc = torch.log_softmax(lb, -1), torch.log_softmax(lc, -1)
        kl = (pb.exp() * (pb - pc)).sum(-1)
        agree = (lb.argmax(-1) == lc.argmax(-1)).double()
        g = out.setdefault(group(os.path.basename(f)[:-len(".safetensors")]), ([], []))
        g[0].append(kl)
        g[1].append(agree)
    return {k: (torch.cat(v[0]), torch.cat(v[1])) for k, v in out.items()}


base = sys.argv[1]
print(f"KL(base || candidate), base {base}")
print(f"{'candidate':<28} {'group':<7} {'rows':>6} {'KLD mean':>9} {'p99':>8} {'top-1':>7}")
for arg in sys.argv[2:]:
    label, _, cand = arg.partition("=")
    res = compare(base, cand)
    allk = []
    for g in ("ctrl", "rand", "shuf", "switch"):
        if g not in res:
            continue
        kl, ag = res[g]
        allk.append(kl)
        print(f"{label:<28} {g:<7} {len(kl):>6} {kl.mean():>9.5f} {kl.quantile(0.99):>8.4f} {ag.mean():>7.4f}")
    if allk:
        k = torch.cat(allk)
        print(f"{label:<28} {'all':<7} {len(k):>6} {k.mean():>9.5f} {k.quantile(0.99):>8.4f}")
