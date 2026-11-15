"""Expert-pinning bench, step 3: one report across approaches. For every approach (a name with its KLD output dir
and/or its serve_bench.py JSON) and every group of the suite: KL(base || approach) against a base run (normally
`exact` at the same batch; a second `exact` run gives the floor), its 99th percentile and top-1 agreement, prompt
reading tok/s, decode tok/s and the share of picks read from CPU memory.
   python3 -I report.py SUITE.json --base KLD_DIR NAME[:kld=DIR][:serve=JSON] [NAME...]
KLD needs torch and safetensors (run it in the vLLM image); serve-only reports run with the standard library."""
import argparse
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("suite")
ap.add_argument("approaches", nargs="+", help="name:kld=DIR:serve=JSON (either part optional)")
ap.add_argument("--base", help="KLD output dir of the base run (the `full` route dir)")
a = ap.parse_args()
group = {w["id"]: w["group"] for w in json.load(open(a.suite))}
groups = sorted(set(group.values()))


def kld(base, cand):
    """{group: (mean KL, p99 KL, top-1 agreement, rows)} over the windows both dirs hold, matched by position."""
    import torch
    from safetensors.torch import load_file
    acc = {}
    for f in sorted(glob.glob(os.path.join(cand, "*-*.safetensors"))):
        wid = os.path.basename(f)[:-len(".safetensors")]
        bf = os.path.join(base, os.path.basename(f))
        if wid not in group or not os.path.exists(bf):
            continue
        c, b = load_file(f), load_file(bf)
        pos_b = {int(p): i for i, p in enumerate(b["positions"])}
        keep = [i for i, p in enumerate(c["positions"]) if int(p) in pos_b]
        if not keep:
            continue
        lb = b["logits"][[pos_b[int(c["positions"][i])] for i in keep]].double()
        lc = c["logits"][keep].double()
        pb, pc = torch.log_softmax(lb, -1), torch.log_softmax(lc, -1)
        g = acc.setdefault(group[wid], ([], []))
        g[0].append((pb.exp() * (pb - pc)).sum(-1))
        g[1].append((lb.argmax(-1) == lc.argmax(-1)).double())
    out = {}
    for g, (ks, ags) in acc.items():
        k, ag = torch.cat(ks), torch.cat(ags)
        out[g] = (float(k.mean()), float(k.quantile(0.99)), float(ag.mean()), len(k))
    allk = [torch.cat(v[0]) for v in acc.values()]
    if allk:
        k = torch.cat(allk)
        out["all"] = (float(k.mean()), float(k.quantile(0.99)), float("nan"), len(k))
    return out


rows = []
for spec in a.approaches:
    name, *parts = spec.split(":")
    d = dict(p.split("=", 1) for p in parts)
    r = {"name": name}
    if "kld" in d and a.base:
        r["kld"] = kld(a.base, d["kld"])
    if "serve" in d and os.path.exists(d["serve"]):
        r["serve"] = json.load(open(d["serve"]))
    rows.append(r)


def table(title, cell, cols):
    print(f"\n{title}\n")
    print("| approach | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for r in rows:
        print(f"| {r['name']} | " + " | ".join(cell(r, g) for g in cols) + " |")


if any("kld" in r for r in rows):
    cols = groups + ["all"]
    table("KLD against the base run (mean; 99th percentile in brackets)",
          lambda r, g: (f"{r['kld'][g][0]:.4f} ({r['kld'][g][1]:.3f})" if g in r.get("kld", {}) else "-"), cols)
    table("Top-1 agreement with the base run",
          lambda r, g: (f"{r['kld'][g][2]:.3f}" if g in r.get("kld", {}) and g != "all" else "-"), groups)
if any("serve" in r for r in rows):
    table("Prompt reading, tokens per second",
          lambda r, g: (f"{r['serve'][g]['prefill_tok_s']:,.0f}" if g in r.get("serve", {}) and "prefill_tok_s" in r['serve'][g] else "-"), groups)
    table("Decode, tokens per second (all streams together)",
          lambda r, g: (f"{r['serve'][g]['decode_tok_s']:,.1f}" if g in r.get("serve", {}) and "decode_tok_s" in r['serve'][g] else "-"), groups)
    table("Share of non-static expert picks read from CPU memory during decode",
          lambda r, g: (f"{r['serve'][g]['misses']['miss_pct']:.1f}%" if (r.get("serve", {}).get(g) or {}).get("misses") else "-"), groups)
