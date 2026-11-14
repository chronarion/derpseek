"""Expert-drop masks for the KLD harness (ref_logits.py --prune): per evaluation window, rank the (layer, expert)
slots by pick count on the OTHER windows (leave-one-window-out, or leave-one-family-out) and penalize the coldest
ones so the router never selects them. Also reports the share of the window's own picks that the drop re-routes.
   python3 make_prune_masks.py ROUTES_DIR OUT_DIR"""
import glob, os, sys, torch
from safetensors import safe_open
src, out = sys.argv[1], sys.argv[2]
L, E, PEN = 40, 384, -1.0e4
counts, eval_ids = {}, []
for f in sorted(glob.glob(f"{src}/*.safetensors")):
    w = os.path.basename(f)[:-12]
    with safe_open(f, "pt") as h:
        e = h.get_tensor("experts").long()                      # [L, T, 6]
    counts[w] = torch.stack([torch.bincount(e[l].flatten(), minlength=E) for l in range(L)])
    if e.shape[1] == 2048:
        eval_ids.append(w)
fam = lambda w: w.split("-")[0]
total = sum(counts.values())
def mask_for(w, frac, mode):
    excl = [x for x in counts if (fam(x) == fam(w) if mode == "family" else x == w)]
    c = (total - sum(counts[x] for x in excl)).float()
    n = int(round(frac * L * E))
    m = torch.zeros(L, E, dtype=torch.bool)
    g = torch.Generator().manual_seed(0)
    jitter = torch.rand(L, E, generator=g) * 1e-3                # deterministic tie-break
    if mode == "uniform":
        per = n // L
        for l in range(L):
            m[l, torch.argsort(c[l] + jitter[l])[:per]] = True
    else:
        m.view(-1)[torch.argsort((c + jitter).view(-1))[:n]] = True
    return m
import sys as _s
ARMS = {"g125": (.125, "global"), "g175": (.175, "global"), "g225": (.225, "global"), "g275": (.275, "global"),
        "g345": (.345, "global"), "all": (1.0, "global"), "u225": (.225, "uniform"), "f225": (.225, "family")}
arms = [(n, *ARMS[n]) for n in (_s.argv[3].split(",") if len(_s.argv) > 3 else ["g125", "g175", "g225", "g275", "u225", "f225"])]
os.makedirs(out, exist_ok=True)
for name, frac, mode in arms:
    masks = torch.stack([mask_for(w, frac, mode) for w in eval_ids])          # [W, L, E]
    rerouted = []
    for k, w in enumerate(eval_ids):
        c = counts[w]
        rerouted.append(float((c * masks[k]).sum() / c.sum()) * 100)
    per_layer = masks.sum(2).float()
    torch.save({"pen": masks.float() * PEN, "windows": eval_ids}, f"{out}/{name}.pt")
    print(f"{name}: drop {frac*100:.1f}% of slots ({mode}); per-layer dropped min/mean/max "
          f"{per_layer.min():.0f}/{per_layer.mean():.0f}/{per_layer.max():.0f}; picks re-routed per window: mean "
          f"{sum(rerouted)/len(rerouted):.2f}% (min {min(rerouted):.2f}, max {max(rerouted):.2f}: "
          f"{eval_ids[rerouted.index(max(rerouted))]})")
