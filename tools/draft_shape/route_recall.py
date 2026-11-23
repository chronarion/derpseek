"""Carbon's pre-router prefetch, first measurement: how well does the state leaving layer l predict layer l+1's routed
experts, in particular the non-static ones (the ones a prefetch would copy in from CPU memory)? Predictor: layer l+1's
gate on RMSNorm(state after layer l) x layer l+1's ffn_norm weight, sqrt(softplus) + bias, top-k (k = 6 and 12).
   python3 -I route_recall.py DIR MODEL_DIR ROWMAP"""
import json
import sys

import numpy as np
import torch
from safetensors import safe_open

D, MD, RM = sys.argv[1:4]
dev = torch.device("cuda")
meta = [json.loads(l) for l in open(f"{D}/acc3.jsonl")]
L, R, H = 40, 8, 5120
st = np.memmap(f"{D}/acc3.jsonl.st", dtype=np.float16, mode="r")
N = min(len(meta), st.size // (L * R * H))
st = st[:N * L * R * H].reshape(N, L, R, H)
tk = np.fromfile(f"{D}/acc3.jsonl.tk", dtype=np.int16)[:N * L * R * 6].reshape(N, L, R, 6)
print(f"{N} steps; expert id range {tk[tk >= 0].min()}..{tk.max()}; rows used: verify rows 0..7")
idx = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]


def load(n):
    with safe_open(f"{MD}/{idx[n]}", "pt") as f:
        return f.get_tensor(n)


static = {int(l): set(v["hot"]) for l, v in json.load(open(RM))["layers"].items()}
sel = np.arange(0, N, max(1, N // 1500))          # ~1,500 steps
tot = {k: [0, 0, 0, 0] for k in (6, 12, 24)}          # hits all, picks all, hits non-static, picks non-static
per_layer = []
for l in range(1, L - 1):
    g = load(f"layers.{l + 1}.ffn.gate.weight").to(dev).float()
    b = load(f"layers.{l + 1}.ffn.gate.bias").to(dev).float()
    nw = load(f"layers.{l + 1}.ffn_norm.weight").to(dev).float()
    x = torch.from_numpy(np.asarray(st[sel, l])).to(dev).float()          # state after layer l, [n, R, H]
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * nw
    logits = x @ g.T
    score = torch.sqrt(torch.nn.functional.softplus(logits)) + b
    actual = torch.from_numpy(tk[sel, l + 1].astype(np.int64)).to(dev)   # [n, R, 6]
    valid = actual >= 0
    s_l = static.get(l + 1, set())
    ns = torch.tensor([e not in s_l for e in range(384)], device=dev)
    rec_l = {}
    for k in (6, 12, 24):
        pred = score.topk(k, -1).indices                                    # [n, R, k]
        # a step's prediction is the union over its rows (the prefetch would fetch the union)
        pu = torch.zeros(len(sel), 384, dtype=torch.bool, device=dev)
        pu.scatter_(1, pred.reshape(len(sel), -1), True)
        au = torch.zeros(len(sel), 384, dtype=torch.bool, device=dev)
        au.scatter_(1, actual.clamp(min=0).reshape(len(sel), -1), valid.reshape(len(sel), -1))
        hit = (pu & au).sum().item(); pick = au.sum().item()
        hn = (pu & au & ns).sum().item(); pn = (au & ns).sum().item()
        tot[k][0] += hit; tot[k][1] += pick; tot[k][2] += hn; tot[k][3] += pn
        rec_l[k] = (hit / max(pick, 1), hn / max(pn, 1), pu.sum().item() / len(sel), (pu & ns).sum().item() / len(sel))
    per_layer.append((l + 1, rec_l))
for lay, r in per_layer[::4]:
    print(f"layer {lay:2d}: " + "  ".join(f"top{k}: recall all {v[0]:.2f} non-static {v[1]:.2f} (predicted {v[2]:.0f} experts, {v[3]:.0f} non-static)" for k, v in r.items() if k in (6, 12)))
for k, v in tot.items():
    print(f"overall top{k} per row (union per step): recall of all routed experts {v[0] / v[1]:.3f}, of non-static ones {v[2] / max(v[3], 1):.3f}")
