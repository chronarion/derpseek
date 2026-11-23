"""Carbon c4 idea 5, scored offline on the acc3t dump: mid-network row pruning at batch 1.
The acceptance tests' uniforms u_k are drawn before the step. After layer 20, a probe estimates p(d_k) from the state of
row k-1 (the row that predicts d_k); the first draft k* it predicts rejected (u_k >= p_hat(d_k) / q(d_k)) keeps rows
0..k* and drops rows k*+1..W from layers 20-39 (their MoE work skipped, as masked verify rows). Exact: whether draft j+1
is verified depends only on drafts <= j, their target states and u_{<=j}.
Outcome per step (real target p and drafter q at the drafted tokens): r = first real rejection (or W + 1). If r <= k*,
nothing is lost; if r > k*, the step ends after k* accepted drafts + the bonus from row k*.
Probe: logit lens (the target's final norm + head on the layer-20 state, full vocabulary), calibrated on the training
steps as p_hat = sigmoid(a * logit(p_lens) + b * log q + c); also the oracle (p_hat = p). Step time: T(W) from the
served periods, minus SAVE ms per dropped row (MoE of layers 20-39).
   python3 -I prune_sim.py DIR MODEL_DIR"""
import json
import sys

import numpy as np
import torch
from safetensors import safe_open

D, MD = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0], D]
exec(open(__file__.replace("prune_sim.py", "fit_acc3.py")).read().split("def main():")[0])
dev = torch.device("cuda")
idx = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]


def load(name):
    with safe_open(f"{MD}/{idx[name]}", "pt") as f:
        return f.get_tensor(name)


E = load("head.weight").to(dev)
nw = load("norm.weight").to(dev).float()
R = 8
L20 = np.memmap(f"{D}/acc3.jsonl.l20", dtype=np.float16, mode="r")
NL = L20.size // (R * 5120)
L20 = L20[:NL * R * 5120].reshape(NL, R, 5120)
n = min(N, NL)
d_tok = np.array([m["d"][:K] for m in meta[:n]])
qi_n = qi[:n]
# p and q at the drafted tokens (drafted token inside q's top-64; else the step is skipped)
pos_in = (qi_n == d_tok[:, :, None])
ok = pos_in.any(-1).all(-1)
j = pos_in.argmax(-1)
p_d = np.take_along_axis(pq[:n].numpy(), j[..., None], -1)[..., 0]
q_d = np.exp(np.take_along_axis(lq[:n].numpy(), j[..., None], -1)[..., 0])
print(f"{n} steps, {ok.mean():.3f} with every draft in q's top-64")

# probe: a ridge map from the layer-20 state to the target's final normed hidden (fitted on training rows), then the
# head: log p_hat(d_k) from row k-1
FINm = np.memmap(f"{D}/acc3.jsonl.fin", dtype=np.float16, mode="r").reshape(-1, R, 5120)
trn = np.array([k for k in groups[("train", "train")] if k < n])
rs = np.random.default_rng(1).choice(trn, size=min(len(trn), 20000), replace=False)
with torch.no_grad():
    X = torch.from_numpy(np.asarray(L20[np.sort(rs), :K])).to(dev).float().reshape(-1, 5120)
    Y = torch.from_numpy(np.asarray(FINm[np.sort(rs), :K])).to(dev).float().reshape(-1, 5120)
    mu = X.mean(0)
    Xc = torch.cat([X - mu, torch.ones(X.shape[0], 1, device=dev)], 1)
    G = Xc.T @ Xc
    G += 1e-2 * G.diagonal().mean() * torch.eye(G.shape[0], device=dev)
    Wr = torch.linalg.solve(G, Xc.T @ Y)
    del X, Y, Xc, G
    print("ridge probe fitted on", len(rs) * K, "training rows")
lens = np.zeros((n, K), dtype=np.float32)
with torch.no_grad():
    for s in range(0, n, 256):
        h = torch.from_numpy(np.asarray(L20[s:s + 256, :K])).to(dev).float()
        hn = torch.cat([h - mu, torch.ones(*h.shape[:-1], 1, device=dev)], -1) @ Wr
        lg = (hn.to(torch.bfloat16) @ E.T).float()
        lp = torch.log_softmax(lg, -1)
        dt = torch.from_numpy(d_tok[s:s + 256]).to(dev)
        lens[s:s + 256] = lp.gather(-1, dt[..., None])[..., 0].cpu().numpy()
print("lens log p(d_k) vs final log p(d_k), correlation by position:",
      [round(float(np.corrcoef(lens[ok, k], np.log(np.maximum(p_d[ok, k], 1e-9)))[0, 1]), 3) for k in range(K)])

tr = np.array([k for k in groups[("train", "train")] if k < n and ok[k]])


def logit(x):
    x = np.clip(x, 1e-7, 1 - 1e-7)
    return np.log(x / (1 - x))


# calibrate: predict p(d_k) from lens and q (least squares on logit scale; probability-space check below)
X = np.stack([logit(np.exp(lens[tr])).ravel(), np.log(q_d[tr]).ravel(), np.ones(tr.size * K)], 1)
y = logit(p_d[tr]).ravel()
coef = np.linalg.lstsq(X, y, rcond=None)[0]
print("calibration coef (lens logit, log q, 1):", np.round(coef, 3))


def p_hat(ix, oracle=False):
    if oracle:
        return p_d[ix]
    z = coef[0] * logit(np.exp(lens[ix])) + coef[1] * np.log(q_d[ix]) + coef[2]
    return 1 / (1 + np.exp(-z))


per = {int(k): v for k, v in periods.items()}
rng = np.random.default_rng(0)


def simulate(ix, W, mode, save, thr=1.0, draws=8):
    """mean tokens per step and mean step time at fixed width W; mode none / probe / oracle"""
    tok, tim = 0.0, 0.0
    ph = p_hat(ix, mode == "oracle") if mode != "none" else None
    for _ in range(draws):
        u = rng.random((len(ix), K))
        acc = u < np.minimum(1, p_d[ix] / q_d[ix])
        r = np.where((~acc[:, :W]).any(1), (~acc[:, :W]).argmax(1) + 1, W + 1)   # first rejected draft (W+1 = none)
        if mode == "none":
            kstar = np.full(len(ix), W + 1)
        else:
            pred_rej = u[:, :W] >= thr * np.minimum(1, ph[:, :W] / q_d[ix][:, :W])
            kstar = np.where(pred_rej.any(1), pred_rej.argmax(1) + 1, W + 1)
        # tokens: accepted drafts + 1; pruning caps the accepted drafts at k*
        accepted = np.minimum(r - 1, kstar)
        tok += (accepted + 1).mean()
        dropped = np.maximum(0, W - kstar)          # rows k*+1..W
        tim += (per[W] - save * dropped).mean()
    return tok / draws, tim / draws


for key in [("lil", "held"), ("lil", "tune"), ("chat", "held"), ("writing", "held"), ("code", "held"), ("cjk", "held")]:
    ix = np.array([k for k in groups.get(key, []) if k < n and ok[k]])
    if len(ix) == 0:
        continue
    out = []
    for mode in ("none", "probe", "oracle"):
        for save in ((0.08, 0.12) if mode != "none" else (0.0,)):
            best = max(((t / tm * 1e3, W, t) for W in range(1, 8) for t, tm in [simulate(ix, W, mode, save)]))
            out.append(f"{mode}{'' if mode == 'none' else f'@{save}'} {best[0]:.0f}@W{best[1]} ({best[2]:.2f})")
    print(f"{key[0]:8s} {key[1]:5s} n={len(ix):5d}: " + " | ".join(out), flush=True)
