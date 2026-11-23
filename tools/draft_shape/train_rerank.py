"""A low-rank correction of DSpark's drafter head, fitted on acc3 dumps (C1 push): logits' = logits + E (U V^T n(H)),
n(H) = the drafter's final RMSNorm of its head hidden state, E = the head's rows (the drafter shares the target's head).
Applied to the candidates of each position: q's top-64 and p's top-64 (outside both, q keeps its shape, scaled by 1/Z).
Trained to maximise the chain value sum_j prod_i<=j a_i on the training prompts; scored per kind on the evaluation prompts
and lil (tok/s at the best fixed width), next to the baseline. GPU (run with the server down).
   python3 -I train_lora.py DIR MODEL_DIR [RANK] [EPOCHS]"""
import json
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.argv += [None] * 4
D, MD = sys.argv[1], sys.argv[2]
RANK = int(sys.argv[3] or 32)
EPOCHS = int(sys.argv[4] or 4)
dev = torch.device("cuda")
torch.manual_seed(0)
sys.argv = [sys.argv[0], D]
exec(open(__file__.replace("train_rerank.py", "fit_acc3.py")).read().split("def main():")[0])   # data, labels, accept()

idx = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]


def load(name):
    with safe_open(f"{MD}/{idx[name]}", "pt") as f:
        return f.get_tensor(name)


E = load("head.weight").to(dev)                      # [V, 5120] bf16
nw = load("mtp.2.norm.weight").to(dev).float()
print("head", tuple(E.shape), "norm", tuple(nw.shape))
Hd = HH.shape[-1]
Ht = torch.from_numpy(HH)                            # [N, K, Hd] fp16 (CPU)
# the target's final normed hidden at the previous step's bonus row (row = its accepted count), same request only
FIN = np.memmap(f"{D}/acc3.jsonl.fin", dtype=np.float16, mode="r")
NF = FIN.size // (8 * Hd)
FIN = FIN[:NF * 8 * Hd].reshape(NF, 8, Hd)
prev = np.zeros((N, Hd), dtype=np.float16)
have = 0
for s_ in range(1, min(N, NF)):
    a_, b_ = meta[s_ - 1], meta[s_]
    if b_["hl"] and a_["hl"] and b_["hl"] > a_["hl"] and b_["t"] - a_["t"] < 2 and a_["acc"] is not None:
        prev[s_] = FIN[s_ - 1, min(int(a_["acc"]), 7)]
        have += 1
PREV = torch.from_numpy(prev)
print(f"previous-step target state for {have} of {N} steps")
qi_t, pi_t = torch.from_numpy(qi), torch.from_numpy(pi)


def nH(ix):
    h = Ht[ix].to(dev).float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * nw
    t = PREV[ix].to(dev).float()[:, None, :].expand(-1, h.shape[1], -1)
    return torch.cat([h, t], -1)                       # [n, K, 2 Hd]: drafter state + the target's last state


class Lora(torch.nn.Module):
    """Carbon's DrafterRerank: a SwiGLU adapter on [drafter state; the target's last state] -> h~, scored against the
    candidates' full head rows (beta_j <h~, E_v>), plus a per-position temperature on the drafter's logits"""
    def __init__(self, width=1024):
        super().__init__()
        self.g = torch.nn.Linear(2 * Hd, width, bias=False, device=dev)
        self.u = torch.nn.Linear(2 * Hd, width, bias=False, device=dev)
        self.d = torch.nn.Linear(width, Hd, bias=False, device=dev)
        torch.nn.init.zeros_(self.d.weight)
        self.beta = torch.nn.Parameter(torch.ones(K, device=dev))
        self.alpha = torch.nn.Parameter(torch.zeros(K, device=dev))
        self.drop = torch.nn.Dropout(0.1)
        self.U = self.d.weight          # for the regularizer below
        self.V = self.g.weight

    def forward(self, x):
        x = self.drop(x)
        h = self.d(torch.nn.functional.silu(self.g(x)) * self.u(x))
        return h * self.beta[None, :, None]


def acc_dev(ix, model):
    """acceptance [n, K] with the correction, on the GPU"""
    hn = nH(ix)
    dl = model(hn)                                     # [n, K, Hd]
    qid, pid = qi_t[ix].to(dev), pi_t[ix].to(dev)
    eq = (E[qid].float() * dl[:, :, None]).sum(-1)
    if isinstance(model, torch.nn.Module):     # per-position temperature on the drafter's own log q
        eq = eq + model.alpha[None, :, None] * lq[ix].to(dev)
    ep = (E[pid].float() * dl[:, :, None]).sum(-1)
    qt, qtl = q_top[ix].to(dev), q_tail[ix].to(dev)
    w = qt * torch.exp(eq)
    exq = extra_q[ix].to(dev) * torch.exp(ep)          # p-top tokens outside q's top: also shifted
    Z = w.sum(-1) + (qtl - extra_q[ix].to(dev).sum(-1)).clamp(min=0) + exq.sum(-1)
    a = torch.minimum(pq[ix].to(dev), w / Z[..., None]).sum(-1) + torch.minimum(extra_p[ix].to(dev), exq / Z[..., None]).sum(-1)
    return a + r_out[ix].to(dev) * torch.clamp(1 / Z, max=1)


def chain_d(a, W=K):
    return torch.cumprod(a[:, :W], 1).sum(1)


LR = float(__import__("os").environ.get("LORA_LR", "2e-3"))


def val_value(model, va):
    if isinstance(model, torch.nn.Module):
        model.eval()
    with torch.no_grad():
        ix = torch.as_tensor(va)
        return torch.cat([chain_d(acc_dev(ix[s:s + 2048], model)) for s in range(0, len(ix), 2048)]).mean().item()


def train(tr):
    # validation: every 5th training request (whole requests, so no shared context between the halves)
    segs = segments(tr)
    va = [k for i, sg in enumerate(segs) if i % 5 == 4 for k in sg]
    frac = float(__import__("os").environ.get("FRAC", "1"))
    trs = [sg for i, sg in enumerate(segs) if i % 5 != 4]
    trs = trs[:max(1, int(round(len(trs) * frac)))]
    tr = [k for sg in trs for k in sg]
    print(f"training on {len(trs)} requests ({len(tr)} steps, fraction {frac})", flush=True)
    model = Lora()
    best = (val_value(model, va), {k: v.clone() for k, v in model.state_dict().items()}, -1)
    print(f"validation: {len(va)} steps of {len(segs) // 5} requests; zero correction {best[0]:.4f}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    tr = np.array(tr)
    bs = 512
    for ep in range(EPOCHS):
        np.random.shuffle(tr)
        tot, n = 0.0, 0
        for s in range(0, len(tr), bs):
            if s and s % (bs * 40) == 0:
                v = val_value(model, va)
                print(f"  step {s // bs}: validation {v:.4f}", flush=True)
            b = tr[s:s + bs]
            model.train()
            a = acc_dev(torch.as_tensor(b), model)
            loss = -chain_d(a).mean() + 1e-4 * (model.U.pow(2).sum() + model.V.pow(2).sum())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += -loss.item() * len(b)
            n += len(b)
        v = val_value(model, va)
        print(f"epoch {ep}: train chain value {tot / n:.4f}, validation {v:.4f}", flush=True)
        if v > best[0]:
            best = (v, {k: t.clone() for k, t in model.state_dict().items()}, ep)
    model.load_state_dict(best[1])
    print(f"kept epoch {best[2]} (validation {best[0]:.4f})", flush=True)
    return model


def tps_d(a):
    best = (0, 0, 0)
    for W in range(1, K + 1):
        tk = 1 + chain_d(a, W).mean().item()
        v = tk / periods[str(W)] * 1e3
        if v > best[0]:
            best = (v, W, tk)
    return best


def score(model, label):
    model.eval()
    print(f"== {label}")
    rows = []
    for key in sorted(groups):
        if key[0] == "train":
            continue
        ix = torch.as_tensor(groups[key])
        with torch.no_grad():
            am = torch.cat([acc_dev(ix[s:s + 2048], model) for s in range(0, len(ix), 2048)])
            ab = torch.cat([acc_dev(ix[s:s + 2048], lambda h: torch.zeros_like(h[..., :Hd])) for s in range(0, len(ix), 2048)])
        b, m = tps_d(ab), tps_d(am)
        rows.append((key, len(ix), b, m))
        print(f"  {key[0]:11s} {key[1]:5s} n={len(ix):6d}  pos1 a {ab[:, 0].mean():.3f}->{am[:, 0].mean():.3f}  "
              f"base {b[0]:.0f}@W{b[1]} ({b[2]:.2f})  model {m[0]:.0f}@W{m[1]} ({m[2]:.2f})  {100 * (m[0] / b[0] - 1):+.1f}%",
              flush=True)
    return rows


if __name__ == "__main__":
    tr = groups[("train", "train")]
    print(f"{len(tr)} training steps, rank {RANK}, {EPOCHS} epochs; hidden {Hd}")
    with torch.no_grad():
        ix = torch.as_tensor(tr[:2048])
        a0d = acc_dev(ix, lambda h: torch.zeros_like(h[..., :Hd]))
        print(f"check at zero correction: max |a - a0| {(a0d.cpu() - a0[ix]).abs().max():.2e}")
    model = train(tr)
    score(model, f"low-rank head correction, rank {RANK}")
    torch.save(model.state_dict(), f"{D}/rerank.pt")
