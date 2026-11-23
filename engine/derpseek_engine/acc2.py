"""Measurement only (C1 push, 2026-11-22): exact acceptance of drafting alternatives at batch 1, from the target's
processed rows. For every verified draft position j of a sampled request (one request per step), one JSON line per
step with, per position:
  a      sum_x min(p(x), q(x)): the acceptance probability of today's draft distribution q (no sampling noise)
  a2     with a second i.i.d. draft from q at that position (multi-draft rejection: a + (1 - a) * sum min(p_res, q),
         p_res = max(p - q, 0) / (1 - a)): what a sibling row would add there
  L, c   the longest suffix match of (history + drafts before j) against the history, and the token that followed it
  dc     whether DSpark's own draft at j equals c
  pc, qc p(c) and q(c)
  mix    for alpha in ALPHAS: sum min(p, alpha * onehot(c) + (1 - alpha) * q) (a copy spliced in as a mixture draft)
  ent    entropy of q (nats), pmax max p
plus the step's accepted count. The chain expected value for any policy is computed offline (tools in the scratch
dir). Rows past position j are conditioned on DSpark's draft at j, so a copy that differs from it is exact only at
its own position. Python per step: run with DSV41_WHOLE_STEP=0 and full-width verify (no dynamic width, no suffix).
   DERPSEEK_ACC2=/path/to/file.jsonl"""
import gc
import json
import os
import sys

import numpy as np

PATH = os.environ.get("DERPSEEK_ACC2")
ALPHAS = (0.3, 0.5, 0.7, 0.9, 1.0)
MAXL = 64          # suffix matches are capped at this length
STATE = {"runner": None, "f": None, "n": 0, "diag": 0}
_LOG = lambda m: sys.stderr.write(f"DERPSEEK_ACC2 {m}\n")


def _runner():
    if STATE["runner"] is None:
        for o in gc.get_objects():
            try:
                if hasattr(o, "req_states") and hasattr(o.req_states, "all_token_ids"):
                    STATE["runner"] = o
                    break
            except Exception:  # noqa: BLE001
                pass
    return STATE["runner"]


def _match(h):
    """longest L <= MAXL with h[-L:] == h[i-L:i] for some i < len(h); returns (L, h[i]) for the latest such i"""
    n = len(h)
    if n < 2:
        return 0, -1
    best_l, best_c = 0, -1
    last = h[-1]
    cand = np.nonzero(h[:-1] == last)[0]   # positions i-1 where the last token recurs
    for e in cand[::-1]:                   # latest first
        L = 1
        while L < MAXL and e - L >= 0 and h[e - L] == h[n - 1 - L]:
            L += 1
        if L > best_l:
            best_l, best_c = L, int(h[e + 1])
            if L >= MAXL:
                break
    return best_l, best_c


def _record(self, processed, draft_logits, draft_sampled, expanded_idx_mapping, expanded_local_pos, out):
    import torch
    if draft_logits is None:
        return
    lp = expanded_local_pos.long()
    st = self.sampler.sampling_states
    K = draft_logits.shape[1]
    sel = ((lp >= 1) & (lp <= K)).nonzero().squeeze(1)
    if sel.numel() == 0:
        return
    req = expanded_idx_mapping[sel].long()
    if int(req.unique().numel()) != 1:
        return                                   # batch 1 only
    slot = int(req[0])
    T = float(st.temperature.gpu[slot])
    if T <= 0:
        return
    p = torch.softmax(processed[sel - 1].float(), -1)
    q = torch.softmax(draft_logits[slot, lp[sel] - 1].float() / T, -1)
    a = torch.minimum(p, q).sum(-1)
    pres = torch.clamp(p - q, min=0)
    pres = pres / pres.sum(-1, keepdim=True).clamp(min=1e-30)
    a2 = a + (1 - a) * torch.minimum(pres, q).sum(-1)
    ent = -(q * torch.log(q.clamp(min=1e-30))).sum(-1)
    pmax = p.max(-1).values
    # ceiling of any reweighting of q over its top-k candidates (the confidence-head tilt is one): q' = p on top-k(q),
    # the rest of the mass on q's tail in q's proportions; acceptance sum min(p, q')
    ceil = {}
    for kk in (1, 8, 32, 128):
        tv, ti = q.topk(kk, -1)
        ptop = p.gather(-1, ti)
        mask = torch.zeros_like(q, dtype=torch.bool).scatter_(-1, ti, True)
        tail_q = q.masked_fill(mask, 0)
        scale = ((1 - ptop.sum(-1)) / tail_q.sum(-1).clamp(min=1e-30)).clamp(min=0)
        ceil[kk] = ptop.sum(-1) + torch.minimum(p.masked_fill(mask, 0), tail_q * scale[:, None]).sum(-1)
    pos = lp[sel].tolist()
    drafts = draft_sampled[sel].tolist()                  # [num_logits]: each row's input token (row j: draft j)
    a_rows = (lp == 0).nonzero().squeeze(1)
    anchor = int(draft_sampled[a_rows[0]]) if a_rows.numel() else -1

    r = _runner()
    hist = None
    if r is not None:
        rs = r.req_states
        n = int(rs.total_len.gpu[slot])
        hist = rs.all_token_ids.gpu[slot, :n].cpu().numpy().astype(np.int64)
        if anchor >= 0 and (len(hist) == 0 or hist[-1] != anchor):
            STATE["appended"] = STATE.get("appended", 0) + 1
            hist = np.append(hist, anchor)
    rows = []
    for k, j in enumerate(pos):
        row = {"j": j, "a": round(float(a[k]), 5), "a2": round(float(a2[k]), 5), "ent": round(float(ent[k]), 3),
               "pmax": round(float(pmax[k]), 4), "ceil": [round(float(ceil[kk][k]), 5) for kk in (1, 8, 32, 128)]}
        if hist is not None:
            ext = np.concatenate([hist, np.asarray(drafts[:j - 1], dtype=np.int64)]) if j > 1 else hist
            L, c = _match(ext)
            row["L"] = L
            if c >= 0:
                pk, qk = p[k], q[k]
                pc, qc = float(pk[c]), float(qk[c])
                row.update(c=c, dc=int(drafts[j - 1] == c) if j - 1 < len(drafts) else -1,
                           pc=round(pc, 5), qc=round(qc, 5))
                # mixture: x != c gets (1 - alpha) q(x): sum min(p, (1 - alpha) q) is not linear, so compute it
                mix = []
                for al in ALPHAS:
                    m = torch.minimum(pk, (1 - al) * qk).sum() - min(pc, (1 - al) * qc) + min(pc, al + (1 - al) * qc)
                    mix.append(round(float(m), 5))
                row["mix"] = mix
        rows.append(row)
    acc = int(out[2][0]) - 1 if isinstance(out, tuple) and len(out) > 2 else None   # num_sampled - 1
    rec = {"s": STATE["n"], "t": round(__import__("time").time(), 3), "T": T, "top_p": float(st.top_p.gpu[slot]), "hl": None if hist is None else len(hist), "app": STATE.get("appended", 0),
           "acc": acc, "rows": rows}
    if STATE["diag"] < 3 and hist is not None:
        STATE["diag"] += 1
        _LOG(f"diag: hist tail {hist[-6:].tolist()}, anchor {anchor}, drafts {drafts[:8]}, pos {pos}, acc {acc}, out types "
             f"{[type(x).__name__ for x in out] if isinstance(out, tuple) else type(out).__name__}")
    STATE["f"].write(json.dumps(rec) + "\n")
    STATE["n"] += 1
    STATE["f"].flush()


def install():
    import importlib
    mod = importlib.import_module("vllm.v1.worker.gpu.spec_decode.rejection_sampler")
    cls = mod.RejectionSampler
    orig = cls._verify
    STATE["f"] = open(PATH, "a")

    def _verify(self, logits, draft_logits, draft_sampled, pos, cu_num_logits, idx_mapping, idx_mapping_np,
                expanded_idx_mapping, expanded_local_pos, *args, **kwargs):
        out = orig(self, logits, draft_logits, draft_sampled, pos, cu_num_logits, idx_mapping, idx_mapping_np,
                   expanded_idx_mapping, expanded_local_pos, *args, **kwargs)
        try:
            _record(self, out[0], draft_logits, draft_sampled, expanded_idx_mapping, expanded_local_pos, out)
        except Exception as e:  # noqa: BLE001  (measurement only)
            _LOG(f"record failed: {e!r}")
        return out

    cls._verify = _verify
    _LOG(f"installed -> {PATH}")
