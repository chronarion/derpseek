"""Measurement only (C1 push, 2026-11-22): per verified draft position at batch 1, the drafter's and the target's
distributions in sparse form plus the drafter's hidden state, to fit and score drafter corrections offline
(confidence tilts, online adaptation, a calibration head). Run with DSV41_WHOLE_STEP=0 and full-width verify (7 drafts,
no dynamic width, no suffix drafter), like acc2.
Per step with 7 verified drafts, one JSON line in PATH (t, T, top_p, history length, accepted count, the anchor and the
7 draft tokens, a_j = sum min(p, q)) and one fixed-size record in PATH.bin: for each position j = 1..7,
  qi[64] int32   the drafter's top-64 token ids          lq[64] f32  log q at them
  pq[64] f32     p at the drafter's top-64               pi[64] int32 the target's top-64 ids
  lp[64] f32     log p at them                           qp[64] f32   q at the target's top-64
and in PATH.h16 the drafter's head hidden state for the 7 positions (fp16 [7, H]), captured by a copy wrapped around
the DSpark model's compute_draft_logits (capture-safe: the copy is part of the drafter's graph).
   DERPSEEK_ACC3=/path/to/file.jsonl"""
import json
import os
import sys
import time

import numpy as np

PATH = os.environ.get("DERPSEEK_ACC3")
K = 7
TOP = 64
STATE = {"f": None, "fb": None, "fh": None, "n": 0, "H": None, "runner": None, "L20": None, "FIN": None}
# DERPSEEK_ACC3_TARGET=1: also the target's state after layer 20 (the hyper-connection mean, as DSpark reads layers 37-39)
# and its final normed hidden state, for every verify row (fp16 [8, H] each, in PATH.l20 / PATH.fin)
TARGET = __import__("os").environ.get("DERPSEEK_ACC3_TARGET", "0") == "1"
PROBE_LAYER = int(__import__("os").environ.get("DERPSEEK_ACC3_LAYER", "20"))
# DERPSEEK_ACC3_ROUTES=1: every layer's state after it (hyper-connection mean, fp16 [40, 8, H] in PATH.st) and every
# layer's routed expert ids (int16 [40, 8, 6] in PATH.tk), for the next-layer routing predictor (Carbon's pre-router)
ROUTES = __import__("os").environ.get("DERPSEEK_ACC3_ROUTES", "0") == "1"
ROWS = 8
_LOG = lambda m: sys.stderr.write(f"DERPSEEK_ACC3 {m}\n")


def _record(self, processed, draft_logits, draft_sampled, expanded_idx_mapping, expanded_local_pos, out):
    import torch
    if draft_logits is None:
        return
    lp_ = expanded_local_pos.long()
    st = self.sampler.sampling_states
    sel = ((lp_ >= 1) & (lp_ <= K)).nonzero().squeeze(1)
    if sel.numel() != K:
        return
    req = expanded_idx_mapping[sel].long()
    if int(req.unique().numel()) != 1:
        return
    slot = int(req[0])
    T = float(st.temperature.gpu[slot])
    if T <= 0:
        return
    pos = lp_[sel]
    if pos.tolist() != list(range(1, K + 1)):
        return
    logp = torch.log_softmax(processed[sel - 1].float(), -1)
    logq = torch.log_softmax(draft_logits[slot, pos - 1].float() / T, -1)
    p, q = logp.exp(), logq.exp()
    a = torch.minimum(p, q).sum(-1)
    lqv, qi = logq.topk(TOP, -1)
    lpv, pi = logp.topk(TOP, -1)
    rec = torch.stack([qi.to(torch.int32).view(torch.float32),
                       lqv, p.gather(-1, qi), pi.to(torch.int32).view(torch.float32), lpv, q.gather(-1, pi)], 1)
    STATE["fb"].write(rec.contiguous().cpu().numpy().tobytes())            # [7, 6, 64] f32 (ids bit-cast)
    if ROUTES and STATE.get("ST") is not None:
        STATE["fs"].write(STATE["ST"][:, :ROWS].to(torch.float16).cpu().numpy().tobytes())
        STATE["ft"].write(STATE["TK"][:, :ROWS].to(torch.int16).cpu().numpy().tobytes())
    if TARGET and STATE["L20"] is not None:
        STATE["fl"].write(STATE["L20"][:ROWS].to(torch.float16).cpu().numpy().tobytes())
        STATE["ff"].write(STATE["FIN"][:ROWS].to(torch.float16).cpu().numpy().tobytes())
    H = STATE["H"]
    if H is not None:
        STATE["fh"].write(H[:K].to(torch.float16).cpu().numpy().tobytes())
    drafts = draft_sampled[sel].tolist()
    a_rows = (lp_ == 0).nonzero().squeeze(1)
    anchor = int(draft_sampled[a_rows[0]]) if a_rows.numel() else -1
    acc = int(out[2][0]) - 1 if isinstance(out, tuple) and len(out) > 2 else None
    hl = None
    r = STATE["runner"]
    if r is not None:
        hl = int(r.req_states.total_len.gpu[slot])
    meta = {"s": STATE["n"], "t": round(time.time(), 3), "T": T, "top_p": float(st.top_p.gpu[slot]), "hl": hl,
            "acc": acc, "anchor": anchor, "d": drafts, "a": [round(float(x), 5) for x in a]}
    STATE["f"].write(json.dumps(meta) + "\n")
    STATE["n"] += 1
    if STATE["n"] % 64 == 0:
        for f in ("f", "fb", "fh") + (("fl", "ff") if TARGET else ()) + (("fs", "ft") if ROUTES else ()):
            STATE[f].flush()


def install():
    import importlib
    import torch
    mod = importlib.import_module("vllm.v1.worker.gpu.spec_decode.rejection_sampler")
    cls = mod.RejectionSampler
    orig = cls._verify
    STATE["f"] = open(PATH, "a")
    STATE["fb"] = open(PATH + ".bin", "ab")
    STATE["fh"] = open(PATH + ".h16", "ab")
    if ROUTES:
        STATE["fs"] = open(PATH + ".st", "ab")
        STATE["ft"] = open(PATH + ".tk", "ab")
        _install_routes()
    if TARGET:
        STATE["fl"] = open(PATH + ".l20", "ab")
        STATE["ff"] = open(PATH + ".fin", "ab")
        _install_target()

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

    smod = importlib.import_module("vllm.v1.worker.gpu.spec_decode.dspark.speculator")
    scls = smod.DSparkSpeculator
    load = scls.load_draft_model

    def load_draft_model(self, *args, **kwargs):
        model = load(self, *args, **kwargs)
        cdl = model.compute_draft_logits
        w = model.lm_head.weight
        STATE["H"] = torch.zeros(int(self.max_num_reqs) * int(self.num_speculative_steps), w.shape[1], dtype=w.dtype,
                                 device=w.device)

        def compute_draft_logits(h):
            n = min(h.shape[0], STATE["H"].shape[0])
            STATE["H"][:n].copy_(h[:n])
            if not STATE.get("seen"):
                STATE["seen"] = 1
                _LOG(f"first hidden capture: {tuple(h.shape)} {h.dtype}")
            return cdl(h)

        model.compute_draft_logits = compute_draft_logits
        _LOG("drafter hidden capture on")
        return model

    scls.load_draft_model = load_draft_model
    _LOG(f"installed -> {PATH}")


def _install_target():
    """capture-safe copies of the target's layer-PROBE_LAYER state and final normed hidden into fixed buffers"""
    import importlib
    import torch
    mm = importlib.import_module("vllm.models.deepseek_v41.nvidia.model")
    layer_cls = mm.DeepseekV4DecoderLayer
    fwd = layer_cls.forward

    def forward(self, *args, capture_previous_aux=False, **kwargs):
        mine = getattr(self, "_acc3_probe", False)
        out = fwd(self, *args, capture_previous_aux=capture_previous_aux or mine, **kwargs)
        if not mine:
            return out
        aux = out[-1]
        if aux is not None:
            buf = STATE["L20"]
            n = min(aux.shape[0], buf.shape[0])
            buf[:n].copy_(aux[:n].reshape(n, -1)[:, :buf.shape[1]])
            if not STATE.get("seen_l"):
                STATE["seen_l"] = 1
                _LOG(f"layer {PROBE_LAYER} capture: {tuple(aux.shape)} {aux.dtype}")
        return out if capture_previous_aux else (*out[:-1], None)

    layer_cls.forward = forward
    r = STATE["runner"]
    load = r.load_model

    def load_model(*a, **k):
        out = load(*a, **k)
        model = r.model
        inner = None
        for m in model.modules():
            if type(m).__name__ == "DeepseekV4Model":
                inner = m
                break
        H = inner.config.hidden_size
        dev = next(inner.parameters()).device
        STATE["L20"] = torch.zeros(64, H, dtype=torch.bfloat16, device=dev)
        STATE["FIN"] = torch.zeros(64, H, dtype=torch.bfloat16, device=dev)
        inner.layers[PROBE_LAYER]._acc3_probe = True

        def norm_hook(mod, inp, outp):
            o = outp[0] if isinstance(outp, tuple) else outp
            n = min(o.shape[0], 64)
            STATE["FIN"][:n].copy_(o[:n])
            if not STATE.get("seen_f"):
                STATE["seen_f"] = 1
                _LOG(f"final hidden capture: {tuple(o.shape)} {o.dtype}")
        inner.norm.register_forward_hook(norm_hook)
        _LOG(f"target capture on: layer {PROBE_LAYER} (state after layers 0..{PROBE_LAYER - 1}) and the final norm")
        return out

    r.load_model = load_model


def _install_routes():
    """capture-safe copies: each decoder layer's input aux (= the state after the previous layer) and each MoE layer's
    routed expert ids from the boundary router"""
    import importlib
    import sys as _s
    import torch
    mm = importlib.import_module("vllm.models.deepseek_v41.nvidia.model")
    layer_cls = mm.DeepseekV4DecoderLayer
    fwd = layer_cls.forward

    def forward(self, *args, capture_previous_aux=False, **kwargs):
        li = getattr(self, "_acc3_idx", None)
        if li is None:
            return fwd(self, *args, capture_previous_aux=capture_previous_aux, **kwargs)
        out = fwd(self, *args, capture_previous_aux=True, **kwargs)
        aux = out[-1]
        if aux is not None and li >= 1 and STATE.get("ST") is not None:
            n = min(aux.shape[0], 64)
            STATE["ST"][li - 1, :n].copy_(aux[:n].reshape(n, -1))
        return out if capture_previous_aux else (*out[:-1], None)

    layer_cls.forward = forward
    r = STATE["runner"]
    load = r.load_model

    def load_model(*a, **k):
        out = load(*a, **k)
        inner = next(m for m in r.model.modules() if type(m).__name__ == "DeepseekV4Model")
        H = inner.config.hidden_size
        dev = next(inner.parameters()).device
        L = len(inner.layers)
        STATE["ST"] = torch.zeros(L, 64, H, dtype=torch.bfloat16, device=dev)
        STATE["TK"] = torch.full((L, 64, 6), -1, dtype=torch.int32, device=dev)
        for i, lay in enumerate(inner.layers):
            if lay is not inner.layers[PROBE_LAYER] or not TARGET:
                lay._acc3_idx = i
        fb = _s.modules.get("fastpath_boundary")
        if fb is not None:
            orig = fb.router

            def router(li, *ra, layer=0, **rk):
                res = orig(li, *ra, layer=layer, **rk)
                ids = res[0]
                if ids is not None and 0 <= layer < L:
                    n = min(ids.shape[0], 64)
                    STATE["TK"][layer, :n].copy_(ids[:n].to(torch.int32))
                return res

            fb.router = router
            _LOG(f"route capture on: {L} layers (states after each layer, routed expert ids)")
        else:
            _LOG("route capture: fastpath_boundary not loaded, expert ids not captured")
        return out

    r.load_model = load_model


def set_runner(runner):
    STATE["runner"] = runner
