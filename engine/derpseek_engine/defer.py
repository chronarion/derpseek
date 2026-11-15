"""Derpseek deferred-fast (DERPSEEK_MISS=defer, Fable round 21's v0): a pick whose expert is in CPU memory never makes
a decode step wait and is never skipped; it joins one sublayer late.

In the band layers (DERPSEEK_MISS_LAYERS, 20-38) of decode calls at up to DEFER_MAX_T rows:
  redirect  cutk.late, in the cut kernel's place (after the router, before MegaMoE): every host-row pick moves from
            MegaMoE's indices to the lane's (-1 left behind), counted as a miss so the residency manager promotes it
  fork      a late stream waits for the redirect and for the off-path FFN mix (which needs the spare SMs first), then
            runs `prepare` into the lane's own symmetric buffer and a second MegaMoE instance of the same fork at
            kNumSMs LATE_SMS (8: the 4 SM pairs the 144-SM decode call leaves), no shared expert, with the layer's
            late control block (exact.py: same host rows, lane word 0, own clone table); it combines its own picks
  join      at the next layer's attention-side seam, after that seam has written X: the late stream adds the late
            term with the seam's post-only mode, X2 = X + post_F(l) (x) y_late (identity comb: rows without a late
            pick are X bit for bit); the layer returns X2 as its residual
  wait      the next FFN-side seam (main stream) waits for the late stream before it reads X2
So attention(l+1) (and its mHC coefficients) sees the residual without the late term; everything from FFN(l+1) on has
it. Outside that scope (prompt chunks, stock steps, other widths, layers 0-19 and 39) the in-call lane reads them."""
import os
import sys

import torch

from . import exact

_LOG = lambda m: sys.stderr.write(f"DERPSEEK {m}\n")
ON = exact.MISS_DEFER
LATE_SMS = int(os.environ.get("DERPSEEK_LATE_SMS", "8"))
DEFER_MAX_T = int(os.environ.get("DERPSEEK_DEFER_MAX_T", "8"))
H, I, TOPK, MAX_ROWS = 5120, 2304, 6, 64

_LANE = {}        # device index -> dict(stream, symm, y, late_idx)
_LATE = []        # forked lanes not joined yet: (lane, T, layer)
_JOIN = []        # late streams the next FFN-side seam waits for
_WARM = set()     # widths whose lane kernels ran eagerly (a capture must not JIT)
_SEEN = set()


def _lane(dev, n_rows: int):
    L = _LANE.get(dev.index)
    if L is None:
        from vllm.distributed import get_ep_group
        from vllm.utils.deep_gemm import _import_deep_gemm
        dg = _import_deep_gemm()
        symm = dg.get_symm_buffer_for_mega_moe(get_ep_group().device_group, n_rows, MAX_ROWS, TOPK, H, I,
                                               num_shared_experts=0, mma_type="fp8xfp4")
        L = _LANE[dev.index] = dict(stream=torch.cuda.Stream(device=dev), symm=symm,
                                    y=torch.empty(MAX_ROWS, H, dtype=torch.bfloat16, device=dev),
                                    late_idx=torch.full((MAX_ROWS, TOPK), -1, dtype=torch.int64, device=dev))
        _LOG(f"deferred-fast lane: {LATE_SMS} SMs, its own symmetric buffer ({n_rows} rows, {MAX_ROWS} tokens), "
             f"layers {min(exact.MISS_LAYERS)}-{max(exact.MISS_LAYERS)}, T <= {DEFER_MAX_T}")
    return L


def eligible(st, T: int, ws_decode: bool = True) -> bool:
    """Whether this MoE call defers its host picks (decided before the redirect: a redirect without a fork loses picks).
    Router-path decode calls in whole-step graphs, vLLM's own decode steps and the KLD harness alike (the join is in
    the same forward); never in piecewise graphs, and a capture only at widths whose lane kernels ran eagerly."""
    if not ON or st.idx not in exact.MISS_LAYERS or getattr(st, "late_ctl", None) is None:
        return False
    if not (1 <= T <= DEFER_MAX_T):
        return False
    from vllm.forward_context import in_piecewise_cudagraph
    if in_piecewise_cudagraph():
        return False
    return T in _WARM or not torch.cuda.is_current_stream_capturing()


def fork(exp, hidden_states, symm, T: int, clamp, pending_streams) -> None:
    """After the router: redirect the host picks, then run the late lane beside MegaMoE (call before _mega_moe_call)."""
    from vllm.utils.deep_gemm import _import_deep_gemm
    from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import prepare_megamoe_inputs
    from . import cutk
    dg = _import_deep_gemm()
    st = exp._mp
    dev = hidden_states.device
    L = _lane(dev, st.n_rows)
    late_idx = L["late_idx"][:T]
    cutk.late(st, symm.topk_idx[:T], late_idx)              # main stream, after the router
    s = L["stream"]
    s.wait_stream(torch.cuda.current_stream())              # edge: the redirect (and so the router)
    for ms in pending_streams:                              # edge: the off-path FFN mix takes the spare SMs first
        s.wait_stream(ms)
    ls = L["symm"]
    with torch.cuda.stream(s):
        prepare_megamoe_inputs(hidden_states, symm.topk_weights[:T], late_idx.clamp(min=0), ls.x[:T], ls.x_sf[:T],
                               ls.topk_idx[:T], ls.topk_weights[:T])
        ls.topk_idx[:T].copy_(late_idx)                     # -1: not a late pick
        prev = dg.get_num_sms()
        dg.set_num_sms(min(LATE_SMS, prev))
        try:
            dg.fp8_fp4_mega_moe(L["y"][:T], exp._transformed_l1_weights, exp._transformed_l2_weights, ls,
                                cumulative_local_expert_recv_stats=st.late_ctl[:st.n_rows],
                                activation_clamp=clamp, fast_math=True)
        finally:
            dg.set_num_sms(prev)
    hidden_states.record_stream(s)
    if not torch.cuda.is_current_stream_capturing():
        _WARM.add(T)
    _LATE.append((L, T, int(st.idx)))
    if (st.idx, T) not in _SEEN and len(_SEEN) < 8:
        _SEEN.add((st.idx, T))
        _LOG(f"deferred-fast: layer {st.idx} T={T} host picks to the late lane "
             f"(capturing {torch.cuda.is_current_stream_capturing()})")


def join_attn_side(seam_fn, res_new, post_layer_mix, pre_mix, norm_weight, T: int, comb_identity):
    """At the next layer's attention-side seam, after it wrote res_new (X): X2 = X + post_F(l) (x) y_late on the late
    stream. Returns X2 (the layer's residual from here on), or res_new when nothing is pending."""
    if not _LATE:
        return res_new
    L, T0, layer = _LATE.pop()
    if T0 != T:
        raise RuntimeError(f"DERPSEEK deferred-fast: lane of layer {layer} forked at T={T0}, joined at T={T}")
    s = L["stream"]
    s.wait_stream(torch.cuda.current_stream())              # edge: seam A has written X
    with torch.cuda.stream(s):
        x2, _, _, _ = seam_fn(L["y"][:T], res_new, post_layer_mix.reshape(T, -1), comb_identity[:T], pre_mix,
                              norm_weight, post_only=True)
    for t in (res_new, x2, post_layer_mix):
        t.record_stream(s)
    _JOIN.append(s)
    return x2


def wait_ffn_side() -> None:
    """Before the next FFN-side seam reads X2."""
    while _JOIN:
        torch.cuda.current_stream().wait_stream(_JOIN.pop())


def join_all() -> None:
    """At the end of a forward: nothing forked may outlive it (a capture must not end with a stream unjoined)."""
    while _LATE:
        L, _, layer = _LATE.pop()
        _LOG(f"deferred-fast: lane of layer {layer} not joined at a seam (its late term is lost); joined at the "
             f"forward's end (capturing {torch.cuda.is_current_stream_capturing()})")
        torch.cuda.current_stream().wait_stream(L["stream"])
    wait_ffn_side()


def join_segment() -> None:
    """At the end of a breakable-capture segment: the main stream waits for every forked lane and pending join (their
    work completes inside this segment); the pending joins stay queued, so the late term still joins at its seam."""
    main = torch.cuda.current_stream()
    for L, _, _ in _LATE:
        main.wait_stream(L["stream"])
    for s in _JOIN:
        main.wait_stream(s)
