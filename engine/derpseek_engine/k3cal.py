"""K3b with a per-position calibrated acceptance estimate (C1 push, Carbon round 2 idea 1, live test).

DERPSEEK_K3B_SHRINK_VEC="s1,...,s7": K3b's batch-1 rule (widen while expected tokens per ms of step time improve, with the
per-width step periods) estimates position j's acceptance as s_j * c_j instead of SHRINK * c_j (c_j = the drafter's max
q at position j). The factors come from the acc3 dump's training steps (the acceptance-weighted ratio of measured
acceptance to c_j at each position, given the earlier positions survived). Draft-only: any width is exact."""
import importlib
import os
import sys

import torch

from vllm.triton_utils import tl, triton

_LOG = lambda m: sys.stderr.write(f"DERPSEEK k3cal: {m}\n")
VEC = [float(x) for x in os.environ.get("DERPSEEK_K3B_SHRINK_VEC", "").split(",") if x]


@triton.jit
def _mul(a, b):
    return a * b


@triton.jit
def _conf_policy_vec(pm, ps, slots, width, periods, seq, word_host, lam, shrink_v, KD: tl.constexpr, KP: tl.constexpr,
                     NSPLIT: tl.constexpr, RATE: tl.constexpr):
    i = tl.program_id(0)
    j = tl.arange(0, KP)
    sp = tl.arange(0, NSPLIT)
    inside = j < KD
    off = (i * KD + j)[:, None] * NSPLIT + sp[None, :]
    m = tl.load(pm + off, mask=inside[:, None], other=0.0)
    l = tl.load(ps + off, mask=inside[:, None], other=1.0)
    gm = tl.max(m, 1)
    c = 1.0 / tl.sum(l * tl.exp(m - gm[:, None]), 1)
    sv = tl.load(shrink_v + j, mask=inside, other=1.0)
    slot = tl.maximum(tl.load(slots + i), 0)
    if RATE:
        surv = tl.associative_scan(tl.where(inside, c * sv, 1.0), 0, _mul)
        rate = (1.0 + tl.cumsum(tl.where(inside, surv, 0.0), 0)) / tl.load(periods + j, mask=inside, other=1.0)
        w = 1
        go = 1
        for t in tl.static_range(1, KD):
            cur = tl.sum(tl.where(j == t, rate, 0.0), 0)
            prev = tl.sum(tl.where(j == t - 1, rate, 0.0), 0)
            go = go * (cur > prev).to(tl.int32)
            w = w + go
        n = tl.load(seq) + 256
        tl.store(seq, n)
        tl.store(word_host, n + w)
    else:
        surv = tl.associative_scan(tl.where(inside, c, 1.0), 0, _mul)
        w = tl.maximum(tl.sum(tl.where(inside & (surv >= lam), 1, 0), 0), 1)
    tl.store(width + slot, w)


class _Launcher:
    def __init__(self, orig):
        self.orig = orig
        self.vec = None

    def __getitem__(self, grid):
        def launch(pm, ps, slots, width, periods, seq, word_host, lam, shrink, **kw):
            if self.vec is None or self.vec.device != pm.device:
                v = (VEC + [VEC[-1]] * 8)[:kw["KP"]]
                self.vec = torch.tensor(v, dtype=torch.float32, device=pm.device)
            return _conf_policy_vec[grid](pm, ps, slots, width, periods, seq, word_host, lam, self.vec, **kw)
        return launch


def install():
    if not VEC:
        return
    cw = importlib.import_module("dsv41_engine.conf_width")
    cw._conf_policy = _Launcher(cw._conf_policy)
    _LOG(f"per-position acceptance factors {VEC} (scalar SHRINK {cw.SHRINK} unused at batch 1)")
