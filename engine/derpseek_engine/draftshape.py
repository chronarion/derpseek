"""Derpseek draft reshape (C1 push, 11-22): DERPSEEK_DRAFT_SHAPE=<json> reshapes DSpark's proposal inside the fused
Markov kernel (overlay/w6mk/w6mk.cu: a context-free function of log q, -log q_max and the draft position, fitted on
self-generated text; draft-only, the kernel writes exactly the logits it draws from into the q cache, so the rejection
sampler stays exact). The reshape makes q sharper, which would inflate K3b's confidence c_j = max q_j; so at batch 1
the kernel also writes the drafter's own (pre-reshape) max and normalizer into K3b's per-split buffers, and this module
skips K3b's _conf_part pass for those steps: the width keeps coming from the drafter's calibrated confidence."""
import importlib
import sys

_LOG = lambda m: sys.stderr.write(f"DERPSEEK draft shape: {m}\n")


def install():
    cw = importlib.import_module("dsv41_engine.conf_width")
    orig = cw._after_propose

    def _after_propose(spec, input_batch):
        w6 = sys.modules.get("w6_markov")
        n = input_batch.num_reqs
        if w6 is None or not w6.CONF[0] or n != 1 or cw.LAMBDA <= 0 or cw.MODE != "graphs":
            return orig(spec, input_batch)
        dl = spec.draft_logits
        cw._buffers(dl.device, dl.shape[0])
        st = cw.STATE
        cw._conf_policy[(1,)](st["pm"], st["ps"], input_batch.idx_mapping, st["width"], st["periods"], st["seq"],
                              cw._HostPtr(st["width_host"]), cw.LAMBDA, cw.SHRINK, KD=cw.KD,
                              KP=cw.triton.next_power_of_2(cw.KD), NSPLIT=cw.NS, RATE=True, num_warps=1)

    cw._after_propose = _after_propose
    _LOG("K3b reads the pre-reshape confidences at batch 1")
