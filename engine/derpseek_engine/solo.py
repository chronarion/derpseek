"""The GB300 alone: experts the rowmap leaves out are never loaded (hook) and never selected (here).

DeepSeek-V4.1 routes with noaux_tc: experts are SELECTED by top-6 of (score + e_score_correction_bias) and WEIGHTED by
the unbiased scores of the selected ones. Adding a large negative number to a left-out expert's bias removes it from
selection on every routing path (the W4 router kernels, vLLM's Mega gate, the stock path), and the next choice takes
its place with its own score. Penalizing in place after load works because every path reads the bias tensor at run
time; it must happen before CUDA graph capture only in the sense that nothing caches a copy (none does here)."""
import json
import os
import re
import sys

import torch

PENALTY = -1.0e4
_LAYER = re.compile(r"(?:^|\.)model\.layers\.(\d+)\.ffn\.gate$")
_LOG = lambda m: sys.stderr.write(f"DERPSEEK {m}\n")


def left_out(rowmap_path: str) -> dict:
    with open(rowmap_path) as f:
        raw = json.load(f)
    layers = raw["layers"] if "layers" in raw else raw
    return {int(k): [int(x) for x in v["cold"]] for k, v in layers.items()}


def penalize_left_out(model, rowmap_path: str) -> int:
    """Add PENALTY to the selection bias of every left-out expert of every target MoE layer. Returns the count."""
    cold = left_out(rowmap_path)
    n = layers = 0
    for name, mod in model.named_modules():
        m = _LAYER.search(name)
        if m is None or not hasattr(mod, "e_score_correction_bias"):
            continue
        idx = int(m.group(1))
        b = mod.e_score_correction_bias
        if b is None or idx not in cold or not cold[idx]:
            continue
        ids = torch.tensor(cold[idx], dtype=torch.long, device=b.device)
        with torch.no_grad():
            b.data[ids] += PENALTY
        n += len(cold[idx])
        layers += 1
    if layers != len([k for k, v in cold.items() if v]):
        raise RuntimeError(f"DERPSEEK: penalized {layers} gate layers, rowmap has {len(cold)} with left-out experts")
    _LOG(f"left-out experts penalized in the routers: {n} over {layers} layers ({rowmap_path})")
    return n


FENCE = [True]   # residency's tune file can switch it off for an A/B ("fence": 0); serving keeps it on


def install_copy_fence(runner) -> None:
    """dsv41 #10: the step's device->host output copy runs on its own stream, ordered only after its own step, while
    the next step (a graph replay that reuses the same pool memory) is already queued. Make the next step wait for the
    copy: after every sample_tokens that returns an async output, the main stream waits on its copy event."""
    orig = runner.sample_tokens

    def sample_tokens(grammar_output, _orig=orig):
        out = _orig(grammar_output)
        ev = getattr(out, "copy_event", None)
        if ev is not None and FENCE[0]:
            torch.cuda.current_stream().wait_event(ev)
            ms = getattr(runner, "main_stream", None)
            if ms is not None and ms != torch.cuda.current_stream():
                ms.wait_event(ev)
        return out

    runner.sample_tokens = sample_tokens
    _LOG("copy fence: each step waits for the previous step's output copy (dsv41 #10)")
