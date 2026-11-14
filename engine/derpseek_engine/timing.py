"""GPU time of each CUDA graph replay (an event pair around the launch on its stream), logged every 30 s.

dsv41's "GPU period" is start-to-start of consecutive replays, so it includes any idle gap the GPU spends waiting for
the host. The replay's own GPU time tells the two apart: period >> replay time means the host (engine loop, residency
tick) is the bottleneck, not the kernels. DERPSEEK_GRAPH_TIMING=0 turns it off."""
import sys
import time

import torch

_PEND, _DONE, _T = [], [], [time.time()]


def _poll():
    while _PEND and _PEND[0][1].query():
        s, e = _PEND.pop(0)
        _DONE.append(s.elapsed_time(e))
    dt = time.time() - _T[0]
    if dt > 30 and _DONE:
        v = sorted(_DONE)
        n = len(v)
        sys.stderr.write(f"DERPSEEK graph replays: {n} in {dt:.0f} s ({n / dt:.1f}/s), GPU time per replay median "
                         f"{v[n // 2]:.3f} ms, mean {sum(v) / n:.3f}, p10 {v[n // 10]:.3f}, p90 {v[9 * n // 10]:.3f}\n")
        _DONE.clear()
        _T[0] = time.time()


def install():
    orig = torch.cuda.CUDAGraph.replay
    if getattr(orig, "_derpseek", False):
        return

    def replay(self, _orig=orig):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        _orig(self)
        e.record()
        _PEND.append((s, e))
        _poll()

    replay._derpseek = True
    torch.cuda.CUDAGraph.replay = replay
