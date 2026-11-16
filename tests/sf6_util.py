"""Helpers for the tests' SF6=1 mode: 6-bit scale codes (engine/derpseek_engine/sf6.py, the fork's derpseek_sf6).
The tests keep comparing against the uncoded all-HBM call, so SF6=1 checks that the coded rows give the same bits.
E0 and the layout are read from the fork header in use ($FORK), so the tests cannot disagree with the kernel."""
import os
import re
import sys

import torch

sys.path.insert(0, os.environ.get("DERPSEEK_ENGINE", "/d/engine/derpseek_engine"))
import sf6  # noqa: E402

_h = open(os.path.join(os.environ["FORK"], "include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh")).read()
CODED_FORK = "namespace derpseek_sf6" in _h          # the fork knows the codes (else only SF6_SCALES makes sense)


def _define(name, default):
    m = re.search(rf"^#define {name} (\d+)", _h, re.M)
    return int(m.group(1)) if m else default


if CODED_FORK:
    sf6.E0, sf6.LAYOUT = _define("DRP_SF6_E0", 96), _define("DRP_SF6_LAYOUT", 2)
# SF6_PROF=1: the MMA warp's counters (written by the profile fork only, FORK=.../fork-prof; report() prints them)
PROF = os.environ.get("SF6_PROF", "0") == "1" and "#define DRP_SF6_PROF" in _h
CLONE_BYTES = sf6.CLONE_BYTES
_KEEP = []                      # counters tensors of the tables built below


def scales(n, rows, groups, g, dev):
    """UE8M0 scale bytes [n, rows, groups] over the checkpoint's whole range 117..127 (r23a/scan-full.json): a base per
    output row in 117..124 plus 0..3 (how the checkpoint's rows look), so every byte value occurs"""
    base = torch.randint(117, 125, (n, rows, 1), device=dev, generator=g)
    return (base + torch.randint(0, 4, (n, rows, groups), device=dev, generator=g)).to(torch.uint8)


def code(t):
    """A transformed scale tensor int32 [n, N, K / 128] -> its coded bank uint8 [n, row_bytes]; the reference decoder
    must give the scales back"""
    c = sf6.encode(t)
    for i in range(0, t.shape[0], 16):
        assert torch.equal(sf6.decode(c[i:i + 16], t.shape[1], t.shape[2]), t[i:i + 16]), "reference decode"
    return c


def pinned(t):
    h = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    h.copy_(t)
    return h


def span(coded, rows, raw):
    """The launcher tensor of a coded bank that claims `rows` experts (raw: the uncoded tensor, for its shape and strides)"""
    assert raw.dtype == torch.int32
    return sf6.span(coded, rows, raw.shape, raw.stride())


def desc_table(host1, host2, hbm1, hbm2, raw1, raw2):
    """A layer's clone table with the coded scale maps after it (sf6.write_tail): host1 / host2 the pinned coded host
    banks of the first / second GEMM, hbm1 / hbm2 the coded HBM banks (the tensors the launcher spans point into), raw1 /
    raw2 uncoded scale tensors of the two matrices (for N and K / 128). With SF6_PROF=1 also the counters."""
    dev = hbm1.device
    t = torch.zeros(CLONE_BYTES + sf6.TAIL_BYTES, dtype=torch.uint8, device=dev)
    stats = sf6.new_stats(dev) if PROF else None
    sf6.write_tail(t[CLONE_BYTES:], host1, host2, hbm1, hbm2, tuple(raw1.shape[1:]), tuple(raw2.shape[1:]), stats)
    if stats is not None:
        _KEEP.append(stats)
    return t


def stats_table(dev):
    """A clone table for an uncoded call of a profile run: no maps, only the counters' address"""
    t = torch.zeros(CLONE_BYTES + sf6.TAIL_BYTES, dtype=torch.uint8, device=dev)
    stats = sf6.new_stats(dev)
    t[CLONE_BYTES + sf6.NUM_MAPS * 128:CLONE_BYTES + sf6.NUM_MAPS * 128 + 8].view(torch.int64)[0] = stats.data_ptr()
    _KEEP.append(stats)
    return t


def ctl_word(coded=True):
    """Control word 11 for the tests' calls (coded rows; with SF6_PROF=1 the counters, also for uncoded rows)"""
    return (sf6.CTL_ON if coded else 0) | (sf6.CTL_PROF if PROF else 0)


def report(label="", reset=True):
    """SF6_PROF=1: one line per counters tensor since the last report (cycles of the SM's clock)"""
    for t in _KEEP:
        if int(t.sum()) != 0:
            print(f"SF6 counters {label}: {sf6.stats_line(t)}", flush=True)
            if reset:
                t.zero_()


def describe(raw1, raw2):
    b0 = raw1[0].numel() * 4 + raw2[0].numel() * 4
    b1 = sf6.row_bytes(*raw1.shape[1:]) + sf6.row_bytes(*raw2.shape[1:])
    return f"SF6: scale banks coded (layout {sf6.LAYOUT}, E0 {sf6.E0}): {b1} B per expert, uncoded {b0} (-{b0 - b1})"
