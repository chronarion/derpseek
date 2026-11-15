"""Derpseek lane probe (deferred-fast D0'): what does a decode step lose when a finite side kernel holds a few SMs and
reads CPU memory over NVLink-C2C, as a late lane would? Served decode MegaMoE runs at kNumSMs 144 (dsv41 E3,
DSV41_MIX_OFFPATH), so 4 SM pairs sit beside it; dsv41's engram_side already runs a 32-SM side GEMM in the step graphs.

DERPSEEK_LANE_PROBE=1 installs, in DeepseekV4DecoderLayer.forward of each layer in DERPSEEK_LANE_LAYERS (21-39), a fork
at the layer's start (after the previous layer's MegaMoE call): a side stream waits for the main stream and launches
`lane_probe` on DERPSEEK_LANE_SMS SMs (8, as 2-CTA clusters of 1,024 threads holding most of the shared memory, as a
MegaMoE instance would), which copies probe_mb MiB from pinned CPU memory into a GPU scratch buffer and exits. The main
stream joins it (DERPSEEK_LANE_JOIN):
  attn   when this layer's attention module returns (before its FFN boundary): the late-join window, never beside MegaMoE
  layer  when this layer's forward returns: the lane may also run beside this layer's router and MegaMoE
The fork / join structure is fixed when the step graphs are captured; the work is switched at run time by the tune file
(residency.TUNE "probe": 0 = the kernel returns at once, 1 = copy; "probe_mb"), so both arms run in one boot.

DERPSEEK_STEP_TIMER=1 (default with the probe) times every CUDA graph replay with CUDA events and logs, every 20 s, the
mean GPU time per graph and probe state ("step timer" lines). Decode forwards only (1 <= T <= 64), never in piecewise
graphs (the engram_side rule)."""
import collections
import os
import sys
import time

import torch

_LOG = lambda m: sys.stderr.write(f"DERPSEEK {m}\n")


def _span(s: str):
    out = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


ON = os.environ.get("DERPSEEK_LANE_PROBE", "0") == "1"
SMS = int(os.environ.get("DERPSEEK_LANE_SMS", "8")) & ~1
LAYERS = _span(os.environ.get("DERPSEEK_LANE_LAYERS", "21-39"))
JOIN = os.environ.get("DERPSEEK_LANE_JOIN", "attn")
# where the fork is: "layer" (the start of each layer in LAYERS: after the previous layer's MegaMoE call) or "router"
# (in the MoE call of each layer in MOE_LAYERS, after its router and its off-path FFN mix, right before MegaMoE: the
# probe then runs beside MegaMoE on the 4 SM pairs the 144-SM call leaves, then beside the next layer's seam and
# attention, and is joined when that attention returns (Fable round 21's deferred-fast fork point)
FORK = os.environ.get("DERPSEEK_LANE_FORK", "layer")
MOE_LAYERS = _span(os.environ.get("DERPSEEK_LANE_MOE_LAYERS", "20-38"))
TIMER = os.environ.get("DERPSEEK_STEP_TIMER", "1" if ON else "0") == "1"
MAX_T = 64
SMEM = 200 * 1024
BUF_MB = 64

_SRC = r"""
#include <cuda_runtime.h>
#include <cstdint>
// One fork's reads: cfg[1] bytes from pinned CPU memory into a GPU buffer, split evenly over the CTAs, four 16-byte
// loads in flight per thread (plain loads read ~135 GB/s on 8 SMs, ~185 on 16; MegaMoE's TMA lane ~350 on 8). cfg[0] == 0: return at once (the probe's off arm keeps the same graph).
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(1024, 1)
lane_probe(const int4* __restrict__ src, int4* __restrict__ dst, const volatile int* cfg) {
  extern __shared__ int4 smem[];
  if (cfg[0] == 0) return;
  const long long n = ((long long)(unsigned)cfg[1]) >> 4;
  const long long per = (n + gridDim.x - 1) / gridDim.x;
  const long long b = per * blockIdx.x;
  const long long e = b + per < n ? b + per : n;
  const long long nt = blockDim.x;
  long long i = b + threadIdx.x;
  for (; i + 3 * nt < e; i += 4 * nt) {
    const int4 a0 = __ldcs(src + i), a1 = __ldcs(src + i + nt), a2 = __ldcs(src + i + 2 * nt), a3 = __ldcs(src + i + 3 * nt);
    __stcs(dst + i, a0);
    __stcs(dst + i + nt, a1);
    __stcs(dst + i + 2 * nt, a2);
    __stcs(dst + i + 3 * nt, a3);
  }
  for (; i < e; i += nt) __stcs(dst + i, __ldcs(src + i));
  if (e < 0 && threadIdx.x == 0) smem[0] = make_int4(0, 0, 0, 0);  // never: keeps the shared-memory request honest
}

int prepare(int64_t smem) {
  return (int)cudaFuncSetAttribute(lane_probe, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
}

int probe(int64_t src, int64_t dst, int64_t cfg, int64_t nblk, int64_t smem, int64_t stream) {
  lane_probe<<<(int)nblk, 1024, (size_t)smem, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const int4*>(src), reinterpret_cast<int4*>(dst), reinterpret_cast<const volatile int*>(cfg));
  return (int)cudaGetLastError();
}
"""
_CPP = "int prepare(int64_t smem);\nint probe(int64_t src, int64_t dst, int64_t cfg, int64_t nblk, int64_t smem, int64_t stream);"

_S = {}               # mod, src, dst, cfg, side (set by _prepare, outside any capture)
_JOINS: list = []     # done events of forks the main stream has not joined yet
_SEEN = set()


def set_probe(on: int, mb: float) -> None:
    """Called by residency._tune: switch the probe's work (no recapture)."""
    cfg = _S.get("cfgn")
    if cfg is None:
        return
    nbytes = int(max(0.0, min(float(mb), BUF_MB)) * 2**20) & ~15
    cfg[1] = nbytes
    cfg[0] = 1 if (on and nbytes > 0) else 0
    _LOG(f"lane probe: {'on' if cfg[0] else 'off'}, {nbytes / 2**20:.1f} MiB per fork, {SMS} SMs, join {JOIN}")


def _prepare(dev) -> None:
    if _S:
        return
    from torch.utils.cpp_extension import load_inline
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/root/.cache/vllm/torch_extensions")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.3a")
    mod = load_inline("derpseek_lane_probe", cpp_sources=_CPP, cuda_sources=_SRC, functions=["prepare", "probe"],
                      extra_cuda_cflags=["-O3"], verbose=False)
    rc = mod.prepare(SMEM)
    src = torch.ones(BUF_MB << 20, dtype=torch.uint8).pin_memory()
    dst = torch.empty(BUF_MB << 20, dtype=torch.uint8, device=dev)
    cfg = torch.zeros(4, dtype=torch.int32).pin_memory()
    side = torch.cuda.Stream(device=dev)
    _S.update(mod=mod, src=src, dst=dst, cfg=cfg, cfgn=cfg.numpy(), side=side)
    rc2 = mod.probe(src.data_ptr(), dst.data_ptr(), cfg.data_ptr(), SMS, SMEM, torch.cuda.current_stream(dev).cuda_stream)
    torch.cuda.synchronize(dev)
    from . import residency
    set_probe(residency.TUNE.get("probe", 0), residency.TUNE.get("probe_mb", 18.0))
    _LOG(f"lane probe ready: layers {min(LAYERS)}-{max(LAYERS)}, {SMS} SMs, join {JOIN} (prepare rc {rc}, warm rc {rc2})")


def _eligible(T: int) -> bool:
    if not (1 <= T <= MAX_T) or not _S:
        return False
    from vllm.forward_context import in_piecewise_cudagraph
    return not in_piecewise_cudagraph()


def _fork(lid: int, T: int) -> None:
    main = torch.cuda.current_stream()
    side = _S["side"]
    start = torch.cuda.Event()
    start.record(main)
    side.wait_event(start)   # never before this layer starts: the previous layer's MegaMoE has been enqueued
    with torch.cuda.stream(side):
        _S["mod"].probe(_S["src"].data_ptr(), _S["dst"].data_ptr(), _S["cfg"].data_ptr(), SMS, SMEM, side.cuda_stream)
        done = torch.cuda.Event()
        done.record(side)
    _JOINS.append(done)
    if (lid, T) not in _SEEN and len(_SEEN) < 8:
        _SEEN.add((lid, T))
        _LOG(f"lane probe forked at layer {lid} (T={T}, capturing {torch.cuda.is_current_stream_capturing()})")


def fork_at_router(layer: int, T: int, mix_streams) -> None:
    """From the boundary's MoE forward (FORK=router), after the router (and the cut), before MegaMoE."""
    if FORK != "router" or layer not in MOE_LAYERS or not _eligible(T):
        return
    main = torch.cuda.current_stream()
    side = _S["side"]
    start = torch.cuda.Event()
    start.record(main)
    side.wait_event(start)
    for ms in mix_streams:   # the off-path FFN mix needs its SMs first (it holds 16 for ~16 us)
        side.wait_stream(ms)
    with torch.cuda.stream(side):
        _S["mod"].probe(_S["src"].data_ptr(), _S["dst"].data_ptr(), _S["cfg"].data_ptr(), SMS, SMEM, side.cuda_stream)
        done = torch.cuda.Event()
        done.record(side)
    _JOINS.append(done)
    if (layer, T, "r") not in _SEEN and len(_SEEN) < 8:
        _SEEN.add((layer, T, "r"))
        _LOG(f"lane probe forked beside MegaMoE of layer {layer} (T={T}, after {len(mix_streams)} mix stream(s), "
             f"capturing {torch.cuda.is_current_stream_capturing()})")


def _join() -> None:
    main = torch.cuda.current_stream()
    while _JOINS:
        main.wait_event(_JOINS.pop())


def _patch_attn_class(acls) -> None:
    """Class level, so instance-level wrappers (engram_side's) that call type(attn).forward still pass through it."""
    if getattr(acls, "_derpseek_lane", False):
        return
    orig = acls.forward

    def forward(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        if _JOINS and JOIN == "attn":
            _join()
            if "attn_join" not in _S:
                _S["attn_join"] = 1
                _LOG(f"lane probe joined when {type(self).__name__} returned (before the FFN boundary)")
        return out

    acls.forward = forward
    acls._derpseek_lane = True


def install(module) -> None:
    """From the boundary overlay's model patch (vllm.models.deepseek_v41.nvidia.model)."""
    if TIMER:
        _install_timer()
    if not ON:
        return
    cls = module.DeepseekV4DecoderLayer
    if getattr(cls, "_derpseek_lane", False):
        return
    orig_init, orig_fwd = cls.__init__, cls.forward

    def __init__(self, vllm_config, prefix, *args, **kwargs):
        orig_init(self, vllm_config, prefix, *args, **kwargs)
        self._derpseek_lid = module.extract_layer_index(prefix)
        attn = getattr(self, "attn", None)
        if self._derpseek_lid in LAYERS and isinstance(attn, torch.nn.Module):
            _patch_attn_class(type(attn))

    def forward(self, x, positions, input_ids, *args, **kwargs):
        lid = getattr(self, "_derpseek_lid", -1)
        T = positions.shape[0]
        if not _S and not torch.cuda.is_current_stream_capturing():
            _prepare(positions.device)
        if FORK == "layer" and lid in LAYERS and _eligible(T):
            _fork(lid, T)
        out = orig_fwd(self, x, positions, input_ids, *args, **kwargs)
        if _JOINS and not (FORK == "router" and lid in MOE_LAYERS):
            _join()   # JOIN=layer; otherwise a safety net (a capture must not end with an unjoined fork)
        return out

    cls.__init__ = __init__
    cls.forward = forward
    cls._derpseek_lane = True
    where = (f"the start of layers {min(LAYERS)}-{max(LAYERS)}" if FORK == "layer"
             else f"the MoE calls of layers {min(MOE_LAYERS)}-{max(MOE_LAYERS)} after router and mix")
    _LOG(f"lane probe installed: fork at {where}, {SMS} SMs, join {JOIN}")


# ------------------------------------------------------------------------------------------------ step timer
_T = {"pool": [], "infl": collections.deque(), "acc": collections.defaultdict(lambda: [0, 0.0]), "ids": {},
      "t": time.time()}


def _ev():
    return _T["pool"].pop() if _T["pool"] else torch.cuda.Event(enable_timing=True)


def _drain(force: bool = False) -> None:
    infl = _T["infl"]
    while infl and (force or infl[0][3].query()):
        key, e0, e1 = infl[0][0], infl[0][2], infl[0][3]
        if force:
            e1.synchronize()
        infl.popleft()
        a = _T["acc"][key]
        a[0] += 1
        a[1] += e0.elapsed_time(e1)
        _T["pool"] += [e0, e1]
    now = time.time()
    if now - _T["t"] >= 20 and _T["acc"]:
        _T["t"] = now
        rows = sorted(_T["acc"].items(), key=lambda kv: -kv[1][0])[:12]
        _LOG("step timer: " + "; ".join(f"g{g} probe {p}: n {n} mean {s / n:.4f} ms" for (g, p), (n, s) in rows))
        _T["acc"].clear()


def _install_timer() -> None:
    cls = torch.cuda.CUDAGraph
    if getattr(cls, "_derpseek_timer", False):
        return
    orig = cls.replay

    def replay(self, _orig=orig):
        if torch.cuda.is_current_stream_capturing():
            return _orig(self)
        s = torch.cuda.current_stream()
        e0, e1 = _ev(), _ev()
        e0.record(s)
        _orig(self)
        e1.record(s)
        g = _T["ids"].setdefault(id(self), len(_T["ids"]))
        cfg = _S.get("cfgn")
        state = "-" if cfg is None else (f"{int(cfg[1]) >> 20}M" if cfg[0] else "off")
        _T["infl"].append(((g, state), None, e0, e1))
        _drain()

    cls.replay = replay
    cls._derpseek_timer = True
    _LOG("step timer: every CUDA graph replay timed with CUDA events (logged every 20 s)")
