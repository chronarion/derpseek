"""Derpseek exact mode (DERPSEEK_MODE=exact): every expert stays exact; the ones not resident in HBM are read from
pinned host memory over NVLink-C2C by MegaMoE itself (a miss costs time, never precision).

Per target MoE layer the rowmap gives `hot` (static, always in HBM), `cache` (the experts that start in the HBM cache
rows) and `host` (every non-static expert: each keeps a permanent exact copy in a host row). MegaMoE's rows:
    [0, D)             host rows (pinned host memory): host[j] at row j. First, because MegaMoE claims tasks in expert
                       order: a host expert's slow tiles start first and overlap the HBM experts' work
    [D, D + S)         static experts (HBM tensor rows 0..S-1)
    [D + S, D + S + C) cache rows (HBM tensor rows S..S+C-1): copies of host experts, swapped by the residency manager
`row_map[e]` (device int32, read by the routers at run time) is the row an expert's picks go to: its static row, its
cache row while cached, else its host row. The launcher tensors start at the HBM rows and claim all D + S + C rows;
the MegaMoE fork reads rows < D through host-based clones of its tensor maps and row r >= D at HBM row r - D
(overlay/megamoe_fork, namespace derpseek_host). The per-layer control block rides in MegaMoE's
`cumulative_local_expert_recv_stats` argument and receives per-row token counts at words 16..
"""
import functools
import os
import sys
import threading

import torch

MAGIC = 0x53505244          # "DRPS", = derpseek_host::kMagic in the fork
E = 384
HDR = 16                    # control block header words; per-row counts follow
# words: 0 magic, 1 host rows D, 2-9 the four host bases (int64), 10 host-lane clusters (the MegaMoE fork's scheduler
# sends host-row tasks only to the last N two-CTA clusters, so link-bound tiles never hold more than 2N SMs; 0 = off),
# 12-13 the lane's claim counters (the kernel zeroes them after each call), 14-15 this layer's tensor-map clone table
# (device address, int64; per layer because a call writes its clones before its dependency wait, i.e. possibly while
# the previous layer's call still loads through its own)
DESC_BYTES = 160 * 4 * 128  # derpseek_host::kMaxSMs CTAs x 4 maps x 128 B
HOST_LANE = int(os.environ.get("DERPSEEK_HOST_LANE", "4"))
# dsv41 a12_retire for exact mode: each layer's two HBM scale-factor banks (L1, L2) live in a generic-compression
# allocation (dsv41 runtime fastpath/native_scales.py): the UE8M0 scales compress well, so MegaMoE reads fewer bytes.
# The host rows stay plain pinned memory; the residency manager's copies into compressed rows are followed by an SM
# rewrite of those rows, since a copy-engine write alone may leave them uncompressed (correct either way).
A12 = os.environ.get("DERPSEEK_A12", "0") == "1"
_KEYS = ("w13", "w13s", "w2", "w2s")
_LOG = lambda m: sys.stderr.write(f"DERPSEEK {m}\n")
CTL = {}                    # data_ptr of a layer's L1 weight tensor -> its control block view (numel = rows)
POOL = {}                   # one contiguous device tensor per kind for all layers: 'ctl' [L, HDR + rows], 'row_map' [L, E]
NUM_LAYERS = 40
LAYERS = {}                 # layer idx -> ExactState
_TS = {torch.int8: "|i1", torch.uint8: "|u1", torch.int32: "<i4", torch.float32: "<f4"}
_lock = threading.Lock()


class _Arr:
    def __init__(self, t, rows):
        self.__cuda_array_interface__ = {
            "shape": (rows,) + tuple(t.shape[1:]), "typestr": _TS[t.dtype], "data": (t.data_ptr(), False),
            "strides": tuple(s * t.element_size() for s in t.stride()), "version": 3}


def spanning(t: torch.Tensor, rows: int) -> torch.Tensor:
    """A view of `t` (rows-major, dense per row) that claims `rows` rows. Only t's own rows are backed: the MegaMoE
    fork never reads past them through this tensor's map (the remaining rows come from the host-based clones)."""
    assert t.stride(0) == t[0].numel(), "rows must be dense"
    out = torch.as_tensor(_Arr(t, rows), device=t.device)
    return out if out.dtype == t.dtype else out.view(t.dtype)


class ExactState:
    def __init__(self, idx: int, hot: list, cache: list, host: list):
        self.idx = idx
        self.hot, self.cache, self.host = list(hot), list(cache), list(host)
        self.S, self.C, self.D = len(self.hot), len(self.cache), len(self.host)
        self.n_hbm = self.S + self.C
        self.n_rows = self.S + self.C + self.D
        if sorted(self.hot + self.host) != list(range(E)) or not set(self.cache) <= set(self.host):
            raise RuntimeError(f"DERPSEEK exact rowmap layer {idx}: hot + host must partition 0..{E - 1}, cache within host")
        self.host_row = {e: j for j, e in enumerate(self.host)}          # kernel row = host tensor row
        self.hbm_row = {e: r for r, e in enumerate(self.hot)}            # HBM tensor row (kernel row = D + it)
        for c, e in enumerate(self.cache):
            self.hbm_row[e] = self.S + c
        self.row = [self.D + self.hbm_row[e] if e in self.hbm_row else self.host_row[e] for e in range(E)]
        self.slot = list(self.cache)          # expert held by each cache slot (residency manager)
        self.stash = {}                       # expert -> {key: CPU tensor}, raw weights of host experts until finalize
        self.row_map = None
        self.hbm = self.host_t = self.ctl_buf = None
        # the hook's ColdState fields the boundary and other components read
        self.n_hot, self.n_cold, self.warm = self.n_hbm, 0, []
        self.row_map_static = self.warm_row_map = self.adapt = None

    # ---------------------------------------------------------------------------------------------- load time
    def stash_raw(self, mod, param, loaded_weight, weight_name, shard_id, expert_id):
        key = {id(mod.w13_weight): "w13", id(mod.w13_weight_scale): "w13s", id(mod.w2_weight): "w2",
               id(mod.w2_weight_scale): "w2s"}.get(id(param))
        if key is None:
            return
        ok = (shard_id in ("w1", "w3") and "w13_" in weight_name) or (shard_id == "w2" and "w2_" in weight_name)
        if not ok:
            return
        with _lock:
            d = self.stash.setdefault(expert_id, {})
            buf = d.get(key)
            if buf is None:
                buf = d[key] = torch.empty(tuple(param.shape[1:]), dtype=param.dtype)
        dst = buf
        if shard_id in ("w1", "w3"):
            dst = dst.narrow(0, 0 if shard_id == "w1" else mod.intermediate_size, mod.intermediate_size)
        dst.copy_(loaded_weight)

    # ------------------------------------------------------------------------------------------------ finalize
    def finalize(self, mod):
        """After the stock finalize transformed the HBM rows: build the host rows, the spanning launcher tensors, the
        control block and the row map. Runs on vLLM's weight-loader thread (devices are explicit)."""
        backend = mod._ensure_backend() if mod._backend is None else mod._backend
        hbm = [mod._transformed_l1_weights[0], mod._transformed_l1_weights[1],
               mod._transformed_l2_weights[0], mod._transformed_l2_weights[1]]
        dev = hbm[0].device
        self.a12 = []
        if A12:
            hbm = self._compress_scales(hbm, dev)
        if self.stash.keys() != set(self.host) or any(set(v) != set(_KEYS) for v in self.stash.values()):
            missing = [e for e in self.host if set(self.stash.get(e, {})) != set(_KEYS)]
            raise RuntimeError(f"DERPSEEK layer {self.idx}: raw weights incomplete for host experts {missing[:8]}")
        host = [torch.empty_strided((self.D,) + tuple(t.shape[1:]), t.stride(), dtype=t.dtype, pin_memory=True)
                for t in hbm]
        B = 8
        for j0 in range(0, self.D, B):
            es = self.host[j0:j0 + B]
            raw = {k: torch.stack([self.stash[e][k] for e in es]).to(dev) for k in _KEYS}
            out = backend.transform_weights(w13_weight=raw["w13"], w13_weight_scale=raw["w13s"], w2_weight=raw["w2"],
                                            w2_weight_scale=raw["w2s"], num_local_experts=len(es),
                                            hidden_size=mod.hidden_size, intermediate_size=mod.intermediate_size)
            outs = (out[0][0], out[0][1], out[1][0], out[1][1])
            for h, o, t in zip(host, outs, hbm):
                if tuple(o.shape[1:]) != tuple(t.shape[1:]) or o.dtype != t.dtype or o.stride()[1:] != t.stride()[1:]:
                    raise RuntimeError(f"DERPSEEK transform layout {tuple(o.shape)} {o.stride()} vs {tuple(t.shape)} {t.stride()}")
                h[j0:j0 + len(es)].copy_(o)
            del raw, out, outs
        torch.cuda.synchronize(dev)
        self.stash = {}
        # self-check: each cached expert's HBM row equals its host row, byte for byte
        for c, e in enumerate(self.cache):
            r, j = self.S + c, self.host.index(e)
            for k, (h, t) in enumerate(zip(host, hbm)):
                if not torch.equal(t[r], h[j].to(dev)):
                    raise RuntimeError(f"DERPSEEK self-check: layer {self.idx} expert {e} ({_KEYS[k]}) HBM row {r} != host row {j}")
        span = [spanning(t, self.n_rows) for t in hbm]
        with _lock:   # all layers' control blocks and row maps live in two contiguous tensors (one copy reads / updates them all)
            if "ctl" not in POOL:
                POOL["ctl"] = torch.zeros(NUM_LAYERS, HDR + self.n_rows, dtype=torch.int32, device=dev)
                POOL["row_map"] = torch.zeros(NUM_LAYERS, E, dtype=torch.int32, device=dev)
                POOL["desc"] = torch.zeros(NUM_LAYERS, DESC_BYTES, dtype=torch.uint8, device=dev)
        if POOL["ctl"].shape[1] != HDR + self.n_rows:
            raise RuntimeError("DERPSEEK exact: every layer needs the same row count")
        ctl = POOL["ctl"][self.idx]
        hdr = torch.zeros(HDR, dtype=torch.int32)
        hdr[0], hdr[1] = MAGIC, self.D     # rows < D are host rows
        hdr[10] = HOST_LANE
        desc = POOL["desc"][self.idx]
        assert desc.data_ptr() % 128 == 0
        hdr.view(torch.int64)[7] = desc.data_ptr()
        hdr.view(torch.int64)[1:5] = torch.tensor([h.data_ptr() for h in host], dtype=torch.int64)
        ctl[:HDR].copy_(hdr.to(dev))
        mod._transformed_l1_weights = (span[0], span[1])
        mod._transformed_l2_weights = (span[2], span[3])
        mod.num_experts = mod.num_local_experts = self.n_rows
        mod.experts_end_idx = mod.experts_start_idx + self.n_rows
        self.hbm, self.host_t, self.span, self.ctl_buf = hbm, host, span, ctl
        self.row_map = POOL["row_map"][self.idx]
        self.row_map.copy_(torch.tensor(self.row, dtype=torch.int32).to(dev))
        self.flat_hbm = [torch.as_strided(t, (t.shape[0], t[0].numel()), (t[0].numel(), 1)) for t in hbm]
        self.flat_host = [torch.as_strided(h, (h.shape[0], h[0].numel()), (h[0].numel(), 1)) for h in host]
        CTL[span[0].data_ptr()] = ctl[:self.n_rows]
        LAYERS[self.idx] = self
        install_ctl_injection()
        from . import prefill
        prefill.reserve(self)
        if self.idx in (0, 39):
            mb = sum(h.numel() * h.element_size() for h in host) >> 20
            _LOG(f"exact layer {self.idx}: {self.S} static + {self.C} cache rows in HBM, {self.D} host rows "
                 f"({mb} MiB pinned), self-check of {self.C} cached experts passed, host lane {HOST_LANE} clusters")

    def _compress_scales(self, hbm, dev):
        """The two scale banks into compressible device memory (same layout, verified equal); keeps the owners."""
        try:
            from native_scales import NativeScales
        except ImportError as e:
            _LOG(f"layer {self.idx}: a12 scales unavailable ({e}); scale banks stay uncompressed")
            return hbm
        out = list(hbm)
        with torch.cuda.device(dev):   # the driver calls need this thread's context (vLLM's loader thread has none)
            for k in (1, 3):
                owner = NativeScales(hbm[k], True)
                if owner.storage.granted != 1:
                    raise RuntimeError(f"DERPSEEK layer {self.idx}: compressible allocation not granted")
                self.a12.append(owner)
                out[k] = owner.tensor
        if self.idx in (0, 39):
            _LOG(f"exact layer {self.idx}: scale banks in compressible memory "
                 f"({sum(o.storage.nbytes for o in self.a12) >> 20} MiB)")
        return out

    def hot_ids(self, topk_ids):
        valid = topk_ids >= 0
        rm = self.row_map[topk_ids.clamp_min(0).long()]
        return torch.where(valid, rm, -1).to(topk_ids.dtype)


def install_ctl_injection():
    """Every routed MegaMoE call of an exact layer carries its control block (the drafter's calls are untouched)."""
    from vllm.utils.deep_gemm import _import_deep_gemm
    dg = _import_deep_gemm()
    f = dg.fp8_fp4_mega_moe
    if getattr(f, "_derpseek", False):
        return

    # functools.wraps keeps the original signature visible: vLLM decides whether to fuse the shared expert into
    # MegaMoE by inspecting it (DeepGemmSm100MegaMoEBackend.supports_shared_experts looks for `shared_l1_weights`);
    # a bare (*a, **k) wrapper disabled the fusion, and with it dsv41's router path, on every layer after the first
    @functools.wraps(f)
    def fp8_fp4_mega_moe(y, l1_weights, l2_weights, sym_buffer, *a, _f=f, **k):
        ctl = CTL.get(l1_weights[0].data_ptr())
        if ctl is not None and k.get("cumulative_local_expert_recv_stats") is None:
            k["cumulative_local_expert_recv_stats"] = ctl
        return _f(y, l1_weights, l2_weights, sym_buffer, *a, **k)

    fp8_fp4_mega_moe._derpseek = True
    dg.fp8_fp4_mega_moe = fp8_fp4_mega_moe
    _LOG("MegaMoE calls of exact layers carry their host-tier control blocks")


# ------------------------------------------------------------------------------------------- precision mode
def _parse_layers(spec: str) -> set:
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


# DERPSEEK_MISS=skip: a pick on a host row is dropped for this step instead of read over NVLink-C2C (a miss costs
# precision, never time); it still counts as a host-row pick in the control block, so the residency manager promotes the
# expert for the next steps. Only in DERPSEEK_MISS_LAYERS (default 20-38: layers 0-19 feed the compressed KV and the
# indexer keys of every later token, and layer 39 has no next layer to settle an approximation). wait (default): exact.
MISS_SKIP = os.environ.get("DERPSEEK_MISS", "wait") == "skip"
MISS_LAYERS = _parse_layers(os.environ.get("DERPSEEK_MISS_LAYERS", "20-38"))
_ONES = {}


def skip_host_picks(st: "ExactState", idx: torch.Tensor) -> None:
    """In place on MegaMoE's [T, topk] row indices (int64 or int32): count the host-row picks into this layer's control
    block, then drop them (-1). Fixed shapes only (CUDA-graph safe); no-op outside MISS_LAYERS."""
    if not MISS_SKIP or st.idx not in MISS_LAYERS or st.ctl_buf is None:
        return
    flat = idx.reshape(-1)
    host = (flat >= 0) & (flat < st.D)
    # counts: host picks add 1 at their row; every other pick adds 0 at row D (a static row: harmless)
    tgt = torch.where(host, flat, torch.full_like(flat, st.D)).long()
    counts = st.ctl_buf[HDR:HDR + st.n_rows]
    counts.index_add_(0, tgt, host.to(counts.dtype))
    flat.masked_fill_(host, -1)
