"""Derpseek prompt chunks: prompt staging (DERPSEEK_STAGE, below) and per-layer diagnostics (DERPSEEK_PREFILL_PROF=1:
CUDA events around every eager MegaMoE call, so the log shows where a prompt chunk's time goes per layer: the MoE call
itself, the gap before it (attention and the rest of the layer), and how many distinct host-row experts the call read).
Eager calls only (prompt chunks and mixed steps); graph replays are timed by timing.py."""
import os
import sys
import time
from collections import OrderedDict

import numpy as np

import torch

PROF = os.environ.get("DERPSEEK_PREFILL_PROF", "0") == "1"
PROF_MIN_T = int(os.environ.get("DERPSEEK_PREFILL_PROF_MIN_T", "64"))
_LOG = lambda m: sys.stderr.write(f"DERPSEEK {m}\n")


class _Prof:
    def __init__(self):
        self.pending = []      # (layer, T, ev0, ev1, n_host device scalar, prev ev1 or None)
        self.acc = {}          # (layer, big) -> [calls, moe ms, gap ms, gap calls, host experts]
        self.prev = None       # (layer, ev1) of the previous call in this step
        self.t = time.time()

    def call(self, st, T, hot_ids, fn):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        flat = hot_ids.reshape(-1)
        host = (flat >= 0) & (flat < st.D)
        nh = torch.bincount(torch.where(host, flat, st.D).long(), minlength=st.D + 1)[:st.D].gt(0).sum()
        prev = self.prev[1] if self.prev is not None and self.prev[0] < st.idx else None
        e0.record()
        out = fn()
        e1.record()
        self.prev = (st.idx, e1)
        self.pending.append((st.idx, T, e0, e1, nh, prev))
        self._drain()
        return out

    def _drain(self):
        while self.pending and self.pending[0][3].query():
            l, T, e0, e1, nh, prev = self.pending.pop(0)
            a = self.acc.setdefault((l, T >= 1024), [0, 0.0, 0.0, 0, 0])
            a[0] += 1
            a[1] += e0.elapsed_time(e1)
            if prev is not None:
                a[2] += prev.elapsed_time(e0)
                a[3] += 1
            a[4] += int(nh)
        if time.time() - self.t > 30 and self.acc:
            self.t = time.time()
            for big in (True, False):
                rows = sorted((l, v) for (l, b), v in self.acc.items() if b == big)
                if not rows:
                    continue
                parts = []
                for lo, hi in ((0, 19), (20, 39)):
                    sel = [v for l, v in rows if lo <= l <= hi]
                    n = sum(v[0] for v in sel)
                    if not n:
                        continue
                    moe = sum(v[1] for v in sel) / n * len(sel)
                    gap_n = sum(v[3] for v in sel)
                    gap = sum(v[2] for v in sel) / max(gap_n, 1) * len(sel)
                    nh = sum(v[4] for v in sel) / n
                    parts.append(f"layers {lo}-{hi}: {n // len(sel)} calls/layer, MoE {moe:.2f} ms + gaps {gap:.2f} ms "
                                 f"per pass of {len(sel)} layers, {nh:.1f} host experts per call")
                per = " | ".join(f"{l}:{v[1] / v[0]:.2f}/{v[4] / v[0]:.0f}" for l, v in rows)
                _LOG(f"prefill prof ({'T >= 1024' if big else f'{PROF_MIN_T} <= T < 1024'}): " + "; ".join(parts)
                     + f" || per layer MoE ms / host experts: {per}")
            self.acc = {}


_P = [None]


def prof_call(st, T, hot_ids, fn):
    if not PROF or T < PROF_MIN_T or torch.cuda.is_current_stream_capturing():
        return fn()
    if _P[0] is None:
        _P[0] = _Prof()
    return _P[0].call(st, T, hot_ids, fn)


# ------------------------------------------------------------------------------------------------- prompt staging
# DERPSEEK_STAGE=1: for prompt chunks (T >= DERPSEEK_STAGE_MIN_T) the host-mapped experts of each staged layer
# (DERPSEEK_STAGE_LAYERS, default 0-19: the layers that run on every prompt token) are copied into an HBM staging slot
# on a copy stream a few layers ahead, and that layer's MegaMoE call reads them from there: its control block's host
# bases point at the slot for the call (host lane off), and the picks of host-mapped experts go to their staging
# positions. A copy engine moves the same bytes the kernel would have read over NVLink-C2C, but while the layers
# before it compute. Exact: the same rows of the same weights, only read from HBM (tests/stage_test.py). The call's
# per-row counts are credited back to the true host rows, so the residency manager keeps learning from prompts. When the
# residency manager has moved a few of a slot's experts since its copy (layers 0-1 are copied during the previous
# chunk), the missing rows are topped up into the slot's spare rows; only if they do not fit does the call read host
# rows as before. The slots are reserved at load time (reserve()), so the KV cache is sized around them.
STAGE = os.environ.get("DERPSEEK_STAGE", "0") == "1"
STAGE_MIN_T = int(os.environ.get("DERPSEEK_STAGE_MIN_T", "1024"))
STAGE_SLOTS = int(os.environ.get("DERPSEEK_STAGE_SLOTS", "2"))
STAGE_SLACK = int(os.environ.get("DERPSEEK_STAGE_SLACK", "4"))     # rows beyond D - C (cache rows still filling)
# a layer is staged only when the last prompt chunk read at least this many distinct host-mapped experts there: once the
# residency manager has the prompt's experts in cache rows (a repetitive prompt), the few left cost less in-kernel than
# a 1.2 GB copy competing for the link
STAGE_MIN_HOST = int(os.environ.get("DERPSEEK_STAGE_MIN_HOST", "8"))


STAGE_ENV = os.environ.get("DERPSEEK_STAGE", "0") in ("1", "live")
_RESERVED = {}


def _alloc_slots(layers, dev):
    from . import exact
    st0 = exact.LAYERS[layers[0]]
    U = max(exact.LAYERS[l].D - exact.LAYERS[l].C for l in layers) + STAGE_SLACK
    return U, [[torch.empty_strided((U,) + tuple(h.shape[1:]), h.stride(), dtype=h.dtype, device=dev)
                for h in st0.host_t] for _ in range(STAGE_SLOTS)]


def reserve(st):
    """End of an exact layer's finalize (load time): once the last staged layer is in, allocate the staging slots, so
    vLLM's memory profile counts them and sizes the KV cache around them (a prompt chunk's attention needs the memory
    vLLM leaves unused at runtime; slots taken from it later ran a 32K prompt out of memory)."""
    if not STAGE_ENV or _RESERVED:
        return
    from . import exact
    want = exact._parse_layers(os.environ.get("DERPSEEK_STAGE_LAYERS", "0-19"))
    if st.idx != max(want):
        return
    layers = [l for l, s in sorted(exact.LAYERS.items()) if l in want and s.D > s.C]
    if not layers:
        return
    _RESERVED["layers"] = layers
    _RESERVED["U"], _RESERVED["bufs"] = _alloc_slots(layers, st.hbm[0].device)
    mb = sum(b.numel() * b.element_size() for bufs in _RESERVED["bufs"] for b in bufs) >> 20
    _LOG(f"prompt staging: {STAGE_SLOTS} slots of {_RESERVED['U']} rows reserved at load ({mb} MiB HBM)")


class _Slot:
    def __init__(self, bufs, words, dmax):
        self.bufs = bufs                    # 4 HBM tensors laid out like the host tensors, U rows each
        self.flat = [torch.as_strided(b, (b.shape[0], b[0].numel()), (b[0].numel(), 1)) for b in bufs]
        self.words = words                  # device int32 [9]: control words 2..10 (the four bases, host lane 0)
        self.layer = -1                     # staged layer, -1: none
        self.pos_np = None                  # host row -> staging position (-1: not staged), host copy
        self.pos = torch.full((dmax,), -1, dtype=torch.int32, device=bufs[0].device)   # the same on the device
        self.rows = torch.zeros(bufs[0].shape[0], dtype=torch.int64, device=bufs[0].device)  # staging position -> host row
        self.n = 0                          # staged rows
        self.ready = None                   # copy stream: the rows have landed
        self.free = None                    # main stream: the last call reading this slot is done


class Stager:
    def __init__(self):
        from . import exact
        self.exact = exact
        want = exact._parse_layers(os.environ.get("DERPSEEK_STAGE_LAYERS", "0-19"))
        self.layers = [l for l, st in sorted(exact.LAYERS.items()) if l in want and st.D > st.C]
        st0 = exact.LAYERS[self.layers[0]]
        self.dev = st0.row_map.device
        if _RESERVED.get("layers") == self.layers:
            self.U, slot_bufs = _RESERVED["U"], _RESERVED["bufs"]
        else:
            self.U, slot_bufs = _alloc_slots(self.layers, self.dev)
        self.Dmax = max(exact.LAYERS[l].D for l in self.layers)
        ctl = exact.POOL["ctl"]
        self.orig = {l: ctl[l, 2:11].clone() for l in self.layers}
        self.stream = torch.cuda.Stream(device=self.dev)
        self.slots = []
        for bufs in slot_bufs:
            w = torch.zeros(9, dtype=torch.int32)
            w[:8].view(torch.int64)[:] = torch.tensor([b.data_ptr() for b in bufs], dtype=torch.int64)
            self.slots.append(_Slot(bufs, w.to(self.dev), self.Dmax))
        self.next_of = {l: self.layers[(i + STAGE_SLOTS) % len(self.layers)] for i, l in enumerate(self.layers)}
        self.slot_of = {l: i % STAGE_SLOTS for i, l in enumerate(self.layers)}
        self.row_bytes = sum(f[0].numel() * f.element_size() for f in st0.flat_host)
        self.comp_bytes = [f[0].numel() * f.element_size() for f in st0.flat_host]
        # one cudaMemcpyBatchAsync per layer (0.14 ms of CPU and ~350 GB/s for 256 row copies, against 2.2 ms and ~290
        # GB/s as single copies); plans are cached per (layer, slot, rows) and never modified, so a plan's pinned
        # position table is never rewritten under a copy in flight
        from cuda.bindings import runtime as rt
        self.rt = rt
        self.attr = rt.cudaMemcpyAttributes()
        self.attr.srcAccessOrder = rt.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderStream
        self.plans = OrderedDict()
        self.timed, self.waits, self.keep = [], [], []
        self.stats = {"staged": 0, "fallback_none": 0, "fallback_stale": 0, "patched": 0, "issued": 0, "skipped": 0,
                      "issue_ms": 0.0,
                      "copy_ms": 0.0, "copy_gb": 0.0, "wait_ms": 0.0, "t": time.time()}
        mb = sum(b.numel() * b.element_size() for s in self.slots for b in s.bufs) >> 20
        _LOG(f"prompt staging: layers {self.layers[0]}-{self.layers[-1]} ({len(self.layers)}), {STAGE_SLOTS} slots of "
             f"{self.U} rows ({mb} MiB HBM), prompt chunks of >= {STAGE_MIN_T} tokens")

    def _host_mapped(self, l):
        """Host rows of layer l whose expert the row map sends to the host row now (in stream order)."""
        st = self.exact.LAYERS[l]
        from . import residency
        R = residency._R[0]
        if R is not None:
            return np.nonzero(~R.mapped_in[l, :st.D])[0]
        cached = set(st.slot)
        return np.array([j for j, e in enumerate(st.host) if e not in cached], dtype=np.int64)

    def _issue(self, l):
        """Copy layer l's host-mapped rows into its slot on the copy stream (after the slot's last reader)."""
        t0 = time.perf_counter()
        slot = self.slots[self.slot_of[l]]
        slot.layer = -1
        from . import residency
        R = residency._R[0]
        if R is not None and R.chunk_nhost is not None and R.chunk_nhost[l] < STAGE_MIN_HOST:
            self.stats["skipped"] += 1     # the last prompt chunk needed few host experts here: read them in-kernel
            return
        rows = self._host_mapped(l)
        if len(rows) > self.U:
            return
        dsts, srcs, sizes, pos, _, _ = self._plan(l, self.slot_of[l], rows)
        if slot.free is not None:
            self.stream.wait_event(slot.free)
        t_ev = torch.cuda.Event(enable_timing=True)
        t_ev.record(self.stream)
        r = self.rt.cudaMemcpyBatchAsync(dsts, srcs, sizes, len(dsts), [self.attr], [0], 1, self.stream.cuda_stream)
        if r[0] != self.rt.cudaError_t.cudaSuccess:
            raise RuntimeError(f"DERPSEEK prompt staging: cudaMemcpyBatchAsync failed: {r}")
        slot.ready = torch.cuda.Event(enable_timing=True)
        slot.ready.record(self.stream)
        self.timed.append((t_ev, slot.ready, len(rows) * self.row_bytes))
        while self.timed and self.timed[0][1].query():
            a, b, nb = self.timed.pop(0)
            self.stats["copy_ms"] += a.elapsed_time(b)
            self.stats["copy_gb"] += nb / 1e9
        slot.pos_np, slot.layer, slot.n = pos, l, len(rows)
        self.stats["issued"] += 1
        self.stats["issue_ms"] += (time.perf_counter() - t0) * 1e3

    def _patch(self, l, slot, missing, rows_now):
        """Copy the missing host-mapped rows into the slot: first into the positions of staged rows whose experts the
        residency manager has mapped into cache rows since (nothing routes to those positions now), then into spare
        rows. False when they do not fit."""
        st = self.exact.LAYERS[l]
        pos = slot.pos_np.copy()             # plans keep their own arrays: never modify one in place
        in_slot = np.nonzero(pos >= 0)[0]
        gone = in_slot[~np.isin(in_slot, rows_now)]
        free = pos[gone].tolist()
        pos[gone] = -1
        extra = len(missing) - len(free)
        targets = free[:len(missing)] + list(range(slot.n, slot.n + max(extra, 0)))
        if targets and max(targets) >= self.U:
            return False
        pos[missing] = np.array(targets, dtype=np.int32)
        pin_pos = torch.from_numpy(pos).pin_memory()
        pin_rows = torch.from_numpy(missing.astype(np.int64)).pin_memory()
        dsts, srcs, sizes = [], [], []
        for i, (j, u) in enumerate(zip(missing.tolist(), targets)):
            for k in range(4):
                dsts.append(slot.flat[k][u].data_ptr())
                srcs.append(st.flat_host[k][j].data_ptr())
                sizes.append(self.comp_bytes[k])
            dsts.append(slot.rows[u].data_ptr())          # staging position -> host row, for the count credit
            srcs.append(pin_rows[i].data_ptr())
            sizes.append(8)
        dsts.append(slot.pos.data_ptr())
        srcs.append(pin_pos.data_ptr())
        sizes.append(pos.nbytes)
        r = self.rt.cudaMemcpyBatchAsync(dsts, srcs, sizes, len(dsts), [self.attr], [0], 1, self.stream.cuda_stream)
        if r[0] != self.rt.cudaError_t.cudaSuccess:
            raise RuntimeError(f"DERPSEEK prompt staging: cudaMemcpyBatchAsync failed: {r}")
        slot.ready = torch.cuda.Event()
        slot.ready.record(self.stream)      # after the slot's earlier copies (same stream)
        self.keep.append((pin_pos, pin_rows))
        if len(self.keep) > 64:
            self.keep.pop(0)
        slot.pos_np, slot.n = pos, max([slot.n] + [u + 1 for u in targets])
        return True

    def _plan(self, l, si, rows):
        """Copy descriptors for layer l's rows into slot si: one per run of consecutive host rows and tensor, then the
        position table (from its own pinned buffer)."""
        key = (l, si, rows.tobytes())
        p = self.plans.get(key)
        if p is not None:
            self.plans.move_to_end(key)
            return p
        st, slot = self.exact.LAYERS[l], self.slots[si]
        pos = np.full(self.Dmax, -1, dtype=np.int32)
        pos[rows] = np.arange(len(rows), dtype=np.int32)
        pin = torch.from_numpy(pos).pin_memory()
        dsts, srcs, sizes = [], [], []
        u = 0
        for run in np.split(rows, np.nonzero(np.diff(rows) != 1)[0] + 1):
            if not len(run):
                continue
            a, n = int(run[0]), len(run)
            for k in range(4):
                dsts.append(slot.flat[k][u].data_ptr())
                srcs.append(st.flat_host[k][a].data_ptr())
                sizes.append(n * self.comp_bytes[k])
            u += n
        dsts.append(slot.pos.data_ptr())
        srcs.append(pin.data_ptr())
        sizes.append(pos.nbytes)
        rpin = torch.from_numpy(rows.astype(np.int64)).pin_memory()
        dsts.append(slot.rows.data_ptr())
        srcs.append(rpin.data_ptr())
        sizes.append(rpin.numel() * 8)
        p = self.plans[key] = (dsts, srcs, sizes, pos, pin, rpin)
        if len(self.plans) > 512:
            self.plans.popitem(last=False)
        return p

    def call(self, st, topk_ids, fn):
        """A prompt chunk's call of staged layer st.idx: returns MegaMoE's output, or None to run the normal path."""
        l = st.idx
        if l not in self.slot_of:
            return None
        slot = self.slots[self.slot_of[l]]
        out = None
        ok = slot.layer == l
        if not ok:
            self.stats["fallback_none"] += 1
        else:
            rows = self._host_mapped(l)
            missing = rows[slot.pos_np[rows] < 0]
            if len(missing):
                # the residency manager moved experts since the copy (layers 0-1 are copied during the previous chunk):
                # top up the few it evicted instead of dropping the slot
                ok = self._patch(l, slot, missing, rows)
                self.stats["patched" if ok else "fallback_stale"] += 1
        if ok:
            cur = torch.cuda.current_stream()
            w0, w1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            w0.record(cur)
            cur.wait_event(slot.ready)
            w1.record(cur)
            self.waits.append((w0, w1))
            rm = st.row_map
            rs = torch.where(rm < st.D, slot.pos[rm.clamp(0, st.D - 1).long()], rm)
            valid = topk_ids >= 0
            r = rs[topk_ids.clamp_min(0).long()]
            hot_ids = torch.where(valid & (r >= 0), r, -1).to(topk_ids.dtype)
            ctl = self.exact.POOL["ctl"][l]
            cnt = ctl[self.exact.HDR:self.exact.HDR + st.n_rows]
            n = slot.n
            before = cnt[:n].clone()
            ctl[2:11].copy_(slot.words)
            out = prof_call(st, topk_ids.shape[0], hot_ids, lambda: fn(hot_ids))
            ctl[2:11].copy_(self.orig[l])
            # the call counted its staged picks on rows [0, n): credit them to their host rows (the residency manager
            # reads per-row counts) and give rows [0, n) back their own counts
            d = cnt[:n] - before
            cnt[:n].copy_(before)
            cnt.index_add_(0, slot.rows[:n], d)
            slot.free = torch.cuda.Event()
            slot.free.record(cur)
            self.stats["staged"] += 1
        slot.layer = -1
        if out is None:
            # the normal call follows in stream order; its slot is free once it is enqueued after this point
            slot.free = torch.cuda.Event()
            slot.free.record(torch.cuda.current_stream())
        self._issue(self.next_of[l])
        while self.waits and self.waits[0][1].query():
            a, b = self.waits.pop(0)
            self.stats["wait_ms"] += a.elapsed_time(b)
        if time.time() - self.stats["t"] > 30:
            s = self.stats
            _LOG(f"prompt staging: {s['staged']} staged calls ({s['wait_ms'] / max(s['staged'], 1):.3f} ms waiting for "
                 f"the copy each), {s['fallback_none']} without a slot ({s['skipped']} skipped: few host experts), "
                 f"{s['fallback_stale']} with a stale slot ({s['patched']} topped up); {s['issued']} copies issued, "
                 f"{s['issue_ms'] / max(s['issued'], 1):.2f} ms CPU each; copies {s['copy_gb']:.1f} GB in "
                 f"{s['copy_ms']:.0f} ms ({s['copy_gb'] / max(s['copy_ms'], 1e-9) * 1e3:.0f} GB/s while copying)")
            self.stats = {k: 0 for k in s}
            self.stats.update(issue_ms=0.0, copy_ms=0.0, copy_gb=0.0, wait_ms=0.0, t=time.time())
        return out


_S = [None]


def staged_call(st, topk_ids, fn):
    if torch.cuda.is_current_stream_capturing():
        return None
    if _S[0] is None:
        _S[0] = Stager()
    return _S[0].call(st, topk_ids, fn)
