"""Derpseek exact mode: the residency manager. Host-row experts that get picks are promoted into HBM cache rows
(least recently used slot out), between steps, without ever stalling a step or exposing a half-copied row.

Each tick (after a step's sample_tokens, on the engine thread):
  1. Copies that have landed (event query, no wait) are mapped in: row_map[e] = cache row (main stream).
  2. Every count copy that has landed (one per step, read in order): per-row pick counts of every layer (MegaMoE
     writes them into the control blocks, exact.py) give the host rows that missed and the cache rows that hit in
     that step.
  3. Host rows with recent misses (at least `min_misses` within about `window` steps, most first, up to `promote`
     per layer and `promote_tick` per tick): take the least recently used cache slot that is not filling and is at
     least `min_age` ticks old, remap its expert back to its host row (main stream, between steps: later steps read the
     exact host copy), then copy the new expert's rows host -> slot on a copy stream that waits for that remap (no
     step reads the slot meanwhile: nothing maps to it).
  4. Start this step's count copy (device -> pinned host ring, ordered after this step's kernels).
Weights are read-only, so eviction never writes back. DERPSEEK_RESIDENCY=0 disables promotion (static cache). The
tunables in TUNE can be changed while serving through DERPSEEK_TUNE_FILE (JSON, re-read every 5 s)."""
import json
import os
import sys
import time

import numpy as np
import torch

from . import exact

TUNE = {
    "promote": int(os.environ.get("DERPSEEK_PROMOTE", "4")),            # promotions per layer per tick
    "promote_tick": int(os.environ.get("DERPSEEK_PROMOTE_TICK", "24")), # promotions per tick over all layers
    "min_age": int(os.environ.get("DERPSEEK_MIN_AGE", "8")),            # ticks a fresh slot is protected from eviction
    "min_misses": float(os.environ.get("DERPSEEK_MIN_MISSES", "1")),    # decayed misses before a promotion
    "window": float(os.environ.get("DERPSEEK_MISS_WINDOW", "64")),      # decay time constant of the miss counts (steps)
    "enabled": 1,
    "fence": 1,         # A/B only: 0 lifts dsv41 #10's copy fence (solo.FENCE)
}
TUNE_FILE = os.environ.get("DERPSEEK_TUNE_FILE", "/root/.cache/vllm/derpseek-tune.json")
RING = 8      # count copies in flight: one per tick, so every read covers exactly one step
# measured MegaMoE call time added by k distinct host-row experts in one call (us; T=8, host-first rows, host lane of
# 4 clusters, tests/step_cost_test.py): the first is hidden, then the link (at T=32 the first three are hidden)
_MISS_US = np.array([0, 0, 22, 86, 151, 216, 282, 352, 422], dtype=np.float64)
_LOG = lambda m: sys.stderr.write(f"DERPSEEK {m}\n")


class Residency:
    def __init__(self):
        self.layers = dict(sorted(exact.LAYERS.items()))
        self.idx = np.array(list(self.layers), dtype=np.int64)       # managed layer indices
        st0 = next(iter(self.layers.values()))
        if any((st.D, st.S, st.C) != (st0.D, st0.S, st0.C) for st in self.layers.values()):
            raise RuntimeError("DERPSEEK residency: every exact layer needs the same D, S and C")
        self.D, self.S, self.C, self.rows = st0.D, st0.S, st0.C, st0.n_rows
        self.dev = st0.row_map.device
        self.ctl = exact.POOL["ctl"]                       # [40, HDR + rows]
        self.row_map = exact.POOL["row_map"]               # [40, E]
        L, D, C = self.ctl.shape[0], self.D, self.C
        self.host_e = np.full((L, D), -1, dtype=np.int64)  # expert of each host row
        self.slot_e = np.full((L, C), -1, dtype=np.int64)  # expert held by each cache slot (-1: empty / filling)
        self.slot_j = np.full((L, C), -1, dtype=np.int64)  # its host row
        self.held = np.zeros((L, D), dtype=bool)           # host row is cached or filling
        for l, st in self.layers.items():
            self.host_e[l] = st.host
            for s, e in enumerate(st.slot):
                self.slot_e[l, s], self.slot_j[l, s] = e, st.host_row[e]
                self.held[l, st.host_row[e]] = True
        self.managed = np.zeros(L, dtype=bool)
        self.managed[self.idx] = True
        self.slot_used = np.zeros((L, C), dtype=np.int64)
        self.slot_born = np.full((L, C), -10**9, dtype=np.int64)
        self.busy = np.zeros((L, C), dtype=bool)
        self.recent = np.zeros((L, D), dtype=np.float32)   # decayed misses per host row
        self.fills = []                                    # (event, [(layer, slot, expert, host row)]) in issue order
        self.cnt_ring = [torch.zeros(L, self.rows, dtype=torch.int32, pin_memory=True) for _ in range(RING)]
        self.cnt_q = []                                    # ring slots with a copy in flight, issue order
        self.cnt_ev = [None] * RING
        self.cnt_next = 0
        self.last = None
        self.copy_stream = torch.cuda.Stream(device=self.dev)
        self.tick_n = 0
        # pinned staging for row-map updates (index into the flat [40 * E] row map, value): a ring, so an update never
        # waits for the previous one (which sits behind the step the GPU is still running)
        self.ring = [(torch.zeros(4096, dtype=torch.long, pin_memory=True), torch.zeros(4096, dtype=torch.int32, pin_memory=True))
                     for _ in range(16)]
        self.ring_ev = [None] * len(self.ring)
        self.ring_i = 0
        self.tune_t, self.tune_mtime = 0.0, None
        self._reset_stats()
        _LOG(f"residency manager: {len(self.layers)} layers, {C} cache rows each, tunables {TUNE}")

    def _reset_stats(self):
        self.stats = {"ticks": 0, "promoted": 0, "missed_picks": 0, "hit_picks": 0, "t": time.time(), "cpu": 0.0,
                      "cpu_max": 0.0, "reads": 0, "hist": np.zeros(9, dtype=np.int64), "miss_us": 0.0}

    # ------------------------------------------------------------------------------------------------ tunables
    def _tune(self):
        now = time.time()
        if now - self.tune_t < 5:
            return
        self.tune_t = now
        try:
            m = os.path.getmtime(TUNE_FILE)
        except OSError:
            return
        if m == self.tune_mtime:
            return
        self.tune_mtime = m
        try:
            new = json.load(open(TUNE_FILE))
            for k, v in new.items():
                if k in TUNE:
                    TUNE[k] = type(TUNE[k])(v)
            from . import solo
            solo.FENCE[0] = bool(TUNE["fence"])
            _LOG(f"residency tunables now {TUNE}")
        except Exception as e:  # a half-written file: keep the old values
            _LOG(f"residency tune file unreadable ({e}); keeping {TUNE}")

    # ------------------------------------------------------------------------------------------- row-map updates
    def _remap(self, updates):
        """[(layer, expert, row)] -> one stream-ordered copy into the flat row map (main stream)."""
        if not updates:
            return
        k = self.ring_i
        self.ring_i = (k + 1) % len(self.ring)
        if self.ring_ev[k] is not None:
            self.ring_ev[k].synchronize()    # 16 updates ago: long done
        si, sv = self.ring[k]
        n = len(updates)
        assert n <= si.numel()
        si.numpy()[:n] = [l * exact.E + e for l, e, _ in updates]
        sv.numpy()[:n] = [r for _, _, r in updates]
        flat = self.row_map.view(-1)
        flat.index_put_((si[:n].to(self.dev, non_blocking=True),), sv[:n].to(self.dev, non_blocking=True))
        self.ring_ev[k] = torch.cuda.Event()
        self.ring_ev[k].record()

    # ------------------------------------------------------------------------------------------------ the tick
    def tick(self):
        t0 = time.perf_counter()
        self._tune()
        self._tick()
        dt = time.perf_counter() - t0
        self.stats["cpu"] += dt
        self.stats["cpu_max"] = max(self.stats["cpu_max"], dt)
        self.stats["ticks"] += 1
        if time.time() - self.stats["t"] > 30:
            st = self.stats
            hp, mp, n = st["hit_picks"], st["missed_picks"], max(st["reads"], 1)
            _LOG(f"residency: {st['ticks']} ticks, {st['reads']} step reads, {st['promoted']} promotions, cache-row picks "
                 f"{hp}, host-row picks {mp} (misses {mp / max(hp + mp, 1) * 100:.1f}% of non-static picks, "
                 f"{mp / n:.1f} per step), {int(self.busy.sum())} copies in flight, tick CPU "
                 f"{st['cpu'] / max(st['ticks'], 1) * 1e3:.3f} ms mean / {st['cpu_max'] * 1e3:.2f} max; distinct host "
                 f"experts per layer-step 0..8+: {'/'.join(str(int(x)) for x in st['hist'])}, est. miss time "
                 f"{st['miss_us'] / n / 1e3:.3f} ms per step")
            self._reset_stats()

    def _tick(self):
        self.tick_n += 1
        t = self.tick_n
        D, S, C = self.D, self.S, self.C
        # 1. map in landed copies (fills complete in issue order on the copy stream)
        ups = []
        while self.fills and self.fills[0][0].query():
            _, items = self.fills.pop(0)
            for l, s, e, j in items:
                self.slot_e[l, s], self.slot_j[l, s] = e, j
                self.busy[l, s] = False
                ups.append((l, e, D + S + s))
        self._remap(ups)
        # 2. every landed count copy: one step's per-row picks
        miss_now = None
        while self.cnt_q and self.cnt_ev[self.cnt_q[0]].query():
            k = self.cnt_q.pop(0)
            cnt = self.cnt_ring[k].numpy().view(np.uint32).copy()
            if self.last is None:
                self.last = cnt
                continue
            delta = (cnt - self.last).astype(np.int64)        # uint32 subtraction wraps correctly
            self.last = cnt
            delta[~self.managed] = 0
            miss = delta[:, :D]
            hits = delta[:, D + S:D + S + C]
            self.slot_used[hits > 0] = t
            st = self.stats
            st["reads"] += 1
            st["hit_picks"] += int(hits.sum())
            st["missed_picks"] += int(miss.sum())
            kk = np.count_nonzero(miss[self.idx], axis=1)
            st["hist"] += np.bincount(np.minimum(kk, 8), minlength=9)
            st["miss_us"] += float(_MISS_US[np.minimum(kk, 8)].sum() + 70.0 * np.maximum(kk - 8, 0).sum())
            self.recent *= np.float32(1.0 - 1.0 / max(TUNE["window"], 1.0))
            self.recent += miss
            miss_now = miss if miss_now is None else miss_now + miss
        # 3. promotions
        if miss_now is not None and TUNE["enabled"]:
            self._promote(miss_now, t)
        # 4. this step's count copy, after its kernels
        if len(self.cnt_q) < RING:
            k = self.cnt_next
            self.cnt_next = (k + 1) % RING
            ev = torch.cuda.Event()
            ev.record()
            self.copy_stream.wait_event(ev)
            with torch.cuda.stream(self.copy_stream):
                self.cnt_ring[k].copy_(self.ctl[:, exact.HDR:], non_blocking=True)
                self.cnt_ev[k] = torch.cuda.Event()
                self.cnt_ev[k].record(self.copy_stream)
            self.cnt_q.append(k)

    def _promote(self, miss_now, t):
        cand = (miss_now > 0) & (self.recent >= TUNE["min_misses"]) & ~self.held
        if not cand.any():
            return
        ls, js = np.nonzero(cand)
        order = np.argsort(-self.recent[ls, js], kind="stable")
        per_layer = {}
        n = 0
        for i in order:
            lst = per_layer.setdefault(int(ls[i]), [])
            if len(lst) >= TUNE["promote"]:
                continue
            lst.append(int(js[i]))
            n += 1
            if n >= TUNE["promote_tick"]:
                break
        evict, copies = [], []
        for l, js_l in per_layer.items():
            ok = (~self.busy[l]) & (t - self.slot_born[l] >= TUNE["min_age"])
            free = [int(s) for s in np.argsort(self.slot_used[l], kind="stable") if ok[s]]   # least recently used first
            for j, s in zip(js_l, free):
                e_old, j_old = int(self.slot_e[l, s]), int(self.slot_j[l, s])
                if e_old >= 0:
                    evict.append((l, e_old, j_old))            # back to its host row
                    self.held[l, j_old] = False
                self.slot_e[l, s], self.slot_j[l, s] = -1, -1
                self.busy[l, s] = True
                self.held[l, j] = True
                self.slot_born[l, s] = self.slot_used[l, s] = t
                copies.append((l, s, int(self.host_e[l, j]), j))
        if not copies:
            return
        # remap the evicted experts to their host rows (main stream, before any later step), then copy the new experts
        # into their slots on the copy stream once that remap is ordered (no step reads those slots now)
        self._remap(evict)
        ev = torch.cuda.Event()
        ev.record()
        self.copy_stream.wait_event(ev)
        with torch.cuda.stream(self.copy_stream):
            for l, s, e, j in copies:
                st = self.layers[l]
                for k, (hb, ho) in enumerate(zip(st.flat_hbm, st.flat_host)):
                    hb[st.S + s].copy_(ho[j], non_blocking=True)     # HBM tensor row of cache slot s
                    if k in (1, 3) and getattr(st, "a12", None):
                        hb[st.S + s].bitwise_xor_(0)                 # SM rewrite: store the scale row compressed
            done = torch.cuda.Event()
            done.record(self.copy_stream)
        self.fills.append((done, copies))
        self.stats["promoted"] += len(copies)


_R = [None]


def tick():
    if os.environ.get("DERPSEEK_RESIDENCY", "1") != "1" or not exact.LAYERS or torch.cuda.is_current_stream_capturing():
        return
    if _R[0] is None:
        _R[0] = Residency()
    _R[0].tick()
