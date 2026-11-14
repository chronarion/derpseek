"""Derpseek exact mode: the residency manager. Host-row experts that get picks are promoted into HBM cache rows
(least recently used slot out), between steps, without ever stalling a step or exposing a half-copied row.

Each tick (after a step's sample_tokens, on the engine thread):
  1. When the previous tick's count copy has landed: per-row pick counts of every layer (MegaMoE writes them into the
     control blocks, exact.py) give the host rows that missed and the cache rows that hit since the last read.
  2. For each missed expert (most picks first, up to DERPSEEK_PROMOTE per layer per tick): take the least recently used
     cache slot that is not still filling, remap its expert back to its host row (main stream, between steps: later
     steps read the exact host copy), then copy the new expert's rows host -> slot on a copy stream that waits for that
     remap (no step reads the slot meanwhile: nothing maps to it).
  3. Copies that have completed (event query, no wait) get mapped in: row_map[e] = slot row (main stream).
  4. Start the next count copy (device -> pinned host, ordered after this step's kernels).
Weights are read-only, so eviction never writes back. DERPSEEK_RESIDENCY=0 disables promotion (static cache)."""
import os
import sys
import time

import numpy as np
import torch

from . import exact

PROMOTE = int(os.environ.get("DERPSEEK_PROMOTE", "4"))       # promotions per layer per tick
PROMOTE_TICK = int(os.environ.get("DERPSEEK_PROMOTE_TICK", "24"))   # promotions per tick over all layers (most misses first)
MIN_AGE = int(os.environ.get("DERPSEEK_MIN_AGE", "8"))        # ticks a fresh slot is protected from eviction
# measured MegaMoE call time added by k distinct host-row experts in one call (us; T=8, host-first rows, host lane of
# 4 clusters, mmtest/step_cost_test.py): the first is hidden, then the link (at T=32 the first three are hidden)
_MISS_US = (0, 0, 22, 86, 151, 216, 282)
def _miss_us(k):
    return _MISS_US[k] if k < len(_MISS_US) else _MISS_US[-1] + 70 * (k - len(_MISS_US) + 1)
_LOG = lambda m: sys.stderr.write(f"DERPSEEK {m}\n")


class Residency:
    def __init__(self):
        self.layers = dict(sorted(exact.LAYERS.items()))
        self.L = len(self.layers)
        st0 = next(iter(self.layers.values()))
        self.dev = st0.row_map.device
        self.ctl = exact.POOL["ctl"]                       # [40, HDR + rows]
        self.row_map = exact.POOL["row_map"]               # [40, E]
        self.rows = st0.n_rows
        self.cnt_host = torch.zeros(self.ctl.shape[0], self.rows, dtype=torch.int32, pin_memory=True)
        self.last = np.zeros((self.ctl.shape[0], self.rows), dtype=np.uint32)   # cumulative int32 counts wrap: deltas mod 2^32
        self.cnt_ev = None
        self.cnt_span = 0          # steps (ticks) the pending count copy covers
        self.last_issue = 0
        self.hist = np.zeros(9, dtype=np.int64)    # distinct host experts per layer per single-step count read
        self.miss_us = 0.0
        self.miss_steps = 0
        self.copy_stream = torch.cuda.Stream(device=self.dev)
        self.tick_n = 0
        self.slot_used = {i: np.zeros(st.C, dtype=np.int64) for i, st in self.layers.items()}      # last tick hit
        self.slot_born = {i: np.full(st.C, -10**9, dtype=np.int64) for i, st in self.layers.items()}
        self.filling = {}       # (layer, slot) -> (expert, event): copies in flight
        # pinned staging for row-map updates (index into the flat [40 * E] row map, value): a ring, so an update never
        # waits for the previous one (which sits behind the step the GPU is still running)
        self.ring = [(torch.zeros(4096, dtype=torch.long, pin_memory=True), torch.zeros(4096, dtype=torch.int32, pin_memory=True))
                     for _ in range(16)]
        self.ring_ev = [None] * len(self.ring)
        self.ring_i = 0
        self.stats = {"ticks": 0, "promoted": 0, "missed_picks": 0, "hit_picks": 0, "t": time.time(), "cpu": 0.0, "cpu_max": 0.0}
        _LOG(f"residency manager: {self.L} layers, {st0.C} cache rows each, up to {PROMOTE} promotions per layer per tick")

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
        self._tick()
        dt = time.perf_counter() - t0
        self.stats["cpu"] += dt
        self.stats["cpu_max"] = max(self.stats["cpu_max"], dt)

    def _tick(self):
        self.tick_n += 1
        t = self.tick_n
        # 3. map in finished copies
        done = [(k, v) for k, v in self.filling.items() if v[1].query()]
        if done:
            ups = []
            for (l, s), (e, _) in done:
                st = self.layers[l]
                st.slot[s] = e
                ups.append((l, e, st.D + st.S + s))
                del self.filling[(l, s)]
            self._remap(ups)
        # 1. read the counts the last copy brought
        if self.cnt_ev is not None and self.cnt_ev.query():
            cnt = self.cnt_host.numpy().view(np.uint32).copy()
            delta = (cnt - self.last).astype(np.int64)        # uint32 subtraction wraps correctly
            self.last = cnt
            self.cnt_ev = None
            evict, copies, cands = [], [], []
            active = np.nonzero(delta.any(axis=1))[0]
            if self.cnt_span == 1:     # one step's worth of counts: per-layer distinct host experts
                ks = [int(np.count_nonzero(delta[l, 0:st.D])) for l, st in self.layers.items()]
                for k in ks:
                    self.hist[min(k, 8)] += 1
                self.miss_us += sum(_miss_us(k) for k in ks)
                self.miss_steps += 1
            for l in active.tolist():
                st = self.layers.get(l)
                if st is None:
                    continue
                d = delta[l]
                cache = d[st.D + st.S:st.n_rows]            # kernel rows of the cache slots
                hit_slots = np.nonzero(cache > 0)[0]
                self.slot_used[l][hit_slots] = t
                self.stats["hit_picks"] += int(cache.sum())
                miss = d[0:st.D]                              # host rows
                self.stats["missed_picks"] += int(miss.sum())
                cand = np.nonzero(miss > 0)[0]
                if not len(cand):
                    continue
                held = {e for e in st.slot if e is not None} | {v[0] for (ll, _), v in self.filling.items() if ll == l}
                cand = [int(j) for j in cand[np.argsort(-miss[cand], kind="stable")] if st.host[int(j)] not in held][:PROMOTE]
                cands += [(int(miss[j]), l, j) for j in cand]
            cands.sort(key=lambda x: -x[0])
            per_layer = {}
            for _, l, j in cands[:PROMOTE_TICK]:
                per_layer.setdefault(l, []).append(j)
            for l, cand in per_layer.items():
                st = self.layers[l]
                busy = {s for (ll, s) in self.filling if ll == l}
                order = np.argsort(self.slot_used[l], kind="stable")          # least recently used first
                free = [int(s) for s in order if int(s) not in busy and t - self.slot_born[l][s] >= MIN_AGE]
                for j, s in zip(cand, free):
                    e_new = st.host[int(j)]
                    e_old = st.slot[s]
                    if e_old is not None:
                        evict.append((l, e_old, st.host_row[e_old]))
                    st.slot[s] = None
                    self.slot_born[l][s] = t
                    self.slot_used[l][s] = t
                    copies.append((l, s, e_new, int(j)))
            if copies:
                # 2a. remap the evicted experts to their host rows (main stream, before any later step)
                self._remap(evict)
                ev = torch.cuda.Event()
                ev.record()
                # 2b. copy the new experts into their slots once the remap is ordered (no step reads those slots now)
                self.copy_stream.wait_event(ev)
                with torch.cuda.stream(self.copy_stream):
                    for l, s, e_new, j in copies:
                        st = self.layers[l]
                        for hb, ho in zip(st.flat_hbm, st.flat_host):
                            hb[st.S + s].copy_(ho[j], non_blocking=True)
                        cev = torch.cuda.Event()
                        cev.record(self.copy_stream)
                        self.filling[(l, s)] = (e_new, cev)
                self.stats["promoted"] += len(copies)
        # 4. next count copy, after this step's kernels
        if self.cnt_ev is None:
            self.cnt_span = t - self.last_issue
            self.last_issue = t
            ev = torch.cuda.Event()
            ev.record()
            self.copy_stream.wait_event(ev)
            with torch.cuda.stream(self.copy_stream):
                self.cnt_host.copy_(self.ctl[:, exact.HDR:], non_blocking=True)
                self.cnt_ev = torch.cuda.Event()
                self.cnt_ev.record(self.copy_stream)
        self.stats["ticks"] += 1
        if time.time() - self.stats["t"] > 30:
            hp, mp = self.stats["hit_picks"], self.stats["missed_picks"]
            _LOG(f"residency: {self.stats['ticks']} ticks, {self.stats['promoted']} promotions, cache-row picks {hp}, "
                 f"host-row picks {mp} (misses {mp / max(hp + mp, 1) * 100:.1f}% of non-static picks), "
                 f"{len(self.filling)} copies in flight, tick CPU {self.stats['cpu'] / max(self.stats['ticks'], 1) * 1e3:.3f} ms "
                 f"mean / {self.stats['cpu_max'] * 1e3:.2f} max; distinct host experts per layer-step 0..8+: "
                 f"{'/'.join(str(int(x)) for x in self.hist)}, est. miss time {self.miss_us / max(self.miss_steps, 1) / 1e3:.2f} "
                 f"ms per step ({self.miss_steps} steps)")
            self.hist[:] = 0
            self.miss_us, self.miss_steps = 0.0, 0
            self.stats.update(ticks=0, promoted=0, missed_picks=0, hit_picks=0, t=time.time(), cpu=0.0, cpu_max=0.0)


_R = [None]


def tick():
    if os.environ.get("DERPSEEK_RESIDENCY", "1") != "1" or not exact.LAYERS or torch.cuda.is_current_stream_capturing():
        return
    if _R[0] is None:
        _R[0] = Residency()
    _R[0].tick()
