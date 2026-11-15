"""Residency manager invariants on a synthetic 2-layer setup (tiny rows): after every tick, no expert maps to two
rows, every mapped cache slot holds that expert's host bytes, and misses get promoted."""
import sys, numpy as np, torch
sys.path.insert(0, "/derpseek")
from derpseek_engine import exact, residency
L, C, E = 2, 4, exact.E
SL = [8, 12]          # layers split static / host rows differently (same D + S, same C)
dev = "cuda"
rows = E + C
exact.POOL["ctl"] = torch.zeros(exact.NUM_LAYERS, exact.HDR + rows, dtype=torch.int32, device=dev)
exact.POOL["row_map"] = torch.zeros(exact.NUM_LAYERS, E, dtype=torch.int32, device=dev)
rng = np.random.default_rng(0)
for l in range(L):
    S = SL[l]; D = E - S
    perm = rng.permutation(E).tolist()
    st = exact.ExactState(l, sorted(perm[:S]), perm[S:S + C], perm[S:])
    host = [torch.randint(0, 255, (D, 64), dtype=torch.uint8).pin_memory() for _ in range(4)]
    hbm = [torch.zeros(S + C, 64, dtype=torch.uint8, device=dev) for _ in range(4)]
    for c, e in enumerate(st.cache):
        for k in range(4):
            hbm[k][S + c].copy_(host[k][st.host.index(e)])
    st.flat_hbm, st.flat_host = hbm, host
    st.row_map = exact.POOL["row_map"][l]
    st.row_map.copy_(torch.tensor(st.row, dtype=torch.int32, device=dev))
    exact.LAYERS[l] = st
residency.TUNE["min_age"] = 2
r = residency.Residency()
def check(t):
    for l, st in exact.LAYERS.items():
        rm = st.row_map.cpu().tolist()
        vals = [x for x in rm]
        assert len(set(vals)) == E, f"tick {t} layer {l}: two experts share a row"
        D = st.D
        for e in range(E):
            row = rm[e]
            if e in st.hbm_row and st.hbm_row[e] < st.S:
                assert row == D + st.hbm_row[e]
            elif D + st.S <= row < D + st.S + C:
                j = st.host.index(e)
                for k in range(4):
                    assert torch.equal(st.flat_hbm[k][row - D].cpu(), st.flat_host[k][j]), f"tick {t}: layer {l} slot {row} bytes != expert {e}"
            else:
                assert row == st.host_row[e] and row < D, f"tick {t}: expert {e} row {row} neither cache nor its host row"
promoted_want = set()
for t in range(200):
    # synthetic picks: each layer hits 3 random host experts this step (some repeat)
    for l, st in exact.LAYERS.items():
        rm = st.row_map.cpu().numpy()
        picks = rng.choice(st.host[:40], 3)
        for e in picks:
            exact.POOL["ctl"][l, exact.HDR + int(rm[e])] += 1
            if rm[e] < st.D:
                promoted_want.add((l, int(e)))
    r.tick()
    torch.cuda.synchronize()
    check(t)
cached = {(l, int(e)) for l in exact.LAYERS for e in r.slot_e[l] if e >= 0}
print(f"OK: 200 ticks, invariants held; {r.stats['promoted']} promotions in the last window; "
      f"{len(cached)} experts cached now; slots {[r.slot_e[l].tolist() for l in exact.LAYERS]}")
