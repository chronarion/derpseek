"""Derpseek prompt staging, standalone: a MegaMoE call whose host-mapped rows were copied into an HBM staging buffer
(control words 2-9 pointing at it, host lane off, picks of those experts sent to their staging positions) must equal
the call that reads the same rows from pinned host memory, and the all-HBM call, bit for bit. The staged rows are a
scattered subset of the host rows, as in serving (the other host rows' experts sit in cache rows). Patched fork
headers from $FORK, as tests/host_tier_test.py."""
import os, shutil, torch
import torch.distributed as dist
from vllm.utils.deep_gemm import _import_deep_gemm
dg = _import_deep_gemm()
FORK = os.environ["FORK"]
root = "/tmp/drp-root"
shutil.rmtree(root, ignore_errors=True)
shutil.copytree(os.path.join(os.path.dirname(dg.__file__), "include"), os.path.join(root, "include"))
for rel in ("include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh", "include/deep_gemm/scheduler/mega_moe.cuh"):
    shutil.copyfile(os.path.join(FORK, rel), os.path.join(root, rel))
dg._C.init(root)
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29534")
dist.init_process_group("nccl", rank=0, world_size=1)
group = dist.group.WORLD
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import DeepGemmSm100MegaMoEBackend
from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import prepare_megamoe_inputs
H, I, K, E, TMAX = 5120, 2304, 6, int(os.environ.get("E", "64")), int(os.environ.get("TMAX", "4096"))
H0 = E // 2          # rows [0, H0) are host rows, [H0, E) HBM rows
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(0)
w13 = torch.randint(-128, 128, (E, 2 * I, H // 2), dtype=torch.int8, device=dev, generator=g)
s13 = torch.randint(118, 124, (E, 2 * I, H // 32), dtype=torch.int32, device=dev, generator=g).to(torch.uint8)
w2 = torch.randint(-128, 128, (E, H, I // 2), dtype=torch.int8, device=dev, generator=g)
s2 = torch.randint(118, 124, (E, H, I // 32), dtype=torch.int32, device=dev, generator=g).to(torch.uint8)
# SF6=1 (Fable round 23a v5): the scale banks of the split call are 6-bit coded (HBM rows, host rows; fork
# derpseek_sf6); the all-HBM reference stays uncoded, so EQUAL means the coded rows give the same bits. Scale bytes
# over the checkpoint's whole range 117..127
SF6 = os.environ.get("SF6", "0") == "1"
if SF6:
    import sf6_util
    s13, s2 = sf6_util.scales(E, 2 * I, H // 32, g, dev), sf6_util.scales(E, H, I // 32, g, dev)
be = DeepGemmSm100MegaMoEBackend()
full = [t for pair in be.transform_weights(w13_weight=w13, w13_weight_scale=s13, w2_weight=w2, w2_weight_scale=s2,
                                           num_local_experts=E, hidden_size=H, intermediate_size=I) for t in pair]
symm = dg.get_symm_buffer_for_mega_moe(group, E, TMAX, K, H, I, num_shared_experts=0, mma_type="fp8xfp4")
TS = {torch.int8: "|i1", torch.uint8: "|u1", torch.int32: "<i4", torch.float32: "<f4", torch.float8_e4m3fn: "|u1"}


class _Arr:
    def __init__(self, t, ptr):
        self.__cuda_array_interface__ = {"shape": tuple(t.shape), "typestr": TS[t.dtype], "data": (ptr, False),
                                         "strides": tuple(s * t.element_size() for s in t.stride()), "version": 3}


def fake(t, ptr):
    out = torch.as_tensor(_Arr(t, ptr), device=dev)
    return out.view(t.dtype) if out.dtype != t.dtype else out


hbm = [t[H0:].clone() for t in full]
host = []
for t in full:
    h = torch.empty_strided(t[:H0].shape, t.stride(), dtype=t.dtype, pin_memory=True)
    h.copy_(t[:H0])
    host.append(h)
span = [fake(t, hb.data_ptr()) for t, hb in zip(full, hbm)]
ctl_buf = torch.zeros(16 + E, dtype=torch.int32, device=dev)
ctl = ctl_buf[:E]
ctl[0] = 0x53505244; ctl[1] = H0; ctl[10] = 4                  # serving default: host lane of 4 clusters
desc_tbl = torch.zeros(160 * 4 * 128, dtype=torch.uint8, device=dev)
if SF6:   # coded scale banks: HBM rows, host rows (the staging slots below copy coded host rows), control word 11, the maps
    c1, c2 = sf6_util.code(full[1]), sf6_util.code(full[3])
    hbm[1], hbm[3] = c1[H0:].clone(), c2[H0:].clone()
    host[1], host[3] = sf6_util.pinned(c1[:H0]), sf6_util.pinned(c2[:H0])
    span[1], span[3] = sf6_util.span(hbm[1], E, full[1]), sf6_util.span(hbm[3], E, full[3])
    ctl[11] = sf6_util.ctl_word()
    desc_tbl = sf6_util.desc_table(host[1], host[3], hbm[1], hbm[3], full[1], full[3])
    print(sf6_util.describe(full[1], full[3]), flush=True)
ctl.view(torch.int64)[7] = desc_tbl.data_ptr()
ctl.view(torch.int64)[1:5] = torch.tensor([h.data_ptr() for h in host], dtype=torch.int64, device=dev)
orig_words = ctl_buf[2:11].clone()

# staged subset: a scattered half of the host rows (+ the first and last), as the residency manager leaves them
staged = sorted(set(range(0, H0, 2)) | {1, H0 - 1})
U = len(staged)
pos = torch.full((H0,), -1, dtype=torch.int64, device=dev)
pos[torch.tensor(staged, device=dev)] = torch.arange(U, device=dev)
slot = [torch.empty_strided((U,) + tuple(h.shape[1:]), h.stride(), dtype=h.dtype, device=dev) for h in host]
for s, h in zip(slot, host):
    for u, j in enumerate(staged):
        s[u].copy_(h[j], non_blocking=True)
torch.cuda.synchronize()
words = torch.zeros(9, dtype=torch.int32)
words[:8].view(torch.int64)[:] = torch.tensor([s.data_ptr() for s in slot], dtype=torch.int64)
words = words.to(dev)                                           # word 10 (host lane) = 0 for staged calls
staged_t = torch.tensor(staged, device=dev)


def route(T):
    """Every token: picks over HBM rows and the staged host rows only (the others are cached elsewhere)."""
    pool = torch.cat([staged_t, torch.arange(H0, E, device=dev)])
    ids = torch.stack([pool[torch.randperm(len(pool), device=dev, generator=g)[:K]] for _ in range(T)])
    w = torch.softmax(torch.randn(T, K, device=dev, generator=g), -1).float()
    return ids.long(), w


def call(x, ids, w, mode, T):
    prepare_megamoe_inputs(x, w, ids, symm.x[:T], symm.x_sf[:T], symm.topk_idx[:T], symm.topk_weights[:T])
    y = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
    if mode == "hbm":
        dg.fp8_fp4_mega_moe(y, (full[0], full[1]), (full[2], full[3]), symm, activation_clamp=10.0, fast_math=True)
    else:
        dg.fp8_fp4_mega_moe(y, (span[0], span[1]), (span[2], span[3]), symm, cumulative_local_expert_recv_stats=ctl,
                            activation_clamp=10.0, fast_math=True)
    return y


rows_dev = staged_t.long()
cnt = ctl_buf[16:16 + E]


def staged_call(x, ids, w, T):
    host_pick = ids < H0
    ids_s = torch.where(host_pick, pos[ids.clamp(0, H0 - 1)], ids)      # staged position for host rows, else the row
    before = cnt[:U].clone()
    ctl_buf[2:11].copy_(words)
    y = call(x, ids_s, w, "split", T)
    ctl_buf[2:11].copy_(orig_words)
    d = cnt[:U] - before                                                 # as prefill.py: credit the host rows
    cnt[:U].copy_(before)
    cnt.index_add_(0, rows_dev, d)
    return y


ok = True
for T in (int(t) for t in os.environ.get("TS", "1,8,64,256,1024,4096").split(",")):
    x = torch.randn(T, H, dtype=torch.bfloat16, device=dev, generator=g)
    ids, w = route(T)
    ya = call(x, ids, w, "hbm", T)
    yb = call(x, ids, w, "split", T)
    yc = staged_call(x, ids, w, T)
    torch.cuda.synchronize()
    eq_b, eq_c = torch.equal(ya, yb), torch.equal(ya, yc)
    ok &= eq_b and eq_c
    ts = []
    for fn in (lambda: call(x, ids, w, "hbm", T), lambda: call(x, ids, w, "split", T), lambda: staged_call(x, ids, w, T)):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(2): fn()
        st.record()
        for _ in range(5): fn()
        en.record(); torch.cuda.synchronize(); ts.append(st.elapsed_time(en) / 5 * 1000)
    dh = len(set(ids[ids < H0].tolist()))
    print(f"T={T:5d} ({dh:2d} host experts picked): host rows {'EQUAL' if eq_b else 'DIFFER'}, staged "
          f"{'EQUAL' if eq_c else 'DIFFER'} vs all-HBM;  all-HBM {ts[0]:8.1f} us, host rows {ts[1]:8.1f} us, "
          f"staged {ts[2]:8.1f} us", flush=True)
# per-row counts after a staged call: exactly its picks, on the original rows (host row j, not its staging position)
x = torch.randn(256, H, dtype=torch.bfloat16, device=dev, generator=g); ids, w = route(256)
c0 = cnt.clone(); staged_call(x, ids, w, 256); torch.cuda.synchronize()
cnt_ok = torch.equal((cnt - c0).long(), torch.bincount(ids.flatten(), minlength=E))
ok &= cnt_ok
print(f"per-row counts after a staged call: {'EXACT' if cnt_ok else 'WRONG'}")
# the control words come back after a staged call: a following host-row call is still exact
x = torch.randn(64, H, dtype=torch.bfloat16, device=dev, generator=g); ids, w = route(64)
again = torch.equal(call(x, ids, w, "hbm", 64), call(x, ids, w, "split", 64))
ok &= again
print(f"host-row call after staged calls: {'EQUAL' if again else 'DIFFER'}")
print("RESULT", "PASS" if ok else "FAIL")
