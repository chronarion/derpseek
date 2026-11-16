"""Derpseek exact Grace tier, standalone: one MegaMoE call with rows < D in pinned host memory (HBM rows after them) must equal the
all-HBM call bit for bit (patched fork headers from $FORK). Also times host-row experts."""
import os, shutil, sys, torch
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
if os.environ.get("NSMS"):
    dg.set_num_sms(int(os.environ["NSMS"]))
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29533")
dist.init_process_group("nccl", rank=0, world_size=1)
group = dist.group.WORLD
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import DeepGemmSm100MegaMoEBackend
from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import prepare_megamoe_inputs
H, I, K, E, TMAX = 5120, 2304, 6, int(os.environ.get("E", "64")), 1024
H0 = E // 2   # D: rows [0, H0) are host rows, [H0, E) HBM rows
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
(l1w, l1s), (l2w, l2s) = be.transform_weights(w13_weight=w13, w13_weight_scale=s13, w2_weight=w2, w2_weight_scale=s2,
                                              num_local_experts=E, hidden_size=H, intermediate_size=I)
for n, t in (("l1w", l1w), ("l1s", l1s), ("l2w", l2w), ("l2s", l2s)):
    print(f"{n}: shape {tuple(t.shape)} stride {t.stride()} dtype {t.dtype} contiguous {t.is_contiguous()}")
    assert t.shape[0] == E and t.stride(0) == t[0].numel() and t[0].is_contiguous() or t.stride(0) == t[0].numel(), n   # expert-major, dense per expert
SH = int(os.environ.get("SHARED", "1"))
if SH:
    gu = (torch.randn(2 * I, H, device=dev, generator=g) * 0.05).to(torch.float8_e4m3fn)
    dn = (torch.randn(H, I, device=dev, generator=g) * 0.05).to(torch.float8_e4m3fn)
    gus = torch.full((H // 128, 2 * I), 0x7F7F7F7F, dtype=torch.int32, device=dev).t()   # 2^0 per 1x32 group, MN-major
    dns = torch.full((I // 128, H), 0x7F7F7F7F, dtype=torch.int32, device=dev).t()
    sh_l1, sh_l2 = dg.transform_weights_for_mega_moe((gu, gus), (dn, dns))
symm = dg.get_symm_buffer_for_mega_moe(group, E, TMAX, K, H, I, num_shared_experts=SH, mma_type="fp8xfp4")
TS = {torch.int8: "|i1", torch.uint8: "|u1", torch.int32: "<i4", torch.float32: "<f4", torch.float8_e4m3fn: "|u1"}
class _Arr:
    def __init__(self, t, ptr):
        self.__cuda_array_interface__ = {"shape": tuple(t.shape), "typestr": TS[t.dtype], "data": (ptr, False),
                                         "strides": tuple(s * t.element_size() for s in t.stride()), "version": 3}
def fake(t, ptr):
    out = torch.as_tensor(_Arr(t, ptr), device=dev)
    return out.view(t.dtype) if out.dtype != t.dtype else out
hbm = [t[H0:].clone() for t in (l1w, l1s, l2w, l2s)]
host = []
for t in (l1w, l1s, l2w, l2s):
    h = torch.empty_strided(t[:H0].shape, t.stride(), dtype=t.dtype, pin_memory=True); h.copy_(t[:H0]); host.append(h)
    assert h.stride() == t.stride()
fl1w, fl1s, fl2w, fl2s = (fake(t, hb.data_ptr()) for t, hb in zip((l1w, l1s, l2w, l2s), hbm))
ctl_buf = torch.zeros(16 + E, dtype=torch.int32, device=dev)   # header 16 words + per-row counts
ctl = ctl_buf[:E]                                              # the launcher checks numel == num experts
ctl[0] = 0x53505244; ctl[1] = H0; ctl[10] = int(os.environ.get("LANE", "0"))
desc_tbl = torch.zeros(160 * 4 * 128, dtype=torch.uint8, device=dev)   # per-layer clone table (ctl words 14-15)
if SF6:   # coded scale banks for the HBM rows and the host rows, control word 11, the coded scale maps after the clones
    c1, c2 = sf6_util.code(l1s), sf6_util.code(l2s)
    hbm[1], hbm[3] = c1[H0:].clone(), c2[H0:].clone()
    host[1], host[3] = sf6_util.pinned(c1[:H0]), sf6_util.pinned(c2[:H0])
    fl1s, fl2s = sf6_util.span(hbm[1], E, l1s), sf6_util.span(hbm[3], E, l2s)
    ctl[11] = sf6_util.ctl_word()
    desc_tbl = sf6_util.desc_table(host[1], host[3], hbm[1], hbm[3], l1s, l2s)
    print(sf6_util.describe(l1s, l2s), flush=True)
ctl.view(torch.int64)[7] = desc_tbl.data_ptr()
ctl.view(torch.int64)[1:5] = torch.tensor([h.data_ptr() for h in host], dtype=torch.int64, device=dev)
def route_heavy(T):
    """all picks but the last of each token on distinct host rows (HBM tasks scarce: the fair-gate deadlock shape)"""
    ids = torch.stack([H0 + torch.randperm(E - H0, device=dev, generator=g)[:K] for _ in range(T)])
    hr = torch.randperm(H0, device=dev, generator=g)
    for t in range(T):
        for k in range(K - 1):
            ids[t, k] = hr[(t * (K - 1) + k) % H0]
    w = torch.softmax(torch.randn(T, K, device=dev, generator=g), -1).float()
    return ids.long(), w
def route(T, n_host_tokens):
    ids = torch.stack([H0 + torch.randperm(E - H0, device=dev, generator=g)[:K] for _ in range(T)])   # HBM rows
    for t in range(min(n_host_tokens, T)):
        ids[t, 0] = t % H0                            # this token's first pick is a host-row expert
    w = torch.softmax(torch.randn(T, K, device=dev, generator=g), -1).float()
    return ids.long(), w
def call(x, ids, w, split, T):
    kw = {}
    if SH:
        sbm = int(dg.get_block_m_for_mega_moe(1, E, TMAX, T, K, "fp8xfp4"))
        prepare_megamoe_inputs(x, w, ids, symm.x[:T], symm.x_sf[:T], symm.topk_idx[:T], symm.topk_weights[:T],
                               shared_x_sf=symm.shared_l1_acts_sf, shared_block_m=sbm)
        kw = dict(shared_l1_weights=sh_l1, shared_l2_weights=sh_l2)
    else:
        prepare_megamoe_inputs(x, w, ids, symm.x[:T], symm.x_sf[:T], symm.topk_idx[:T], symm.topk_weights[:T])
    y = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
    if split:
        dg.fp8_fp4_mega_moe(y, (fl1w, fl1s), (fl2w, fl2s), symm, cumulative_local_expert_recv_stats=ctl,
                            activation_clamp=10.0, fast_math=True, **kw)
    else:
        dg.fp8_fp4_mega_moe(y, (l1w, l1s), (l2w, l2s), symm, activation_clamp=10.0, fast_math=True, **kw)
    return y
ok = True
for T in (int(t) for t in os.environ.get('TS', '1,6,24,48,64,256,1024').split(',')):
    for nh in sorted(set(int(v) for v in os.environ.get("NHS", "0,1,2,4,8").split(",") if int(v) <= T)):
        x = torch.randn(T, H, dtype=torch.bfloat16, device=dev, generator=g)
        ids, w = route(T, nh)
        print(f"T={T} nh={nh} ...", end=" ", flush=True)
        ya = call(x, ids, w, False, T); yb = call(x, ids, w, True, T); torch.cuda.synchronize()
        same = torch.equal(ya, yb)
        ok &= same
        ts = []
        for split in (False, True):
            st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for _ in range(3): call(x, ids, w, split, T)
            st.record()
            for _ in range(10): call(x, ids, w, split, T)
            en.record(); torch.cuda.synchronize(); ts.append(st.elapsed_time(en) / 10 * 1000)
        hosts = int((ids < H0).sum()); dh = len(set(ids[ids < H0].tolist()))
        print(f"T={T:5d} host picks {hosts:4d} ({dh:2d} experts): bitwise {'EQUAL' if same else 'DIFFER'}  "
              f"all-HBM {ts[0]:8.1f} us  split {ts[1]:8.1f} us (incl. staging)", flush=True)
if os.environ.get("HEAVY", "0") == "1":
    for T in (1, 2, 3, 4, 8, 16):
        x = torch.randn(T, H, dtype=torch.bfloat16, device=dev, generator=g)
        ids, w = route_heavy(T)
        print(f"HEAVY T={T} ...", end=" ", flush=True)
        ya = call(x, ids, w, False, T); yb = call(x, ids, w, True, T); torch.cuda.synchronize()
        same = torch.equal(ya, yb); ok &= same
        print(f"host experts {len(set(ids[ids < H0].tolist()))}, HBM experts {len(set(ids[ids >= H0].tolist()))}: "
              f"bitwise {'EQUAL' if same else 'DIFFER'}", flush=True)
# per-row counts in ctl[16:]: one split call on a fresh routing must add exactly its picks per row
x = torch.randn(48, H, dtype=torch.bfloat16, device=dev, generator=g); ids, w = route(48, 8)
before = ctl_buf[16:].clone(); call(x, ids, w, True, 48); torch.cuda.synchronize()
got = (ctl_buf[16:] - before).long(); want = torch.bincount(ids.flatten(), minlength=E)
cnt_ok = torch.equal(got, want); ok &= cnt_ok
print(f"per-row counts in the control block: {'EXACT' if cnt_ok else 'WRONG'} (sum {int(got.sum())}, want {int(want.sum())})")
print("RESULT", "PASS" if ok else "FAIL")
